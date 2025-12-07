import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import monotonicnetworks as lmn
from data_utils import (
    process_one_frame,
    row_normalize_edge_weight,
    normalize_edge_weight,
    build_random_walk_matrix,
    occupation_time_ppr,
    sparse_from_matrix,
    get_graph_embedding,
    recon_loss,
    make_contrast_samples,
    contrastive_dcl_loss
)

# ==== Shared MLP projection head for node/graph representations ====
class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# ==== Node Encoder: GCN + MLP projection ====
class NodeEncoder(nn.Module):
    def __init__(self, in_dim, gnn_dim, out_dim, shared_mlp):
        super().__init__()
        self.gnn = GCNConv(in_dim, gnn_dim)
        self.shared_mlp = shared_mlp
    def forward(self, x, edge_index, edge_weight=None):
        h = self.gnn(x, edge_index, edge_weight=edge_weight)
        h = self.shared_mlp(h)
        return h

class GraphMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# ==== Monotonic Gating Network for edge fusion (complexity-driven) ====
class MonotonicGateNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            lmn.LipschitzLinear(in_dim, hidden_dim, kind="one-inf"),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(hidden_dim, 1, kind="one-inf"),
            nn.Sigmoid()
        )
    def forward(self, complexity):
        return self.net(complexity).squeeze(-1)

# ==== Monotonic Network for jump (restart) probability in PPR diffusion ====
class MonotonicJumpNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=8, min_alpha=0.2, max_alpha=0.8):
        super().__init__()
        mlp = nn.Sequential(
            lmn.LipschitzLinear(in_dim, hidden_dim, kind="one-inf"),
            lmn.GroupSort(2),
            lmn.LipschitzLinear(hidden_dim, 1, kind="one-inf")
        )
        self.monotonic_nn = lmn.MonotonicWrapper(mlp, monotonic_constraints=[-1]*in_dim)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, complexity):
        prob = self.monotonic_nn(complexity)
        alpha = torch.sigmoid(prob)
        alpha = torch.clamp(alpha, min=self.min_alpha, max=self.max_alpha)
        return alpha

# ==== Edge Fusion Layer: implements edge-level risk consensus attention ====
# This module combines multilayer edge attributes (e.g., multi-risk) into a single fused edge weight,
# explicitly modeling consensus, suppression, and enhancement as described in the main text.
class EdgeFusionLayer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.b = nn.Parameter(torch.ones(1))
        self.c = nn.Parameter(torch.ones(1))
        self.mlp = nn.Linear(num_layers, num_layers)
    def forward(self, edge_attr):
        r_mean = edge_attr.mean(dim=1, keepdim=True)
        e = edge_attr + self.b * r_mean - self.c * (edge_attr - r_mean) ** 2
        z = self.mlp(e)
        alpha = F.softmax(z, dim=1)
        fused_edge_weight = (alpha * edge_attr).sum(dim=1)
        return fused_edge_weight

# ==== Build complexity-driven monotonic temperature network (for attention scaling) ====
def build_temp_net(in_dim):
    mlp = nn.Sequential(
        lmn.LipschitzLinear(in_dim, 16, kind="one-inf"),
        lmn.GroupSort(2),
        lmn.LipschitzLinear(16, 8, kind="one-inf"),
        lmn.GroupSort(2),
        lmn.LipschitzLinear(8, 1, kind="one-inf"),
    )
    temp_net = lmn.MonotonicWrapper(mlp, monotonic_constraints=[-1] * in_dim)
    return temp_net

def map_temperature_to_range(temp_raw, t_min=0.01, t_max=3.0):
    return t_min + (t_max - t_min) * torch.sigmoid(temp_raw)

# ==== Main joint training step for a single batch ====
# Implements both warmup and joint phase (fusion structure, semantic, and diffusion-contrastive loss).
# Args:
#   batch: Data object for one graph
#   edge_fusion_layer: edge fusion module
#   temp_net: monotonic temperature net
#   gate_net: monotonic gating net
#   jump_net: monotonic jump probability net (for occupation-time PPR)
#   layer_encoders: list of NodeEncoder (one per interaction graph layer)
#   fused_encoder: NodeEncoder for fused graph
#   diffuse_encoder: NodeEncoder for diffusion graph
#   graph_mlp: graph-level readout
#   optimizer: torch optimizer
#   device: CUDA or CPU
#   warmup: bool, if True use structure+semantic, else structure+semantic+diffusion
# Returns:
#   loss, struct_loss, sem_loss, contrast_loss (all float)
def train_batch_joint(
    batch, edge_fusion_layer, temp_net, gate_net, jump_net,
    layer_encoders, fused_encoder, diffuse_encoder, graph_mlp, optimizer, device,
    warmup, num_neg=5, feature_noise_std=0.2, mask_ratio=0.3, edge_drop_rate=0.2
):
    from data_utils import process_batch, recon_loss, make_contrast_samples, contrastive_dcl_loss, build_random_walk_matrix, occupation_time_ppr, sparse_from_matrix, normalize_edge_weight, get_graph_embedding
    batch = batch.to(device)
    N = batch.x.shape[0]
    # --- Fusion graph edge weights and structure loss
    fused_final, _, _, _, _, _, _ = process_batch(batch, edge_fusion_layer, temp_net, gate_net)
    struct_loss = recon_loss(fused_final, batch.edge_attr)
    # --- Node embeddings from all base interaction layers
    h_layers = [
        layer_encoders[l](batch.x, batch.edge_index, edge_weight=batch.edge_attr[:, l].to(device))
        for l in range(5)
    ]
    h_fused = fused_encoder(batch.x, batch.edge_index, edge_weight=fused_final)
    if warmup:
        # --- Semantic alignment: fuse vs each risk layer, node-wise contrast
        sem_loss = 0.
        for h_layer in h_layers:
            pos_score, neg_scores = make_contrast_samples(
                h_fused, h_layer, h_layer, num_neg=num_neg,
                feature_noise_std=feature_noise_std, mask_ratio=mask_ratio
            )
            sem_loss += contrastive_dcl_loss(pos_score, neg_scores)
        sem_loss = sem_loss / len(h_layers)
        loss = struct_loss + sem_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss), float(struct_loss), float(sem_loss), 0.
    # --- Compute occupation-time PPR diffusion (node-adaptive restart prob)
    alpha = jump_net(batch.complexity).squeeze(-1)
    norm_edge_weight = normalize_edge_weight(batch.edge_index, fused_final, N)
    P = build_random_walk_matrix(batch.edge_index, norm_edge_weight, N)
    S = occupation_time_ppr(P, alpha)
    edge_index_diff, edge_weight_diff = sparse_from_matrix(S, threshold=0)
    x_diff = batch.x
    # --- Reverse edge direction for node-to-graph alignment
    edge_index_fused_rev = batch.edge_index[[1,0],:]
    h_fused, g_fused = get_graph_embedding(fused_encoder, batch.x, edge_index_fused_rev, fused_final, graph_mlp)
    h_diff, g_diff = get_graph_embedding(diffuse_encoder, x_diff, edge_index_diff, edge_weight_diff, graph_mlp)
    # --- Node-to-graph contrast: fusion node vs diffusion graph, diffusion node vs fusion graph
    pos_score1, neg_scores1 = make_contrast_samples(
        h_fused, g_diff.expand_as(h_fused), g_diff.expand_as(h_fused),
        num_neg=num_neg, feature_noise_std=feature_noise_std, mask_ratio=mask_ratio
    )
    contrast1 = contrastive_dcl_loss(pos_score1, neg_scores1)
    pos_score2, neg_scores2 = make_contrast_samples(
        h_diff, g_fused.expand_as(h_diff), g_fused.expand_as(h_diff),
        num_neg=num_neg, feature_noise_std=feature_noise_std, mask_ratio=mask_ratio
    )
    contrast2 = contrastive_dcl_loss(pos_score2, neg_scores2)
    # --- Semantic alignment as in warmup
    sem_loss = 0.
    for h_layer in h_layers:
        pos_score, neg_scores = make_contrast_samples(
            h_fused, h_layer, h_layer, num_neg=num_neg,
            feature_noise_std=feature_noise_std, mask_ratio=mask_ratio
        )
        sem_loss += contrastive_dcl_loss(pos_score, neg_scores)
    sem_loss = sem_loss / len(h_layers)
    contrast_loss = contrast1 + contrast2
    loss = struct_loss + sem_loss + contrast_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss), float(struct_loss), float(sem_loss), float(contrast_loss)
