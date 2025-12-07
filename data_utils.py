import os
import re
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

# ==== Core Data Preprocessing ====
# Load a single frame and build a torch_geometric Data object.
# Args:
#   node_csv_path: path to node feature csv
#   edge_csv_paths: list of adjacency matrix csvs for all risk layers
#   feature_columns: list of feature columns to use
# Returns:
#   Data object with x, edge_index, edge_attr, and .complexity attributes
def process_one_frame(node_csv_path, edge_csv_paths, feature_columns):
    node_df = pd.read_csv(node_csv_path, index_col=0)
    x = torch.tensor(node_df[feature_columns].values, dtype=torch.float)
    A_list = [pd.read_csv(p, index_col=0).values for p in edge_csv_paths]
    A_stack = np.stack(A_list, axis=-1)
    N = A_stack.shape[0]
    edge_index_list, edge_attr_list = [], []
    for i in range(N):
        for j in range(N):
            edge_feats = A_stack[i, j, :]
            if np.any(edge_feats > 0):
                edge_index_list.append([i, j])
                edge_attr_list.append(edge_feats.tolist())
    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    complexity = compute_node_complexity(x, edge_index, edge_attr)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.complexity = complexity
    return data

# ==== Node Complexity Calculation ====
# Compute local complexity vector for each node, used for downstream attention/gating.
def compute_node_complexity(x, edge_index, edge_attr, sie_col_idx=-2):
    N = x.shape[0]
    device = x.device
    degree = torch.zeros(N, device=device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
    mean_entropy = torch.zeros(N, device=device)
    neighbors_sie_sum = torch.zeros(N, device=device)
    sie = x[:, sie_col_idx]
    for i in range(N):
        mask = (edge_index[0] == i)
        edge_weights = edge_attr[mask]
        if edge_weights.shape[0] == 0:
            continue
        probs = edge_weights / (edge_weights.sum(dim=0, keepdim=True) + 1e-12)
        entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=0)
        mean_entropy[i] = entropy.mean().item()
        neighbors = edge_index[1][mask]
        neighbors_sie_sum[i] = sie[neighbors].sum().item() if len(neighbors) > 0 else 0.0
    temp_data = Data(edge_index=edge_index, num_nodes=N)
    G = to_networkx(temp_data, to_undirected=True)
    clustering_dict = nx.clustering(G)
    clustering_coeff = torch.tensor([clustering_dict[i] for i in range(N)], dtype=torch.float, device=device)
    embed_avg_dist = torch.zeros(N, device=device)
    complexity = torch.stack([degree, clustering_coeff, mean_entropy, neighbors_sie_sum, embed_avg_dist], dim=1)
    return complexity

def update_complexity_with_embed_dist(batch, h_fused):
    N = h_fused.shape[0]
    device = h_fused.device
    neighbors = [[] for _ in range(N)]
    src, dst = batch.edge_index
    for i in range(src.shape[0]):
        neighbors[src[i].item()].append(dst[i].item())
    embed_avg_dist = torch.zeros(N, device=device)
    for i in range(N):
        nbs = neighbors[i]
        if len(nbs) > 0:
            dists = torch.norm(h_fused[i] - h_fused[nbs], dim=1)
            embed_avg_dist[i] = dists.mean()
        else:
            embed_avg_dist[i] = 0.0
    complexity_new = batch.complexity.clone()
    complexity_new[:, -1] = embed_avg_dist
    batch.complexity = complexity_new
    return batch

def mask_features(x, mask_ratio=0.3):
    mask = torch.rand_like(x) > mask_ratio
    return x * mask.double()

def corrupt_features(x, std=0.2, mask_ratio=0.3):
    x_noisy = x + torch.randn_like(x) * std
    return mask_features(x_noisy, mask_ratio=mask_ratio)

# ==== Key Contrastive Loss Function ====
# DCL loss for contrastive node/graph embedding training.
def contrastive_dcl_loss(pos_score, neg_scores, tau=0.07):
    pos_sim = pos_score / tau
    neg_sim = neg_scores / tau
    loss_pos = -F.logsigmoid(pos_sim).mean()
    loss_neg = -F.logsigmoid(-neg_sim).mean()
    return loss_pos + loss_neg

def make_contrast_samples(anchor, pos, neg_base, num_neg=5, feature_noise_std=0.2, mask_ratio=0.3):
    pos_score = F.cosine_similarity(anchor, pos)
    neg_scores = [F.cosine_similarity(anchor, corrupt_features(neg_base, feature_noise_std, mask_ratio))
                  for _ in range(num_neg)]
    return pos_score, torch.stack(neg_scores, dim=1)

def row_normalize_edge_weight(edge_index, edge_weight, N):
    device = edge_weight.device
    out_sum = torch.zeros(N, device=device).scatter_add_(0, edge_index[0], edge_weight)
    return edge_weight / (out_sum[edge_index[0]] + 1e-8)

def normalize_edge_weight(edge_index, edge_weight, N):
    return row_normalize_edge_weight(edge_index, edge_weight, N)

def build_random_walk_matrix(edge_index, edge_weight, N):
    device = edge_weight.device
    P = torch.zeros(N, N, device=device)
    P[edge_index[0], edge_index[1]] = edge_weight
    deg = P.sum(dim=1, keepdim=True)
    return P / (deg + 1e-8)

def occupation_time_ppr(P, alpha):
    N = P.shape[0]
    device = P.device
    A = torch.diag(alpha)
    I = torch.eye(N, device=device)
    S = torch.linalg.inv(I - A @ P)
    return S

def sparse_from_matrix(S, threshold=0):
    device = S.device
    edge_index, edge_weight = [], []
    N = S.shape[0]
    for i in range(N):
        for j in range(N):
            if S[i, j] > threshold and i != j:
                edge_index.append([j, i])  # reverse direction
                edge_weight.append(float(S[i, j].item()))
    if not edge_index:
        return torch.zeros((2,0), dtype=torch.long, device=device), torch.zeros(0, device=device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T
    edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)
    return edge_index, edge_weight

# ==== Dataset Path Collection ====
# Scans node/adj csv directories and pairs each frame's node/adjacencies.
def collect_dataset_paths(node_dir, adj_dir, sample_num=0, random_seed=42):
    node_files = [f for f in os.listdir(node_dir) if f.startswith('nodes_case') and f.endswith('.csv')]
    dataset_pairs = []
    for fname in node_files:
        match = re.match(r'nodes_case([\d\.]+)_win(\d+)_frame(\d+)\.csv', fname)
        if match:
            case_id, win, frame = match.groups()
            node_csv_path = os.path.join(node_dir, fname)
            adj_csv_paths = [os.path.join(adj_dir, f'adj_{layer}_case{case_id}_win{win}_frame{frame}.csv')
                             for layer in ["angle", "latv", "lond", "latd", "lonv"]]
            if all(os.path.exists(p) for p in adj_csv_paths):
                dataset_pairs.append((node_csv_path, adj_csv_paths))
    if sample_num > 0 and len(dataset_pairs) > sample_num:
        random.seed(random_seed)
        dataset_pairs = random.sample(dataset_pairs, sample_num)
    return dataset_pairs

# ==== Plot Training Losses ====
def plot_loss_curve(loss_curve, struct_curve, sem_curve, contrast_curve, save_path=None):
    plt.figure()
    plt.plot(loss_curve, label="Total")
    plt.plot(struct_curve, label="Struct")
    plt.plot(sem_curve, label="SemAlign")
    plt.plot(contrast_curve, label="Contrast")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curve")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ==== Main Embedding Calculation ====
# Returns node and graph embedding for a single graph.
def get_graph_embedding(encoder, x, edge_index, edge_weight, graph_mlp, batch=None):
    h = encoder(x, edge_index, edge_weight=edge_weight)
    device = h.device
    if batch is None:
        batch = torch.zeros(h.shape[0], dtype=torch.long, device=device)
    g_pre = global_mean_pool(h, batch)
    g = graph_mlp(g_pre)
    return h, g

# ==== Fusion and Losses ====
def recon_loss(fused_final, edge_attr):
    return ((fused_final.unsqueeze(1) - edge_attr) ** 2).sum()

def process_batch(batch, edge_fusion_layer, temp_net, gate_net):
    fused_edge_weight = edge_fusion_layer(batch.edge_attr)
    temp_raw = temp_net(batch.complexity).squeeze(-1)
    temperature = map_temperature_to_range(temp_raw)
    src = batch.edge_index[0]
    temp_per_edge = temperature[src]
    logits = fused_edge_weight / temp_per_edge
    alpha = softmax(logits, src)    # grouped by src
    fused_weight_with_temp = alpha * fused_edge_weight
    gate = gate_net(batch.complexity)
    gate_per_edge = gate[src]
    fused_final = (1 - gate_per_edge) * fused_edge_weight + gate_per_edge * fused_weight_with_temp
    fused_final = row_normalize_edge_weight(batch.edge_index, fused_final, batch.x.size(0))
    return fused_final, fused_edge_weight, temperature, gate, src, alpha, fused_weight_with_temp

def map_temperature_to_range(temp_raw, t_min=0.01, t_max=3.0):
    return t_min + (t_max - t_min) * torch.sigmoid(temp_raw)

# ==== Key Analysis Utility ====
# Compute node embedding after removing a given edge .
def get_embedding_with_specific_edge_removed(
    x, edge_index, edge_attr, complexity,
    fused_final, edge_index_fused, edge_index_diff, edge_weight_diff,
    fused_encoder, diffuse_encoder,
    target_idx, remove_fused=None, remove_diff=None, device="cuda"
):
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    complexity = complexity.to(device)
    fused_final = fused_final.to(device)
    edge_index_fused = edge_index_fused.to(device)
    edge_index_diff = edge_index_diff.to(device)
    edge_weight_diff = edge_weight_diff.to(device)
    mask_fused = torch.ones(edge_index_fused.shape[1], dtype=torch.bool, device=device)
    if remove_fused is not None:
        src, tgt = remove_fused
        for i in range(edge_index_fused.shape[1]):
            if edge_index_fused[0, i] == src and edge_index_fused[1, i] == tgt:
                mask_fused[i] = 0
    edge_index_fused_mod = edge_index_fused[:, mask_fused]
    fused_final_mod = fused_final[mask_fused]
    h_fused = fused_encoder(x, edge_index_fused_mod, edge_weight=fused_final_mod)
    mask_diff = torch.ones(edge_index_diff.shape[1], dtype=torch.bool, device=device)
    if remove_diff is not None:
        src, tgt = remove_diff
        for i in range(edge_index_diff.shape[1]):
            if edge_index_diff[0, i] == src and edge_index_diff[1, i] == tgt:
                mask_diff[i] = 0
    edge_index_diff_mod = edge_index_diff[:, mask_diff]
    edge_weight_diff_mod = edge_weight_diff[mask_diff]
    h_diff = diffuse_encoder(x, edge_index_diff_mod, edge_weight_diff_mod)
    return h_fused + h_diff  # [N, out_dim]
