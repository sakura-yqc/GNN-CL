import os
import glob
import random
import argparse
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# === Import model and data modules ===
from model import (
    SharedMLP, NodeEncoder, GraphMLP, MonotonicGateNet, MonotonicJumpNet,
    EdgeFusionLayer, build_temp_net, train_batch_joint
)
from data_utils import process_one_frame

# ==== Argument parsing for main training script ====
def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN-CL model with hyperparameter search.")

    parser.add_argument("--train_nodes_dir", type=str, required=False,
        default="../../data/processed/highD/nodes/train",
        help="Training node CSV dir")
    parser.add_argument("--train_adj_dir", type=str, required=False,
        default="../../data/processed/highD/adj/train",
        help="Training adj CSV dir")
    parser.add_argument("--val_nodes_dir", type=str, required=False,
        default="../../data/processed/highD/nodes/val",
        help="Validation node CSV dir")
    parser.add_argument("--val_adj_dir", type=str, required=False,
        default="../../data/processed/highD/adj/val",
        help="Validation adj CSV dir")
    parser.add_argument("--feature_columns", nargs="+",
        default=["x", "y", "vx", "vy", "sie", "heading_angle"],
        help="Node feature column names")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per setting")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gnn_dim", type=int, default=64, help="GNN hidden dim (fixed)")
    parser.add_argument("--out_dim_list", type=int, nargs="+", default=[16, 32, 64, 128],
        help="Embedding dim search space")
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01, 0.001, 0.0001],
        help="Learning rate search space")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs for contrastive learning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_ckpt", type=str, default="../../results/gnncl_best.pth",
        help="Path to save best model checkpoint")
    parser.add_argument("--sample_num", type=int, default=0,
        help="If >0, randomly sample N graphs for training")

    return parser.parse_args()



# ==== Collects processed graph data objects for model training/validation ====
def collect_data_list(nodes_dir, adj_dir, feature_columns, adj_types, sample_num=0, seed=42):
    node_files = sorted(glob.glob(os.path.join(nodes_dir, "*.csv")))
    if sample_num > 0:
        random.seed(seed)
        node_files = random.sample(node_files, sample_num)
    data_list = []
    for node_path in node_files:
        suffix = os.path.basename(node_path).replace("nodes_", "")
        adj_paths = [os.path.join(adj_dir, f"adj_{t}_{suffix}") for t in adj_types]
        if not all(os.path.exists(p) for p in adj_paths):
            print(f"Missing adj file, skip: {suffix}")
            continue
        data = process_one_frame(node_path, adj_paths, feature_columns)
        data_list.append(data)
    return data_list

# ==== Validation step: computes average loss over validation set ====
@torch.no_grad()
def validate(val_loader, device, model_objs):
    edge_fusion_layer, temp_net, gate_net, jump_net, layer_encoders, fused_encoder, diffuse_encoder, graph_mlp = model_objs
    total_loss, total_struct, total_sem, total_contrast, n_batch = 0., 0., 0., 0., 0
    for batch in val_loader:
        batch = batch.to(device)
        loss, struct_loss, sem_loss, contrast_loss = train_batch_joint(
            batch, edge_fusion_layer, temp_net, gate_net, jump_net,
            layer_encoders, fused_encoder, diffuse_encoder, graph_mlp, None, device, warmup=False, eval_mode=True
        )
        total_loss += float(loss)
        total_struct += float(struct_loss)
        total_sem += float(sem_loss)
        total_contrast += float(contrast_loss)
        n_batch += 1
    return total_loss / n_batch if n_batch > 0 else float('inf')

# ==== Main training, validation, and grid search pipeline ====
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adj_types = ['angle', 'latv', 'lond', 'latd', 'lonv']

    print("Loading training data ...")
    train_data = collect_data_list(args.train_nodes_dir, args.train_adj_dir, args.feature_columns, adj_types, args.sample_num, args.seed)
    print("Loading validation data ...")
    val_data = collect_data_list(args.val_nodes_dir, args.val_adj_dir, args.feature_columns, adj_types, 0, args.seed)
    if not train_data or not val_data:
        print("No valid data found! Please check your paths.")
        return

    node_feat_dim = train_data[0].x.shape[1]
    complexity_dim = train_data[0].complexity.shape[1]
    gnn_dim = args.gnn_dim

    best_val_loss = float('inf')
    best_setting = None
    best_model_states = None

    # ==== Grid search over output embedding dim and learning rate ====
    for out_dim in args.out_dim_list:
        for lr in args.lr_list:
            print(f"\n[GridSearch] Try out_dim={out_dim}, lr={lr:.1e}")
            shared_mlp = SharedMLP(gnn_dim, out_dim).to(device)
            layer_encoders = torch.nn.ModuleList([
                NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device) for _ in range(5)
            ])
            fused_encoder = NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device)
            diffuse_encoder = NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device)
            edge_fusion_layer = EdgeFusionLayer(num_layers=5).to(device)
            temp_net = build_temp_net(complexity_dim).to(device)
            gate_net = MonotonicGateNet(complexity_dim, hidden_dim=8).to(device)
            jump_net = MonotonicJumpNet(complexity_dim, hidden_dim=8).to(device)
            graph_mlp = GraphMLP(out_dim, out_dim, hidden_dim=64).to(device)

            params = list(edge_fusion_layer.parameters()) + \
                     list(temp_net.parameters()) + \
                     list(gate_net.parameters()) + \
                     list(layer_encoders.parameters()) + \
                     list(fused_encoder.parameters()) + \
                     list(diffuse_encoder.parameters()) + \
                     list(jump_net.parameters()) + \
                     list(graph_mlp.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

            # ==== Training loop ====
            for epoch in range(1, args.epochs + 1):
                warmup = epoch <= args.warmup_epochs
                for batch in train_loader:
                    batch = batch.to(device)
                    train_batch_joint(
                        batch, edge_fusion_layer, temp_net, gate_net, jump_net,
                        layer_encoders, fused_encoder, diffuse_encoder, graph_mlp,
                        optimizer, device, warmup
                    )
            # ==== Validation & checkpoint ====
            model_objs = [
                edge_fusion_layer, temp_net, gate_net, jump_net,
                layer_encoders, fused_encoder, diffuse_encoder, graph_mlp
            ]
            val_loss = validate(val_loader, device, model_objs)
            print(f"[Val] out_dim={out_dim}, lr={lr:.1e}, val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_setting = dict(out_dim=out_dim, lr=lr)
                best_model_states = {
                    'layer_encoders': layer_encoders.state_dict(),
                    'fused_encoder': fused_encoder.state_dict(),
                    'diffuse_encoder': diffuse_encoder.state_dict(),
                    'edge_fusion_layer': edge_fusion_layer.state_dict(),
                    'temp_net': temp_net.state_dict(),
                    'gate_net': gate_net.state_dict(),
                    'jump_net': jump_net.state_dict(),
                    'graph_mlp': graph_mlp.state_dict(),
                }
    # ==== Save best model checkpoint ====
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)  # 自动创建results目录
    torch.save(best_model_states, args.save_ckpt)
    print(f"Best model saved to {args.save_ckpt}")


if __name__ == "__main__":
    main()
