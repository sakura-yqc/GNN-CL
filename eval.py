import os
import re
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch_geometric.utils import softmax
from torch_geometric.data import Data

from model import (
    SharedMLP, NodeEncoder, GraphMLP, MonotonicGateNet, MonotonicJumpNet,
    EdgeFusionLayer, build_temp_net,
)
from data_utils import (
    process_one_frame, collect_dataset_paths,
    row_normalize_edge_weight, normalize_edge_weight,
    build_random_walk_matrix, occupation_time_ppr,
    sparse_from_matrix, get_graph_embedding,get_embedding_with_specific_edge_removed,map_temperature_to_range
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GNN-CL model on full test set.")
    parser.add_argument("--nodes_dir", type=str, required=False,
        default="../../data/processed/highD/nodes/test",
        help="Directory of test node CSV files")
    parser.add_argument("--adj_dir", type=str, required=False,
        default="../../data/processed/highD/adj/test",
        help="Directory of test adjacency CSV files")
    parser.add_argument("--ckpt", type=str, required=False,
        default="../../results/gnncl_best.pth",
        help="Path to trained model checkpoint")
    parser.add_argument("--feature_columns", nargs="+",
        default=["x", "y", "vx", "vy", "sie", "heading_angle"])
    parser.add_argument("--threshold", type=float, default=0.0001, help="Importance delta threshold")
    parser.add_argument("--k", type=int, default=5, help="K for Precision@K and NDCG@K")
    parser.add_argument("--acc_ratio", type=float, default=0.95, help="Accumulated importance ratio for sparsity")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    return parser.parse_args()



"""
    Collect processed graph data objects for training/validation.

    Args:
        nodes_dir (str): Directory containing node CSVs.
        adj_dir (str): Directory containing adjacency CSVs.
        feature_columns (list): Feature column names to extract.
        adj_types (list): List of adjacency type suffixes.
        sample_num (int): If >0, randomly sample N graphs.
        seed (int): Random seed for sampling.

    Returns:
        list: [torch_geometric.data.Data,...] for each graph/frame.
    """
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

"""
    For a given target node, perform ablation analysis to get neighbor importance ranking,
    compare with spatial pseudo-distance topK, and compute sparsity.

    Args:
        data: graph Data object (PyG).
        fused_final, edge_index_fused: fused attention edge weights and structure.
        edge_index_diff, edge_weight_diff: diffusion (propagation) edge indices/weights.
        model_objs: Dict of model modules.
        target_idx: Ego node index.
        node_df: Original node dataframe (for positions, etc).
        threshold: Importance threshold.
        device: Device.
        k: TopK for metric evaluation.
        acc_ratio: Ratio for sparsity.

    Returns:
        (precision_k, ndcg_k, k_eval, sparsity)
"""
def neighbor_importance_analysis_joint_with_threshold(
    data, fused_final, edge_index_fused, edge_index_diff, edge_weight_diff,
    model_objs, target_idx, node_df, threshold=0.0001, device="cuda", k=5, acc_ratio=0.95
):
    """
    针对目标节点，重要性Δ值排序，与伪距离全场排序的TopK比较，同时统计稀疏性。
    """
    h_orig = get_embedding_with_specific_edge_removed(
        data.x, data.edge_index, data.edge_attr, data.complexity,
        fused_final, edge_index_fused, edge_index_diff, edge_weight_diff,
        model_objs['fused_encoder'], model_objs['diffuse_encoder'],
        target_idx, remove_fused=None, remove_diff=None, device=device
    )[target_idx].cpu()

    neighbors_fused = set([tgt.item() for src, tgt in zip(edge_index_fused[0], edge_index_fused[1]) if src.item() == target_idx])
    neighbors_diff = set([tgt.item() for src, tgt in zip(edge_index_diff[0], edge_index_diff[1]) if src.item() == target_idx])
    all_neighbors = sorted(neighbors_fused | neighbors_diff)
    results_dict = {}
    for neighbor in all_neighbors:
        remove_fused = (target_idx, neighbor) if neighbor in neighbors_fused else None
        remove_diff = (target_idx, neighbor) if neighbor in neighbors_diff else None
        h_new = get_embedding_with_specific_edge_removed(
            data.x, data.edge_index, data.edge_attr, data.complexity,
            fused_final, edge_index_fused, edge_index_diff, edge_weight_diff,
            model_objs['fused_encoder'], model_objs['diffuse_encoder'],
            target_idx, remove_fused=remove_fused, remove_diff=remove_diff, device=device
        )[target_idx].cpu()
        delta = torch.norm(h_new - h_orig).item()
        results_dict[neighbor] = delta

    full_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    model_sort = [n for n, s in full_results if s >= threshold]

    N = node_df.shape[0]
    delta_vals = np.array([s for n, s in full_results if s >= threshold])
    total = delta_vals.sum()
    if total > 0:
        delta_sorted = np.sort(delta_vals)[::-1]
        cumsum = np.cumsum(delta_sorted)
        num = np.searchsorted(cumsum, acc_ratio * total) + 1
        sparsity = num / (N - 1)
    else:
        sparsity = 0.

    xs = node_df['x'].values
    ys = node_df['y'].values
    vxs = node_df['vx'].values
    vys = node_df['vy'].values
    headings_deg = node_df['heading_angle'].values
    vs = np.sqrt(vxs**2 + vys**2)
    headings = np.deg2rad(headings_deg)
    x_ego, y_ego, v_ego, phi_ego = xs[target_idx], ys[target_idx], vs[target_idx], headings[target_idx]
    dx_global = xs - x_ego
    dy_global = ys - y_ego
    dx_rot = dx_global * np.cos(-phi_ego) - dy_global * np.sin(-phi_ego)
    dy_rot = dx_global * np.sin(-phi_ego) + dy_global * np.cos(-phi_ego)
    l, w, alpha = 4.5, 1.8, 0.056
    pseudo_distance = np.sqrt((dx_rot / (l * np.exp(alpha * v_ego)))**2 + (dy_rot / w)**2)
    pseudo_distance[target_idx] = np.inf
    pseudo_scores = {i: 1 / (pseudo_distance[i] + 1e-6) for i in range(len(xs)) if i != target_idx}
    gt_full_sort = [n for n, _ in sorted(pseudo_scores.items(), key=lambda x: x[1], reverse=True)]

    k_eval = min(k, N-1)
    if k_eval == 0 or len(model_sort) == 0:
        return 0, 0, 0, 0.

    gt_topk = gt_full_sort[:k_eval]
    model_topk = model_sort[:k_eval]

    # Precision@K
    prec_k = sum([1 for n in gt_topk if n in model_sort]) / k_eval

    # NDCG@K
    dcg = 0
    for i, node in enumerate(model_topk):
        if node in gt_topk:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(k_eval))
    ndcg_k = dcg / (idcg + 1e-8)

    return prec_k, ndcg_k, k_eval, sparsity
"""
    Process a single graph batch to compute final fused edge weights.

    Args:
        batch: Batch Data object.
        edge_fusion_layer: Edge fusion network.
        temp_net: Temperature regularization network.
        gate_net: Monotonic gating network.

    Returns:
        fused_final: Tensor of edge weights (after temp/gating).
        ...         (plus several intermediate results for debugging/analysis)
    """
def process_batch(batch, edge_fusion_layer, temp_net, gate_net):
    fused_edge_weight = edge_fusion_layer(batch.edge_attr)
    temp_raw = temp_net(batch.complexity).squeeze(-1)
    temperature = map_temperature_to_range(temp_raw)
    src = batch.edge_index[0]
    temp_per_edge = temperature[src]
    logits = fused_edge_weight / temp_per_edge

    alpha = softmax(logits, src)
    fused_weight_with_temp = alpha * fused_edge_weight
    gate = gate_net(batch.complexity)
    gate_per_edge = gate[src]
    fused_final = (1 - gate_per_edge) * fused_edge_weight + gate_per_edge * fused_weight_with_temp
    fused_final = row_normalize_edge_weight(batch.edge_index, fused_final, batch.x.size(0))
    return fused_final, fused_edge_weight, temperature, gate, src, alpha, fused_weight_with_temp

def process_all_graphs(data, model_objs, device="cuda"):
    # 确保输入全在 device
    data = data.to(device)
    # Step1. 融合图
    fused_final, _, _, _, _, _, _ = process_batch(
        data,
        model_objs['edge_fusion_layer'],
        model_objs['temp_net'],
        model_objs['gate_net']
    )
    edge_index_fused = data.edge_index.to(device)  # shape: [2, num_edges]
    # Step2. 扩散图
    N = data.x.shape[0]
    alpha = model_objs['jump_net'](data.complexity.to(device)).squeeze(-1)
    norm_edge_weight = normalize_edge_weight(data.edge_index.to(device), fused_final, N)
    P = build_random_walk_matrix(data.edge_index.to(device), norm_edge_weight, N)
    S = occupation_time_ppr(P, alpha)
    edge_index_diff, edge_weight_diff = sparse_from_matrix(S, threshold=0)
    # 全部保证是device上的tensor
    return fused_final, edge_index_fused, edge_index_diff, edge_weight_diff



"""
    Evaluate all nodes in one frame for attribution metrics.

    Args:
        node_csv_path: CSV file for node features.
        adj_csv_paths: List of CSVs for adjacency.
        feature_columns: List of feature columns.
        model_objs: dict of model modules.
        node_df: DataFrame of nodes.
        threshold: Importance threshold.
        device: device.
        k: TopK for metrics.
        acc_ratio: Sparsity metric ratio.

    Returns:
        (mean_precision, mean_ndcg, mean_sparsity, ...)
    """

def evaluate_frame_with_threshold(
    node_csv_path, adj_csv_paths, feature_columns, model_objs,
    node_df, threshold=0, device="cuda", k=5, acc_ratio=0.95
):
    data = process_one_frame(node_csv_path, adj_csv_paths, feature_columns)
    data = data.to(device)
    fused_final, edge_index_fused, edge_index_diff, edge_weight_diff = process_all_graphs(data, model_objs, device)
    num_nodes = data.x.shape[0]
    all_precision, all_ndcg, all_k, all_sparsity = [], [], [], []
    for ego_idx in tqdm(range(num_nodes)):
        prec_k, ndcg_k, k_eval, sparsity = neighbor_importance_analysis_joint_with_threshold(
            data, fused_final, edge_index_fused, edge_index_diff, edge_weight_diff,
            model_objs, target_idx=ego_idx, node_df=node_df, threshold=threshold, device=device, k=k, acc_ratio=acc_ratio
        )
        if k_eval > 0:
            all_precision.append(prec_k)
            all_ndcg.append(ndcg_k)
            all_k.append(k_eval)
            all_sparsity.append(sparsity)
    mean_precision = np.mean(all_precision) if all_precision else 0
    mean_ndcg = np.mean(all_ndcg) if all_ndcg else 0
    mean_sparsity = np.mean(all_sparsity) if all_sparsity else 0
    print("\n========== 评估结果 ==========")
    print(f"平均 Precision@K: {mean_precision:.4f}")
    print(f"平均 NDCG@K: {mean_ndcg:.4f}")
    print(f"平均 Top-K 大小: {np.mean(all_k):.2f}")
    print(f"平均 稀疏性(Sparsity): {mean_sparsity:.4f}")
    print("=============================")
    return mean_precision, mean_ndcg, mean_sparsity, all_precision, all_ndcg, all_sparsity
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feature_columns = args.feature_columns


    node_feat_dim = 6
    complexity_dim = 5
    gnn_dim, out_dim = 64, 32  # 建议与训练保持一致

    shared_mlp = SharedMLP(gnn_dim, out_dim).to(device)
    layer_encoders = torch.nn.ModuleList([NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device) for _ in range(5)])
    fused_encoder = NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device)
    diffuse_encoder = NodeEncoder(node_feat_dim, gnn_dim, out_dim, shared_mlp).to(device)
    edge_fusion_layer = EdgeFusionLayer(num_layers=5).to(device)
    temp_net = build_temp_net(complexity_dim).to(device)
    gate_net = MonotonicGateNet(complexity_dim, hidden_dim=8).to(device)
    jump_net = MonotonicJumpNet(complexity_dim, hidden_dim=8).to(device)
    graph_mlp = GraphMLP(out_dim, out_dim, hidden_dim=64).to(device)

    # === 加载模型权重 ===
    ckpt = torch.load(args.ckpt, map_location=device)
    layer_encoders.load_state_dict(ckpt['layer_encoders'])
    fused_encoder.load_state_dict(ckpt['fused_encoder'])
    diffuse_encoder.load_state_dict(ckpt['diffuse_encoder'])
    edge_fusion_layer.load_state_dict(ckpt['edge_fusion_layer'])
    temp_net.load_state_dict(ckpt['temp_net'])
    gate_net.load_state_dict(ckpt['gate_net'])
    jump_net.load_state_dict(ckpt['jump_net'])
    model_objs = dict(
        layer_encoders=layer_encoders,
        fused_encoder=fused_encoder,
        diffuse_encoder=diffuse_encoder,
        edge_fusion_layer=edge_fusion_layer,
        temp_net=temp_net,
        gate_net=gate_net,
        jump_net=jump_net,
        graph_mlp=graph_mlp
    )
    for m in model_objs.values(): m.eval()


    dataset_pairs = collect_dataset_paths(args.nodes_dir, args.adj_dir)
    print(f"实际评估 {len(dataset_pairs)} 组样本")

    all_frame_precision, all_frame_ndcg, all_frame_sparsity = [], [], []
    for node_csv_path, adj_csv_paths in tqdm(dataset_pairs, desc="Evaluating"):
        node_df = pd.read_csv(node_csv_path)
        mean_precision, mean_ndcg, mean_sparsity, _, _, _ = evaluate_frame_with_threshold(
            node_csv_path, adj_csv_paths, feature_columns, model_objs,
            node_df=node_df, threshold=args.threshold, device=args.device, k=args.k, acc_ratio=args.acc_ratio
        )
        all_frame_precision.append(mean_precision)
        all_frame_ndcg.append(mean_ndcg)
        all_frame_sparsity.append(mean_sparsity)

    dataset_mean_precision = float(torch.tensor(all_frame_precision).mean()) if all_frame_precision else 0
    dataset_mean_ndcg = float(torch.tensor(all_frame_ndcg).mean()) if all_frame_ndcg else 0
    dataset_mean_sparsity = float(torch.tensor(all_frame_sparsity).mean()) if all_frame_sparsity else 0

    print("\n========== Dataset Evaluation Results ==========")
    print(f"Overall mean Precision@K: {dataset_mean_precision:.4f}")
    print(f"Overall mean NDCG@K: {dataset_mean_ndcg:.4f}")
    print(f"Overall mean Sparsity: {dataset_mean_sparsity:.4f}")
    print(f"Number of valid frames: {len(all_frame_precision)}")
    print("===============================================")

    # ==== Write to results directory ====
    results_dir = "../../results"
    os.makedirs(results_dir, exist_ok=True)
    eval_summary_path = os.path.join(results_dir, "eval_summary.txt")
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        f.write("========== Dataset Evaluation Results ==========\n")
        f.write(f"Overall mean Precision@K: {dataset_mean_precision:.4f}\n")
        f.write(f"Overall mean NDCG@K: {dataset_mean_ndcg:.4f}\n")
        f.write(f"Overall mean Sparsity: {dataset_mean_sparsity:.4f}\n")
        f.write(f"Number of valid frames: {len(all_frame_precision)}\n")
        f.write("===============================================\n")
    print(f"\nEvaluation results saved to: {eval_summary_path}")


if __name__ == "__main__":
    main()
