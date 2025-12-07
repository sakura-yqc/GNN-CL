import argparse
import train
import eval

def parse_args():
    parser = argparse.ArgumentParser(description="Unified GNN-CL training/evaluation entry point.")
    # === Main mode selection ===
    parser.add_argument("mode", choices=["train", "eval"], help="Choose running mode: train or eval.")

    # === Shared/general parameters ===
    parser.add_argument("--feature_columns", nargs="+", default=["x", "y", "vx", "vy", "sie", "heading_angle"],
                        help="List of node feature column names.")

    # === Training-specific parameters ===
    parser.add_argument("--train_nodes_dir", type=str, help="Directory of training node CSV files.")
    parser.add_argument("--train_adj_dir", type=str, help="Directory of training adjacency CSV files.")
    parser.add_argument("--val_nodes_dir", type=str, help="Directory of validation node CSV files.")
    parser.add_argument("--val_adj_dir", type=str, help="Directory of validation adjacency CSV files.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per setting.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--gnn_dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--out_dim_list", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="List of embedding dimensions to search.")
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.01, 0.001, 0.0001],
                        help="List of learning rates to search.")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs for contrastive learning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_ckpt", type=str, default="gnncl_best.pth", help="Path to save the best model checkpoint.")
    parser.add_argument("--sample_num", type=int, default=0, help="If >0, randomly sample N graphs for training.")

    # === Evaluation-specific parameters ===
    parser.add_argument("--nodes_dir", type=str, help="Directory of test node CSV files.")
    parser.add_argument("--adj_dir", type=str, help="Directory of test adjacency CSV files.")
    parser.add_argument("--ckpt", type=str, help="Path to trained model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.0001, help="Importance delta threshold for evaluation.")
    parser.add_argument("--k", type=int, default=5, help="K value for Precision@K and NDCG@K metrics.")
    parser.add_argument("--acc_ratio", type=float, default=0.95, help="Accumulated importance ratio for sparsity metric.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation: cuda or cpu.")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        train.main(args)
    elif args.mode == "eval":
        eval.main(args)

if __name__ == "__main__":
    main()
