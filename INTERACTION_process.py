import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import argparse
import random


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
print("Switched Working Directory:", os.getcwd())



"""
Determine whether the target vehicle is within the perception range of the ego vehicle.
The range considers both longitudinal and lateral fields based on motion and geometry.
"""
def is_in_perception_range(ego, target, lane_width=3.75):
    vx = ego['vx']
    vy = ego['vy']
    speed_mps = np.sqrt(vx ** 2 + vy ** 2)
    V = speed_mps * 3.6
    tP_L3, Ad, tT, Adp = 4.3, 3.4, 3.8, 2.5
    term1 = 0.278 * V * tP_L3
    term2 = 0.039 * (V - (Adp / 9.81)) ** 2 / Ad
    term3 = (Adp * tT ** 2) / 19.62
    FR_L3 = term1 + term2 - term3
    rear_L = speed_mps * 2
    dx = target['x'] - ego['x']
    dy = target['y'] - ego['y']
    theta = np.radians(ego['heading_angle'])
    dx_rot = dx * np.cos(theta) + dy * np.sin(theta)
    dy_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    lateral_half_width = lane_width + 0.5 * ego.get('width', 2.0)
    return (abs(dy_rot) <= lateral_half_width) and (-rear_L <= dx_rot <= FR_L3)


"""
    Scan through frames and extract valid, non-overlapping time windows
    where all vehicles are visible and sufficient activity is present.
    Returns a list of (start_frame, end_frame) tuples.
"""

def find_valid_windows(track_df, window_size=50):
    vehicle_frames = track_df.groupby('track_id')['frame_id'].agg(['min', 'max']).rename(
        columns={'min': 'start', 'max': 'end'})
    frame_to_vehicles = track_df.groupby('frame_id')['track_id'].apply(set).to_dict()
    valid_windows = []
    selected_vehicles = set()
    frame_ids = sorted(track_df['frame_id'].unique())
    min_frame, max_frame = min(frame_ids), max(frame_ids)
    current_frame = min_frame
    while current_frame <= max_frame - window_size + 1:
        end_frame = current_frame + window_size - 1
        window_vehicles = set()
        for f in range(current_frame, end_frame + 1):
            window_vehicles.update(frame_to_vehicles.get(f, set()))
        if not window_vehicles or (window_vehicles & selected_vehicles):
            current_frame += 1
            continue
        valid = all(vehicle_frames.loc[vid, 'start'] <= current_frame for vid in window_vehicles)
        if not valid:
            current_frame += 1
            continue
        if len(frame_to_vehicles.get(end_frame, set())) < 3:
            current_frame += 1
            continue
        valid_windows.append((current_frame, end_frame))
        selected_vehicles.update(window_vehicles)
        current_frame = end_frame + 1
    print(f"Found {len(valid_windows)} valid windows")
    return valid_windows


"""
Calculate the total number of unique neighbors observed by a vehicle
across the historical window for SIE computation.
"""
def compute_accumulated_degree(vehicle_id, history):
    neighbors = set()
    for frame_neighbors in history[vehicle_id]['neighbors_window']:
        neighbors.update(frame_neighbors)
    return len(neighbors)

def compute_sie(zeta_window):
    if len(zeta_window) < 3:
        return 0.0
    diffs = [abs(zeta_window[i + 1] - 2 * zeta_window[i] + zeta_window[i - 1]) for i in range(1, len(zeta_window) - 1)]
    return np.mean(diffs) if diffs else 0.0

"""
Compute five pairwise risk features between ego and target vehicles
for each edge in the multi-view risk graph.
"""


def calculate_all_components(ego, target):
    lane_width = 3.75
    def get_heading_angle(vx, vy):
        angle_rad = np.arctan2(vy, vx)
        return np.degrees(angle_rad) % 360
    ego_heading = get_heading_angle(ego['vx'], ego['vy'])
    target_heading = get_heading_angle(target['vx'], target['vy'])
    delta_theta = abs(ego_heading - target_heading) % 360
    delta_theta = min(delta_theta, 360 - delta_theta)
    theta = min(delta_theta, 180 - delta_theta)
    f1 = 0.5 * (1 - np.cos(theta * np.pi / (101.25) + np.pi / 10))
    v_lateral_max, v_lateral_min = 2, -2
    dy = target['y'] - ego['y']
    v_lat = target['vy'] - ego['vy']
    v_prime = (v_lat - v_lateral_min) / (v_lateral_max - v_lateral_min)
    v_prime = np.clip(v_prime, 1e-5, 1 - 1e-5)
    f2 = (1 - v_prime) * np.log(1 / v_prime) if 0 < v_prime < 1 else 0
    dx = target['x'] - ego['x']
    theta_rad = np.radians(ego['heading_angle'])
    dx_rot = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
    d_lon = abs(dx_rot)
    d_longitudinal_max = 0
    d_longitudinal_min = ego['FR_L3']
    d_prime = (d_longitudinal_max - d_lon) / (d_longitudinal_max - d_longitudinal_min)
    d_prime = np.clip(d_prime, 1e-5, 1 - 1e-5)
    f3 = (1 - d_prime) * np.log(1 / d_prime) if 0 < d_prime < 1 else 0
    d_lat = abs(dy)
    d_lateral_max = 0
    d_lateral_min = lane_width * 4
    d_lateral_prime = (d_lateral_max - d_lat) / (d_lateral_max - d_lateral_min)
    d_lateral_prime = np.clip(d_lateral_prime, 1e-5, 1 - 1e-5)
    f4 = (1 - d_lateral_prime) * np.log(1 / d_lateral_prime)
    v_lon_max, v_lon_min = 30, -30
    effective_dv = abs(ego['vx']) - abs(target['vx'])
    v_lon_prime = (effective_dv - v_lon_min) / (v_lon_max - v_lon_min)
    v_lon_prime = np.clip(v_lon_prime, 1e-5, 1 - 1e-5)
    f5 = (1 - v_lon_prime) * np.log(1 / v_lon_prime) if 0 < v_lon_prime < 1 else 0
    return f1, f2, f3, f4, f5

"""
For each frame, build a neighbor graph for all vehicles and update
neighbor history, accumulated degree, and closeness for SIE computation.
"""
def update_history(frame_data, history):
    vehicles = []
    for _, row in frame_data.iterrows():
        vx, vy = row['vx'], row['vy']
        heading_angle = np.degrees(np.arctan2(vy, vx)) % 360
        width = row['width'] if 'width' in row else 2.0
        speed_mps = np.sqrt(vx ** 2 + vy ** 2)
        V = speed_mps * 3.6
        tP_L3, Ad, tT, Adp = 4.3, 3.4, 3.8, 2.5
        FR_L3 = 0.278 * V * tP_L3 + 0.039 * (V - Adp / 9.81) ** 2 / Ad - (Adp * tT ** 2) / 19.62
        rear_L = speed_mps * 2
        vehicle_dict = {
            'id': row['track_id'], 'x': row['x'], 'y': row['y'],
            'vx': vx, 'vy': vy, 'heading_angle': heading_angle, 'width': width,
            'FR_L3': FR_L3, 'rear_L': rear_L
        }
        vehicles.append(vehicle_dict)
    N = len(vehicles)
    G_temp = nx.DiGraph()
    for v in vehicles:
        G_temp.add_node(v['id'])
    for i in range(N):
        ego = vehicles[i]
        for j in range(N):
            if i == j: continue
            target = vehicles[j]
            if is_in_perception_range(ego, target):
                G_temp.add_edge(ego['id'], target['id'])
    closeness = nx.closeness_centrality(G_temp) if G_temp.number_of_edges() > 0 else {v['id']: 0. for v in vehicles}
    for node_id in G_temp.nodes:
        current_neighbors = list(G_temp.successors(node_id))
        history[node_id]['neighbors_window'].append(current_neighbors)
        zeta_d = compute_accumulated_degree(node_id, history)
        history[node_id]['zeta_d_window'].append(zeta_d)
        zeta_c = closeness.get(node_id, 0.0)
        history[node_id]['zeta_c_window'].append(zeta_c)

def build_multi_layer_network(frame_data, history):
    vehicles = []
    for _, row in frame_data.iterrows():
        vx, vy = row['vx'], row['vy']
        heading_angle = np.degrees(np.arctan2(vy, vx)) % 360
        width = row['width'] if 'width' in row else 2.0
        speed_mps = np.sqrt(vx ** 2 + vy ** 2)
        V = speed_mps * 3.6
        tP_L3, Ad, tT, Adp = 4.3, 3.4, 3.8, 2.5
        FR_L3 = 0.278 * V * tP_L3 + 0.039 * (V - Adp / 9.81) ** 2 / Ad - (Adp * tT ** 2) / 19.62
        rear_L = speed_mps * 2
        vehicle_dict = {
            'id': row['track_id'], 'x': row['x'], 'y': row['y'],
            'vx': vx, 'vy': vy, 'heading_angle': heading_angle, 'width': width,
            'FR_L3': FR_L3, 'rear_L': rear_L
        }
        vehicles.append(vehicle_dict)
    N = len(vehicles)
    nodes = [v['id'] for v in vehicles]
    node2idx = {vid: idx for idx, vid in enumerate(nodes)}
    sie_list = []
    for v in vehicles:
        sie_A = max(compute_sie(history[v['id']]['zeta_c_window']), compute_sie(history[v['id']]['zeta_d_window']))
        sie_list.append(sie_A)
    feature_names = ['x', 'y', 'vx', 'vy', 'heading_angle', 'width', 'sie']
    X = np.zeros((N, len(feature_names)))
    for i, v in enumerate(vehicles):
        vals = [v['x'], v['y'], v['vx'], v['vy'], v['heading_angle'], v['width'], sie_list[i]]
        X[i] = vals
    A_angle = np.zeros((N, N))
    A_lat_v = np.zeros((N, N))
    A_lon_d = np.zeros((N, N))
    A_lat_d = np.zeros((N, N))
    A_lon_v = np.zeros((N, N))
    for i in range(N):
        ego = vehicles[i]
        for j in range(N):
            if i == j: continue
            target = vehicles[j]
            if is_in_perception_range(ego, target):
                f1, f2, f3, f4, f5 = calculate_all_components(ego, target)
                sie_A = max(compute_sie(history[ego['id']]['zeta_c_window']),
                            compute_sie(history[ego['id']]['zeta_d_window']))
                sie_B = max(compute_sie(history[target['id']]['zeta_c_window']),
                            compute_sie(history[target['id']]['zeta_d_window']))
                complexity = 1 + sie_A + sie_B
                idx_ego = node2idx[ego['id']]
                idx_target = node2idx[target['id']]
                A_angle[idx_ego, idx_target] = complexity * f1
                A_lat_v[idx_ego, idx_target] = complexity * f2
                A_lon_d[idx_ego, idx_target] = complexity * f3
                A_lat_d[idx_ego, idx_target] = complexity * f4
                A_lon_v[idx_ego, idx_target] = complexity * f5

    def row_normalize(A):
        A = A.copy()
        s = A.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return A / s

    A_angle = row_normalize(A_angle)
    A_lat_v = row_normalize(A_lat_v)
    A_lon_d = row_normalize(A_lon_d)
    A_lat_d = row_normalize(A_lat_d)
    A_lon_v = row_normalize(A_lon_v)
    return X, nodes, A_angle, A_lat_v, A_lon_d, A_lat_d, A_lon_v, feature_names

def main(args):
    window_size = args.window_size
    input_path = args.input
    nodes_dir = args.nodes_dir
    adj_dir = args.adj_dir

    # 创建 nodes/adj 的 train/val/test 子文件夹（确保全部存在）
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(nodes_dir, split), exist_ok=True)
        os.makedirs(os.path.join(adj_dir, split), exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.rename(columns={
        'case_id': 'case_id',
        'track_id': 'track_id',
        'frame_id': 'frame_id',
        'x': 'x',
        'y': 'y',
        'vx': 'vx',
        'vy': 'vy',
        'width': 'width'
    })

    for case_id, case_df in df.groupby('case_id'):
        print(f"\n==== Processing scenario case_id={case_id} ====")
        valid_windows = find_valid_windows(case_df, window_size)

        # Split windows into train/val/test (ratio 6:2:2)
        random.seed(42)
        idx = list(range(len(valid_windows)))
        random.shuffle(idx)
        n = len(valid_windows)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        split_map = {}
        for i in train_idx: split_map[i] = 'train'
        for i in val_idx:   split_map[i] = 'val'
        for i in test_idx:  split_map[i] = 'test'
        print(f"Window split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        for window_idx, (start_frame, end_frame) in enumerate(valid_windows):
            split = split_map[window_idx]
            print(f"== {split.upper()} Window {window_idx} [{start_frame}-{end_frame}] ==")
            history = defaultdict(lambda: {
                'neighbors_window': deque(maxlen=window_size),
                'zeta_c_window': deque(maxlen=window_size),
                'zeta_d_window': deque(maxlen=window_size)
            })
            for frame_num in range(start_frame, end_frame + 1):
                frame_data = case_df[case_df['frame_id'] == frame_num]
                if frame_data.empty:
                    continue
                update_history(frame_data, history)
                if frame_num == end_frame:
                    X, nodes, A_angle, A_lat_v, A_lon_d, A_lat_d, A_lon_v, feature_names = build_multi_layer_network(
                        frame_data, history)
                    node_filename = os.path.join(nodes_dir, split,
                                                 f"nodes_case{case_id}_win{window_idx}_frame{frame_num}.csv")
                    pd.DataFrame(X, index=nodes, columns=feature_names).to_csv(node_filename, index_label='id')
                    risk_names = ['angle', 'latv', 'lond', 'latd', 'lonv']
                    adj_mats = [A_angle, A_lat_v, A_lon_d, A_lat_d, A_lon_v]
                    for risk, A in zip(risk_names, adj_mats):
                        adj_filename = os.path.join(adj_dir, split,
                                                    f"adj_{risk}_case{case_id}_win{window_idx}_frame{frame_num}.csv")
                        pd.DataFrame(A, index=nodes, columns=nodes).to_csv(adj_filename, index=True, header=True)
    print("All export done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess INTERACTION dataset with multi-view risk graphs and split (6:2:2).")
    parser.add_argument('--input', type=str, required=False,
        default='data/raw/INTERACTION/DR_CHN_Merging_ZS.csv',
        help='Path to INTERACTION csv file (with all cases)')
    parser.add_argument('--window_size', type=int, default=10, help='Sliding window size')
    parser.add_argument('--nodes_dir', type=str, required=False,
        default='data/processed/INTERACTION/nodes',
        help='Output directory for node feature csv files')
    parser.add_argument('--adj_dir', type=str, required=False,
        default='data/processed/INTERACTION/adj',
        help='Output directory for adjacency matrix csv files')
    args = parser.parse_args()
    main(args)
