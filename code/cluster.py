import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
import os
import pandas as pd
import math
import json
import random
import textwrap

# ==========================================
# 1. model import
# ==========================================
try:
    from GRU_model import GRUAutoencoder
    from LSTM_model import LSTMAutoencoder
    from ResNet_model import ResNetAutoencoder

    from ODE_model_1 import Autoencoder as NodeGruAutoencoder
except ImportError as e:
    print(f"Error importing models: {e}")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


CURRENT_USER_NAME = 'yaozhiran'
# chenzhilin,huxinyu,huxinyu2,liucunyang,yaozhiran,zhoujun


BASE_MODEL_DIR = 

CONFIG = {

    'MODEL_PATHS': {
        'ODE': f"C:/Users/MINT/Desktop/20241111华为wifi切换/data0113/template/{CURRENT_USER_NAME}/node_gru_autoencoder.pth",
        'ResNet': f"{BASE_MODEL_DIR}/{CURRENT_USER_NAME}/resnet_autoencoder.pth",
        'LSTM': f"{BASE_MODEL_DIR}/{CURRENT_USER_NAME}/lstm_autoencoder.pth",
        'GRU': f"{BASE_MODEL_DIR}/{CURRENT_USER_NAME}/gru_autoencoder.pth"
    },


    'DATA_ROOT_DIR': ,


    'TARGET_FOLDER_NAME': CURRENT_USER_NAME,


    'ALL_AVAILABLE_FOLDERS': None,


    'SEQ_LENGTH': 32,
    'AFTER_DOOR_POINTS': 1,
    'HIDDEN_DIM': 32,
    'INPUT_DIM': 1,


    'INITIAL_THRESHOLD_PERCENTAGE': 0.22,

    'ODE_RECOGNITION_TOLERANCE_SCALE': 1.1,
    'DEFAULT_RECOGNITION_TOLERANCE_SCALE': 1.1,


    'CLUSTER_COUNT_MIN': 3,
    'CLUSTER_COUNT_MAX': 5,
    'SEARCH_START_PERCENTAGE': 0.10,
    'SEARCH_STEP': 0.005,


    'RECALL_SAMPLE_RATIO': 0.3,
    'ACCURACY_SAMPLE_COUNT': 25,


    'MAX_RETRIES': 10
}


# ==========================================

# ==========================================
def read_trash_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            row = line.strip().split(',')
            if len(row) >= 7:
                selected_row = row[:4] + row[-3:]
                data.append(selected_row)
    if not data or len(data) < 2: return None
    df = pd.DataFrame(data[1:], columns=data[0])
    if 'rssi' not in df.columns: return None
    try:
        df['rssi'] = pd.to_numeric(df['rssi'])
    except:
        return None
    if 'manual_flag' in df.columns:
        df['manual_flag'] = df['manual_flag'].apply(lambda x: 'Door' if x != 'default' else x)
    return df


def process_single_file(file_path, seq_length, after_door1):
    df = read_trash_data(file_path)
    if df is None or 'manual_flag' not in df.columns: return None
    door1_line = df[df['manual_flag'] == 'Door'].index
    if len(door1_line) == 0: return None
    start_index = door1_line[0] - seq_length + after_door1
    if start_index < 0: return None
    try:
        seq_data = df['rssi'][start_index: start_index + seq_length].values
        if len(seq_data) == seq_length:
            seq_tensor = torch.from_numpy(seq_data).float()
            time_points = torch.arange(seq_length).float()
            return (seq_tensor, time_points)
    except:
        return None
    return None


def load_real_data(folder_path, seq_length=32, after_door1=1):
    original_sequences = []
    if not os.path.exists(folder_path):
        print(f"Warning: Path not found: {folder_path}")
        return []
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.csv') or f.endswith('.txt'):
                all_files.append(os.path.join(root, f))
    for file_path in all_files:
        seq = process_single_file(file_path, seq_length, after_door1)
        if seq is not None:
            original_sequences.append(seq)
    return original_sequences


def load_accuracy_data_dynamic(root_dir, target_folder, all_folders_list, sample_count, seq_length, after_door1):
    print(f"\nPreparing Accuracy Dataset (Excluding '{target_folder}')...")
    if all_folders_list is None:
        all_folders_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    candidate_files = []
    for folder_name in all_folders_list:
        if folder_name == target_folder: continue
        folder_path = os.path.join(root_dir, folder_name)
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.csv') or f.endswith('.txt'):
                    candidate_files.append(os.path.join(root, f))
    if len(candidate_files) == 0: return []
    if len(candidate_files) > sample_count:
        selected_files = random.sample(candidate_files, sample_count)
    else:
        selected_files = candidate_files
        print(f"  -> Using all {len(selected_files)} available files.")
    accuracy_sequences = []
    for file_path in selected_files:
        seq = process_single_file(file_path, seq_length, after_door1)
        if seq is not None:
            accuracy_sequences.append(seq)


    print(f"  -> Total valid accuracy sequences loaded: {len(accuracy_sequences)}")

    return accuracy_sequences


# ==========================================

# ==========================================
def extract_embeddings(model_name, model, sequences):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq, times in sequences:
            if model_name == 'ResNet':
                input_tensor = seq.unsqueeze(0).unsqueeze(0).to(device)
                times_tensor = times.to(device)
                emb = model.encoder(input_tensor, times_tensor)
            elif model_name == 'ODE':


                input_tensor = seq.unsqueeze(1).unsqueeze(1).to(device)
                times_tensor = times.to(device)


                emb = model.encoder(input_tensor, times_tensor)
            else:
                input_tensor = seq.unsqueeze(1).unsqueeze(1).to(device)
                times_tensor = times.to(device)
                emb = model.encoder(input_tensor, times_tensor)
            embeddings.append(emb.cpu().numpy().squeeze())
    return np.array(embeddings)


# ==========================================

# ==========================================
def optimize_threshold_for_ode(embeddings, linkage_matrix, max_dist):
    print("\n[ODE Optimization] Searching for optimal threshold percentage...")
    min_clusters = CONFIG['CLUSTER_COUNT_MIN']
    max_clusters = CONFIG['CLUSTER_COUNT_MAX']
    current_p = CONFIG['SEARCH_START_PERCENTAGE']
    step = CONFIG['SEARCH_STEP']
    best_p = CONFIG['INITIAL_THRESHOLD_PERCENTAGE']
    found = False

    while current_p < 0.60:
        dist_thresh = max_dist * current_p
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, linkage='ward')
        try:
            cluster_preds = clustering.fit_predict(embeddings)
            n_clusters = len(set(cluster_preds))
        except Exception:
            n_clusters = len(embeddings)

        if min_clusters <= n_clusters <= max_clusters:
            print(f"  >>> FOUND: P={current_p:.3f} yields {n_clusters} clusters")
            best_p = current_p
            found = True
            break

        if n_clusters < min_clusters:
            break

        current_p += step

    if not found:
        print(f"  >>> Using default/fallback P={best_p}")

    return best_p


def plot_combined_dendrograms(dendrogram_data_list):
    print("\n>>> Generating Combined Dendrograms (2x2)...")

    icml_style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,


        "axes.labelsize": 10,

        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    }

    if not dendrogram_data_list: return

    with plt.rc_context(icml_style):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=350)
        axes_flat = axes.flatten()

        set_link_color_palette(['#C0392B', '#2980B9', '#F39C12', '#27AE60'])

        for idx, data in enumerate(dendrogram_data_list):
            if idx >= 4: break

            ax = axes_flat[idx]
            model_name = data['model_name']
            linkage_matrix = data['linkage']
            dist_thresh = data['threshold']

            raw_max_dist = linkage_matrix[:, 2].max()
            linkage_norm = linkage_matrix.copy()
            linkage_norm[:, 2] = linkage_norm[:, 2] / raw_max_dist
            thresh_norm = dist_thresh / raw_max_dist

            dendrogram(
                linkage_norm,
                ax=ax,
                color_threshold=thresh_norm,
                above_threshold_color='#666666',
                truncate_mode='lastp',
                p=50,
                show_leaf_counts=True,
                leaf_rotation=0.,
                leaf_font_size=6.,
            )

            ax.axhline(y=thresh_norm, c='#C0392B', linestyle=(0, (5, 5)), lw=1.0, alpha=0.9, label='Threshold')


            ax.set_title(f"{model_name}", pad=6, fontsize=10, fontweight='bold')


            ax.set_ylabel("Normalized Distance" if idx % 2 == 0 else "")


            if idx >= 2:

                ax.set_xlabel("Sample Index")
            else:
                ax.set_xlabel("")

            ax.set_xticks([])

            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', linestyle=':', alpha=0.6, linewidth=1, zorder=0)

        plt.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.1)



        plt.show()
        set_link_color_palette(None)


# ==========================================

# ==========================================
def perform_clustering_and_visualization(model_name, sequences, embeddings, threshold_percentage, tolerance_scale):

    linkage_matrix = linkage(embeddings, method='ward')
    max_dist = linkage_matrix[-1, 2]
    dist_thresh = max_dist * threshold_percentage

    print(f"[{model_name}] Threshold %: {threshold_percentage:.3f}")

    dendrogram_data = {
        'model_name': model_name,
        'linkage': linkage_matrix,
        'threshold': dist_thresh
    }

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, linkage='ward')
    cluster_preds = clustering.fit_predict(embeddings)

    unique_labels = sorted(list(set(cluster_preds)))


    if len(unique_labels) > 9:
        print(f"[{model_name}] Too many clusters ({len(unique_labels)}). Keeping top 9 by size.")
        unique_labels.sort(key=lambda x: -np.sum(cluster_preds == x))
        clusters_to_show = unique_labels[:9]
    else:
        clusters_to_show = unique_labels

    n_clusters_show = len(clusters_to_show)
    print(f"[{model_name}] Visualizing {n_clusters_show} clusters.")

    all_templates = []

    icml_style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    }

    if n_clusters_show > 0:
        with plt.rc_context(icml_style):
            n_cols = 3
            n_rows = math.ceil(n_clusters_show / n_cols)


            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 13), dpi=300, squeeze=False)
            axes_flat = axes.flatten()

            for idx, cluster_id in enumerate(clusters_to_show):
                ax = axes_flat[idx]
                member_indices = np.where(cluster_preds == cluster_id)[0]
                n_members = len(member_indices)


                for i in member_indices:
                    seq, times = sequences[i]
                    ax.plot(times.numpy(), seq.numpy(),
                            color='#7F7F7F', alpha=0.35, linewidth=0.6, linestyle='-')


                if n_members < 2:
                    k = 1
                    sub_centers = [np.mean(embeddings[member_indices], axis=0)]
                    sub_labels = [0] * n_members
                else:
                    k_target = max(2, int(math.sqrt(n_members)))
                    k = min(k_target, n_members)
                    member_embs = embeddings[member_indices]
                    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                    sub_labels = kmeans.fit_predict(member_embs)
                    sub_centers = kmeans.cluster_centers_

                colors = plt.cm.jet(np.linspace(0, 1, k))

                for sub_idx in range(k):
                    if k == 1:
                        sub_indices_local = np.arange(n_members)
                    else:
                        sub_indices_local = np.where(sub_labels == sub_idx)[0]
                    if len(sub_indices_local) == 0: continue

                    sub_emb = embeddings[member_indices[sub_indices_local]]
                    center_vec = sub_centers[sub_idx]
                    dists = np.linalg.norm(sub_emb - center_vec, axis=1)
                    final_radius = np.max(dists) * tolerance_scale

                    nearest_local = np.argmin(dists)
                    nearest_global = member_indices[sub_indices_local[nearest_local]]
                    rep_signal = sequences[nearest_global][0].numpy()

                    all_templates.append({
                        'embedding': center_vec,
                        'threshold': final_radius,
                        'cluster_id': cluster_id,
                        'sub_id': sub_idx
                    })


                    ax.plot(range(len(rep_signal)), rep_signal,
                            color=colors[sub_idx], linewidth=1.2, alpha=0.8, linestyle='-',
                            label=f'P{sub_idx}')

                indices_str = str(member_indices.tolist())
                if len(indices_str) > 25:
                    title_idx = indices_str[:22] + "..."
                else:
                    title_idx = indices_str

                ax.set_title(f"{model_name} Cluster {cluster_id} (N={n_members})", fontsize=9)


                if idx % n_cols == 0:
                    ax.set_ylabel("RSSI (dBm)", fontsize=9)


                current_row = idx // n_cols

                if current_row == n_rows - 1:
                    ax.set_xlabel("time (s)", fontsize=9)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_visible(True)
                ax.tick_params(top=False, right=False, direction='in', width=0.6)
                ax.grid(axis='both', linestyle=':', alpha=0.6, linewidth=1)  

            for i in range(n_clusters_show, len(axes_flat)):
                axes_flat[i].set_visible(False)


            plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.95, hspace=0.8, wspace=0.3)
            # plt.show()

    return dendrogram_data, all_templates


# ==========================================

# ==========================================
def test_metrics(model_name, model, test_data, templates):
    if not templates or not test_data: return 0.0
    embeddings = extract_embeddings(model_name, model, test_data)

    proto_tensor = torch.tensor([t['embedding'] for t in templates]).float().to(device)
    thresh_tensor = torch.tensor([t['threshold'] for t in templates]).float().to(device)
    emb_tensor = torch.from_numpy(embeddings).float().to(device)

    dists = torch.norm(emb_tensor.unsqueeze(1) - proto_tensor.unsqueeze(0), dim=2)
    matches = dists < thresh_tensor.unsqueeze(0)
    any_match = torch.any(matches, dim=1)

    match_count = torch.sum(any_match).item()
    return (match_count / len(test_data)) * 100


# ==========================================

# ==========================================
def run_evaluation():
    target_folder_name = CONFIG['TARGET_FOLDER_NAME']
    root_dir = CONFIG['DATA_ROOT_DIR']
    target_folder_path = os.path.join(root_dir, target_folder_name)
    print(f"Target User: {target_folder_name}")

    full_train_data = load_real_data(target_folder_path, CONFIG['SEQ_LENGTH'], CONFIG['AFTER_DOOR_POINTS'])
    print(f"Loaded {len(full_train_data)} sequences.")
    if len(full_train_data) == 0: return

    acc_data = load_accuracy_data_dynamic(
        root_dir=root_dir,
        target_folder=target_folder_name,
        all_folders_list=CONFIG['ALL_AVAILABLE_FOLDERS'],
        sample_count=CONFIG['ACCURACY_SAMPLE_COUNT'],
        seq_length=CONFIG['SEQ_LENGTH'],
        after_door1=CONFIG['AFTER_DOOR_POINTS']
    )

    models_list = ['ODE', 'ResNet', 'LSTM', 'GRU']
    global_optimized_percentage = CONFIG['INITIAL_THRESHOLD_PERCENTAGE']

    all_dendrograms = []


    final_metrics = []

    for i, model_name in enumerate(models_list):
        print("\n" + "=" * 60)
        print(f"STARTING EVALUATION FOR: {model_name}")
        print("=" * 60)

        model_path = CONFIG['MODEL_PATHS'][model_name]
        try:
            if model_name == 'ODE':
            
                model = NodeGruAutoencoder(CONFIG['INPUT_DIM'], CONFIG['HIDDEN_DIM']).to(device)
            elif model_name == 'ResNet':
                model = ResNetAutoencoder(CONFIG['INPUT_DIM'], CONFIG['HIDDEN_DIM']).to(device)
            elif model_name == 'LSTM':
                model = LSTMAutoencoder(CONFIG['INPUT_DIM'], CONFIG['HIDDEN_DIM'], num_layers=2).to(device)
            elif model_name == 'GRU':
                model = GRUAutoencoder(CONFIG['INPUT_DIM'], CONFIG['HIDDEN_DIM'], num_layers=2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading {model_name}: {e}. Skipping.")
            continue

        print(f"[{model_name}] Extracting features...")
        embeddings = extract_embeddings(model_name, model, full_train_data)

        if model_name == 'ODE':
            linkage_matrix = linkage(embeddings, method='ward')
            max_dist = linkage_matrix[-1, 2]
            best_p = optimize_threshold_for_ode(embeddings, linkage_matrix, max_dist)
            global_optimized_percentage = best_p
            current_threshold_p = best_p
            current_tolerance = CONFIG['ODE_RECOGNITION_TOLERANCE_SCALE']
            print(f">>> ODE Optimization Complete. P={global_optimized_percentage:.3f}")
        else:
            current_threshold_p = global_optimized_percentage
            current_tolerance = CONFIG['DEFAULT_RECOGNITION_TOLERANCE_SCALE']
            print(f">>> Using Global Percentage from ODE: {current_threshold_p:.3f}")

        
        dendro_data, templates = perform_clustering_and_visualization(
            model_name,
            full_train_data,
            embeddings,
            threshold_percentage=current_threshold_p,
            tolerance_scale=current_tolerance
        )

        all_dendrograms.append(dendro_data)

        if not templates:
            print("No templates. Skipping tests.")
            continue


    
    plot_combined_dendrograms(all_dendrograms)

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE.")
    print("=" * 60)


if __name__ == '__main__':
    run_evaluation()