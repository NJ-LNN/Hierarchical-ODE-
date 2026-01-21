import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from ODE_model import LatentODE  # 导入模型

# --- Check for torchdiffeq ---
try:
    from torchdiffeq import odeint
except ImportError:
    print("\nERROR: torchdiffeq is not installed. Please run 'pip install torchdiffeq'\n")
    exit()

device = torch.device('cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    'TRAIN_FOLDER_PATH': ,
    'MODEL_SAVE_PATH': ,
    'SEQ_LENGTH': 32,
    'INPUT_DIM': 1,
    'HIDDEN_DIM': 32,
    'EPOCHS': 350,
    'BATCH_SIZE': 32,
    'NUM_AUGMENTATIONS': 20, 
    'AFTER_DOOR_POINTS': 1
}


# ==========================================
# 2. Data Loading & Augmentation
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
    return df


def load_real_data(folder_path, seq_length=32, after_door1=1):
    original_sequences = []
    print(f"Loading real data from: {folder_path}")
    folders = [root for root, dirs, files in os.walk(folder_path) if files]

    for folder_dir in folders:
        files = [os.path.join(folder_dir, f) for f in os.listdir(folder_dir) if
                 f.endswith('.csv') or f.endswith('.txt')]
        for file in files:
            df = read_trash_data(file)
            if df is None or 'manual_flag' not in df.columns: continue

            df['manual_flag'] = df['manual_flag'].apply(lambda x: 'Door' if x != 'default' else x)
            door_indices = df[df['manual_flag'] == 'Door'].index
            if len(door_indices) == 0: continue

            start_idx = door_indices[0] - seq_length + after_door1
            if start_idx < 0: continue

            try:
                seq_data = df['rssi'][start_idx: start_idx + seq_length].values
                if len(seq_data) != seq_length: continue

                seq_tensor = torch.from_numpy(seq_data).float()
                time_points = torch.arange(seq_length).float()
                original_sequences.append((seq_tensor, time_points))
            except:
                continue

    print(f"Successfully loaded {len(original_sequences)} sequences.")
    return original_sequences


def data_augmentation(sample, num_augmentations=10):
    sequence, time_points = sample
    augmented_data = [sample]  

    for _ in range(num_augmentations):
        
        noise = torch.randn_like(sequence) * 0.3
        new_seq = sequence + noise
       
        scaling_factor = np.random.uniform(0.95, 1.05)
        new_seq = new_seq * scaling_factor

        augmented_data.append((new_seq, time_points))
    return augmented_data


def collate_fn_train(batch):
    sequences, times, lengths = [], [], []
    for seq, time in batch:
        sequences.append(seq)
        times.append(time)
        lengths.append(len(seq))

    padded_sequences = pad_sequence(sequences, batch_first=False, padding_value=0)
    max_len = max(lengths)
    time_points = torch.arange(max_len).float()

   
    mask = torch.zeros(max_len, len(batch), 1, dtype=torch.float)
    for i, l in enumerate(lengths):
        mask[:l, i, :] = 1.0

    return padded_sequences, time_points, mask


# ==========================================
# 3. Main Training Execution
# ==========================================
if __name__ == '__main__':
    # 1. Load Data
    full_data = load_real_data(CONFIG['TRAIN_FOLDER_PATH'], CONFIG['SEQ_LENGTH'], CONFIG['AFTER_DOOR_POINTS'])
    if not full_data: exit()

    # Split Train/Test 
    split_idx = int(len(full_data) * 1)  
    train_data_raw = full_data[:split_idx]
    val_data = full_data[split_idx:]

    print(f"Raw Train samples: {len(train_data_raw)}, Validation samples: {len(val_data)}")

    # 2. Augmentation
    print(f"Applying data augmentation (x{CONFIG['NUM_AUGMENTATIONS']})...")
    train_data_augmented = []
    for sample in train_data_raw:
        train_data_augmented.extend(data_augmentation(sample, num_augmentations=CONFIG['NUM_AUGMENTATIONS']))
    print(f"Augmented Train samples: {len(train_data_augmented)}")

    # 3. DataLoader
    train_loader = DataLoader(train_data_augmented, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,
                              collate_fn=collate_fn_train)

    # 4. Initialize Model
    model = LatentODE(CONFIG['INPUT_DIM'], CONFIG['HIDDEN_DIM']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 5. Training Loop
    print("--- Training Phase ---")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, t, mask = batch
            x, t, mask = x.to(device), t.to(device), mask.to(device)
            x = x.unsqueeze(-1)  # (Seq, Batch, 1)

            optimizer.zero_grad()
            recon = model(x, t, mask)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}, Avg Loss: {total_loss / len(train_loader):.5f}")

    # 6. Save Model
    os.makedirs(os.path.dirname(CONFIG['MODEL_SAVE_PATH']), exist_ok=True)
    torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
    print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")