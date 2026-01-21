import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm


from ResNet_model import ResNetAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


CONFIG = {
    'TRAIN_FOLDER': ,
    'MODEL_SAVE_PATH': ,
    'SEQ_LENGTH': 32,
    'AFTER_DOOR_POINTS': 1,
    'NUM_AUGMENTATIONS': 30,
    'EPOCHS': 200,
    'BATCH_SIZE': 32,
    'HIDDEN_DIM': 32,
    'INPUT_DIM': 1
}



def read_trash_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            row = line.strip().split(',')
            if len(row) >= 7:
                selected_row = row[:4] + row[-3:]
                data.append(selected_row)
    if len(data) < 2: return None
    df = pd.DataFrame(data[1:], columns=data[0])
    if 'rssi' not in df.columns: return None
    try:
        df['rssi'] = pd.to_numeric(df['rssi'])
    except:
        return None
    return df


def load_real_data(folder_path, seq_length=32, after_door1=1):
    sequences = []
    print(f"Loading training data from: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            df = read_trash_data(file_path)
            if df is None or 'manual_flag' not in df.columns: continue

            df['manual_flag'] = df['manual_flag'].apply(lambda x: 'Door' if x != 'default' else x)
            door_indices = df[df['manual_flag'] == 'Door'].index
            if len(door_indices) == 0: continue

            start_idx = door_indices[0] - seq_length + after_door1
            if start_idx < 0: continue

            try:
                seq_data = df['rssi'][start_idx: start_idx + seq_length].values
                if len(seq_data) == seq_length:
                    seq_tensor = torch.from_numpy(seq_data).float()
                    
                    time_points = torch.arange(seq_length).float()
                    sequences.append((seq_tensor, time_points))
            except:
                continue
    print(f"Loaded {len(sequences)} training sequences.")
    return sequences


def data_augmentation(sample, num_augmentations=20):
    sequence, time_points = sample
    augmented = []
    augmented.append(sample)  
    for _ in range(num_augmentations):
        noise = torch.randn_like(sequence) * 0.5
        new_seq = (sequence + noise) * np.random.uniform(0.95, 1.05)
        augmented.append((new_seq, time_points))
    return augmented


def collate_fn(batch):
    sequences, times, lengths = [], [], []
    for seq, time in batch:
        sequences.append(seq)
        times.append(time)
        lengths.append(len(seq))

    if not sequences:
        return None, None, None

    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    max_len = max(lengths)

    
    time_points_for_batch = next((t for s, t in batch if len(s) == max_len), batch[0][1])

    # CNN Mask shape: (N, 1, L)
    mask = torch.zeros(len(batch), 1, max_len, dtype=torch.float)
    for i, l in enumerate(lengths):
        mask[i, :, :l] = 1.0

    return padded_sequences, time_points_for_batch, mask



if __name__ == '__main__':

    raw_data = load_real_data(CONFIG['TRAIN_FOLDER'], CONFIG['SEQ_LENGTH'], CONFIG['AFTER_DOOR_POINTS'])
    if not raw_data: exit()

    aug_data = []
    for s in raw_data:
        aug_data.extend(data_augmentation(s, CONFIG['NUM_AUGMENTATIONS']))

    loader = DataLoader(aug_data, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)


    print("Initializing 1D-ResNet Autoencoder...")
    model = ResNetAutoencoder(input_dim=CONFIG['INPUT_DIM'], hidden_dim=CONFIG['HIDDEN_DIM']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')


    print(f"Starting training for {CONFIG['EPOCHS']} epochs...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
            padded_data, batch_times, mask = batch
            if padded_data is None: continue

            padded_data, batch_times, mask = padded_data.to(device), batch_times.to(device), mask.to(device)


            padded_data = padded_data.unsqueeze(1)

            optimizer.zero_grad()
            reconstruction = model(padded_data, batch_times)

 
            loss_unmasked = criterion(reconstruction, padded_data)
            loss_masked = (loss_unmasked * mask).sum() / mask.sum()

            loss_masked.backward()
            optimizer.step()
            total_loss += loss_masked.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(loader):.5f}")

 
    os.makedirs(os.path.dirname(CONFIG['MODEL_SAVE_PATH']), exist_ok=True)
    torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
    print(f"Model saved to {CONFIG['MODEL_SAVE_PATH']}")