import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

FEATURE_NAMES = [
    "total_connections",
    "sum_syn_ack_rst_flags",
    "unique_dest_ports",
    "avg_bytes_per_sec",
    "avg_packets_per_sec",
    "avg_iat_mean",
    "avg_iat_variance",
    "pkt_ttl_variance",
    "pkt_tcp_window_avg",
    "pkt_ip_frag_fraction",
    "pkt_payload_mean",
    "pkt_payload_std",
    "pkt_retrans_count",
    "pkt_portscan_delta_std"
]

WINDOW_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20
ATTACK_LOSS_WEIGHT = 0.5

class IndexedSequenceDataset(Dataset):
    def __init__(self, states: np.ndarray, labels: np.ndarray, target_indices: np.ndarray, window_size: int = 5):
        self.samples = []
        for t in target_indices:
            if t >= window_size:
                seq_x = states[t - window_size : t]
                target_state = states[t]
                target_attack = labels[t]
                self.samples.append((seq_x, target_state, target_attack, t))
                
        self.X = torch.tensor(np.array([s[0] for s in self.samples]), dtype=torch.float32)
        self.target_states = torch.tensor(np.array([s[1] for s in self.samples]), dtype=torch.float32)
        self.target_attacks = torch.tensor(np.array([s[2] for s in self.samples]), dtype=torch.float32).unsqueeze(1)
        self.target_indices = np.array([s[3] for s in self.samples])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.X[idx], self.target_states[idx], self.target_attacks[idx]

class CyberWorldModel(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 1):
        super(CyberWorldModel, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.state_head = nn.Linear(hidden_size, num_features)
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        next_state = self.state_head(last_hidden)
        attack_prob = self.attack_head(last_hidden)
        return next_state, attack_prob

def train_v2():
    states = np.load("states.npy")
    labels = np.load("labels.npy")
    train_idx = np.load("train_indices.npy")
    val_idx = np.load("val_indices.npy")
    
    train_dataset = IndexedSequenceDataset(states, labels, train_idx, window_size=WINDOW_SIZE)
    val_dataset = IndexedSequenceDataset(states, labels, val_idx, window_size=WINDOW_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate pos_weight on the TRAINING split specifically
    y_train = train_dataset.target_attacks.squeeze().numpy()
    n_benign = (y_train == 0.0).sum()
    n_attack = (y_train > 0.0).sum()
    pos_weight = float(n_benign / n_attack)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = len(FEATURE_NAMES)
    model = CyberWorldModel(num_features=num_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    
    criterion_state = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_bce = 0.0
        
        for batch_x, batch_target_state, batch_target_attack in train_loader:
            batch_x = batch_x.to(device)
            batch_target_state = batch_target_state.to(device)
            batch_target_attack = batch_target_attack.to(device)
            
            optimizer.zero_grad()
            pred_state, pred_attack = model(batch_x)
            
            loss_mse = criterion_state(pred_state, batch_target_state)
            
            # Class-weighted BCE loss using pos_weight on training split
            weights = torch.where(
                batch_target_attack > 0.0,
                torch.tensor(pos_weight, device=device),
                torch.tensor(1.0, device=device)
            )
            loss_bce = (F.binary_cross_entropy(pred_attack, batch_target_attack, reduction='none') * weights).mean()
            
            total_loss = loss_mse + ATTACK_LOSS_WEIGHT * loss_bce
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item() * len(batch_x)
            epoch_mse += loss_mse.item() * len(batch_x)
            epoch_bce += loss_bce.item() * len(batch_x)
            
        train_loss = epoch_loss / len(train_dataset)
        train_mse = epoch_mse / len(train_dataset)
        train_bce = epoch_bce / len(train_dataset)
        
        # Validation
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_mse = 0.0
        val_epoch_bce = 0.0
        
        with torch.no_grad():
            for val_bx, val_by_state, val_by_attack in val_loader:
                val_bx = val_bx.to(device)
                val_by_state = val_by_state.to(device)
                val_by_attack = val_by_attack.to(device)
                
                v_pred_state, v_pred_attack = model(val_bx)
                v_loss_mse = criterion_state(v_pred_state, val_by_state)
                
                val_weights = torch.where(
                    val_by_attack > 0.0,
                    torch.tensor(pos_weight, device=device),
                    torch.tensor(1.0, device=device)
                )
                v_loss_bce = (F.binary_cross_entropy(v_pred_attack, val_by_attack, reduction='none') * val_weights).mean()
                v_total = v_loss_mse + ATTACK_LOSS_WEIGHT * v_loss_bce
                
                val_epoch_loss += v_total.item() * len(val_bx)
                val_epoch_mse += v_loss_mse.item() * len(val_bx)
                val_epoch_bce += v_loss_bce.item() * len(val_bx)
                
        val_loss = val_epoch_loss / len(val_dataset)
        val_mse = val_epoch_mse / len(val_dataset)
        val_bce = val_epoch_bce / len(val_dataset)

    # Print only final train/val loss
    print(f"Final Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Weighted-BCE: {train_bce:.4f})")
    print(f"Final Val Loss:   {val_loss:.4f} (MSE: {val_mse:.4f}, Weighted-BCE: {val_bce:.4f})")

    # Save to world_model_real_v2.pt
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_features': num_features,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'window_size': WINDOW_SIZE,
        'pos_weight': pos_weight
    }, "world_model_real_v2.pt")
    
    return model

if __name__ == "__main__":
    train_v2()
