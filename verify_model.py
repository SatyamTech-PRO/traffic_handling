#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import joblib

# Load world model class definition
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

def main():
    print("=== MODEL VERIFICATION & GENERALIZATION CHECK ===")
    
    # Check states and labels
    states = np.load("states.npy")
    labels = np.load("labels.npy")
    print(f"Total dataset windows in states.npy: {len(states)} windows, shape: {states.shape}")
    print(f"Total labels: {len(labels)}, unique values / stats: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.4f}")
    
    # Check scaler
    scaler = joblib.load("scaler.joblib")
    print(f"Scaler mean_: {scaler.mean_}")
    print(f"Scaler scale_: {scaler.scale_}")
    
    # Check model
    model = CyberWorldModel(num_features=7, hidden_size=64, num_layers=1)
    state_dict = torch.load("world_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("\nModel state_dict keys and shapes:")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")
        
    # Check train/val split in training script: split_idx = int(len(states) * 0.8)
    split_idx = int(len(states) * 0.8)
    print(f"\nTraining set: windows 0 to {split_idx-1} ({split_idx} windows)")
    print(f"Validation / Holdout set: windows {split_idx} to {len(states)-1} ({len(states) - split_idx} windows)")
    
    # Check labels distribution across train and holdout
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]
    print(f"Train attack fraction mean: {train_labels.mean():.4f} (attacks: {(train_labels > 0.5).sum()} windows)")
    print(f"Holdout attack fraction mean: {val_labels.mean():.4f} (attacks: {(val_labels > 0.5).sum()} windows)")
    
    # Calibration windows used in demo_scenarios.json:
    # Normal: 180-184
    # PortScan: 60-64
    # DoS: 117-121
    # FTP-Patator: 165-169
    known_windows = set(range(180, 185)) | set(range(60, 65)) | set(range(117, 122)) | set(range(165, 170))
    print(f"\nKnown demo windows set: {sorted(list(known_windows))}")
    
    # Select several NOVEL windows from the holdout set (and from across the dataset)
    # We want input sequences of length 5: windows [w-4:w+1]
    # Let's inspect diverse novel windows:
    test_indices = [
        # Novel windows in the holdout set (split_idx onwards)
        split_idx + 2,
        split_idx + 8,
        split_idx + 15,
        split_idx + 22,
        split_idx + 30,
        # Novel windows in train set that were not in calibration
        25,
        45,
        85,
        140
    ]
    
    print("\nTesting novel windows outside demo scenarios:")
    from train_world_model import map_to_mitre_stage
    
    for end_idx in test_indices:
        if end_idx >= len(states) or end_idx < 4:
            continue
        # Input sequence of 5 windows:
        seq = states[end_idx-4 : end_idx+1]
        x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            next_state, attack_prob = model(x_tensor)
            
        prob_val = attack_prob.item()
        pred_state_np = next_state.squeeze(0).numpy()
        mitre_stage = map_to_mitre_stage(pred_state_np)
        true_label = labels[end_idx]
        is_holdout = end_idx >= split_idx
        
        print(f"Window {end_idx-4}:{end_idx} ({'HOLDOUT' if is_holdout else 'TRAIN'}) | True label: {true_label:.2f} | Pred prob: {prob_val:.4f} | MITRE: {mitre_stage}")
        
    print("\nVerification script finished.")

if __name__ == "__main__":
    main()
