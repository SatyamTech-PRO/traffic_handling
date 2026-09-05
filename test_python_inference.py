#!/usr/bin/env python3
import json
import numpy as np
import torch
import torch.nn as nn

class CyberWorldModel(nn.Module):
    def __init__(self, num_features=7, hidden_size=64, num_layers=1):
        super(CyberWorldModel, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
        self.state_head = nn.Linear(hidden_size, num_features)
        self.attack_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        next_state = self.state_head(last_hidden)
        attack_prob = self.attack_head(last_hidden)
        return next_state, attack_prob

def main():
    model = CyberWorldModel()
    checkpoint = torch.load("world_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open("demo_scenarios.json") as f:
        demo = json.load(f)

    print("=== PYTHON STEP-BY-STEP FORECAST RESULTS ===")
    for scenario_name in ["normal", "portscan", "dos", "ftp_patator"]:
        print(f"\n--- Scenario: {scenario_name} ---")
        history = np.array(demo[scenario_name]["history"])
        rolling = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
        
        for step in range(1, 6):
            with torch.no_grad():
                pred_s, p_att = model(rolling)
            prob = float(p_att.item())
            state_vec = pred_s.squeeze(0).numpy()
            print(f"Step {step} (+{step*10}s): prob={prob:.6f}, state={np.round(state_vec[:3], 4).tolist()}...")
            rolling = torch.cat([rolling[:, 1:, :], pred_s.unsqueeze(1)], dim=1)

if __name__ == "__main__":
    main()
