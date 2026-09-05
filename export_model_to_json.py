#!/usr/bin/env python3
import json
import numpy as np
import torch
import torch.nn as nn
import joblib

def main():
    print("=== EXPORTING PYTORCH CYBER WORLD MODEL TO JSON ===")
    
    # 1. Load weights
    checkpoint = torch.load("world_model.pt", map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # 2. Load scaler
    scaler = joblib.load("scaler.joblib")
    
    # 3. Print the full head structure
    print("\n--- Full Model Architecture & Head Structure ---")
    print("Recurrent Backbone:")
    print(f"  LSTM(input_size=7, hidden_size=64, num_layers=1, batch_first=True)")
    print("Head 1 (State Dynamics Head):")
    print(f"  Linear(in_features=64, out_features=7, bias=True)")
    print("Head 2 (Attack Probability Head):")
    print(f"  Sequential(")
    print(f"    (0): Linear(in_features=64, out_features=1, bias=True)")
    print(f"    (1): Sigmoid()")
    print(f"  )")
    
    # 4. Extract tensors to serializable python lists
    export_data = {
        "metadata": {
            "model_name": "CyberWorldModel",
            "framework": "PyTorch",
            "num_features": 7,
            "hidden_size": 64,
            "num_layers": 1,
            "gate_order": ["input", "forget", "cell", "output"],
            "feature_names": [
                "total_connections",
                "sum_syn_ack_rst_flags",
                "unique_dest_ports",
                "avg_bytes_per_sec",
                "avg_packets_per_sec",
                "avg_iat_mean",
                "avg_iat_variance"
            ]
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist() if hasattr(scaler, 'var_') else (scaler.scale_**2).tolist(),
        },
        "weights": {
            "lstm_weight_ih": state_dict["lstm.weight_ih_l0"].cpu().numpy().tolist(),  # (256, 7)
            "lstm_weight_hh": state_dict["lstm.weight_hh_l0"].cpu().numpy().tolist(),  # (256, 64)
            "lstm_bias_ih": state_dict["lstm.bias_ih_l0"].cpu().numpy().tolist(),      # (256,)
            "lstm_bias_hh": state_dict["lstm.bias_hh_l0"].cpu().numpy().tolist(),      # (256,)
            "state_head_weight": state_dict["state_head.weight"].cpu().numpy().tolist(),  # (7, 64)
            "state_head_bias": state_dict["state_head.bias"].cpu().numpy().tolist(),      # (7,)
            "attack_head_weight": state_dict["attack_head.0.weight"].cpu().numpy().tolist(), # (1, 64)
            "attack_head_bias": state_dict["attack_head.0.bias"].cpu().numpy().tolist(),     # (1,)
        },
        "mitre_thresholds": {
            "THRESH_RECON_PORTS": 1.01,
            "THRESH_IMPACT_FLAGS": 2.0,
            "THRESH_IMPACT_PKTS": 2.0,
            "THRESH_BRUTE_BYTES_MAX": -0.4,
            "THRESH_BRUTE_PORTS_MAX": -0.8,
            "THRESH_EXFIL_BYTES": 1.5,
            "THRESH_EXFIL_PORTS_MAX": 0.5,
            "THRESH_LATERAL_CONNS": 1.0,
            "THRESH_LATERAL_PORTS": 0.6,
            "THRESH_C2_IAT_VAR_MAX": -0.4,
            "THRESH_C2_BYTES_MAX": 0.3
        }
    }
    
    # Save to src/data/world_model_weights.json and root
    with open("src/data/world_model_weights.json", "w") as f:
        json.dump(export_data, f, indent=2)
        
    with open("world_model_weights.json", "w") as f:
        json.dump(export_data, f, indent=2)
        
    print(f"\n[Export Success] Successfully exported weights to src/data/world_model_weights.json ({len(json.dumps(export_data))} bytes)")
    
if __name__ == "__main__":
    main()
