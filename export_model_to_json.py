#!/usr/bin/env python3
import json
import numpy as np
import torch
import torch.nn as nn
import joblib

def main():
    print("=== EXPORTING PYTORCH CYBER WORLD MODEL TO JSON ===")
    
    # 1. Load weights
    checkpoint = torch.load("world_model_real.pt", map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # 2. Load scaler
    scaler_obj = joblib.load("state_scaler.joblib")
    scaler = scaler_obj["scaler"] if isinstance(scaler_obj, dict) and "scaler" in scaler_obj else scaler_obj
    
    feature_names = [
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
    num_features = len(feature_names)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 1)

    # 3. Print the full head structure
    print("\n--- Full Model Architecture & Head Structure ---")
    print("Recurrent Backbone:")
    print(f"  LSTM(input_size={num_features}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True)")
    print("Head 1 (State Dynamics Head):")
    print(f"  Linear(in_features={hidden_size}, out_features={num_features}, bias=True)")
    print("Head 2 (Attack Probability Head):")
    print(f"  Sequential(")
    print(f"    (0): Linear(in_features={hidden_size}, out_features=1, bias=True)")
    print(f"    (1): Sigmoid()")
    print(f"  )")
    
    # 4. Extract tensors to serializable python lists
    export_data = {
        "metadata": {
            "model_name": "CyberWorldModel",
            "framework": "PyTorch",
            "num_features": num_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "gate_order": ["input", "forget", "cell", "output"],
            "feature_names": feature_names
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist() if hasattr(scaler, 'var_') else (scaler.scale_**2).tolist(),
        },
        "weights": {
            "lstm_weight_ih": state_dict["lstm.weight_ih_l0"].cpu().numpy().tolist(),  # (256, 14)
            "lstm_weight_hh": state_dict["lstm.weight_hh_l0"].cpu().numpy().tolist(),  # (256, 64)
            "lstm_bias_ih": state_dict["lstm.bias_ih_l0"].cpu().numpy().tolist(),      # (256,)
            "lstm_bias_hh": state_dict["lstm.bias_hh_l0"].cpu().numpy().tolist(),      # (256,)
            "state_head_weight": state_dict["state_head.weight"].cpu().numpy().tolist(),  # (14, 64)
            "state_head_bias": state_dict["state_head.bias"].cpu().numpy().tolist(),      # (14,)
            "attack_head_weight": state_dict["attack_head.0.weight"].cpu().numpy().tolist(), # (1, 64)
            "attack_head_bias": state_dict["attack_head.0.bias"].cpu().numpy().tolist(),     # (1,)
        },
        "mitre_thresholds": {
            "THRESH_RECON_PORTS": 1.01,
            "THRESH_IMPACT_FLAGS": 0.93,
            "THRESH_IMPACT_PKTS": 0.94,
            "THRESH_BRUTE_BYTES_MAX": -0.45,
            "THRESH_BRUTE_PORTS_MAX": -0.48,
            "THRESH_EXFIL_BYTES": 1.5,
            "THRESH_EXFIL_PORTS_MAX": 0.5,
            "THRESH_LATERAL_CONNS": 1.0,
            "THRESH_LATERAL_PORTS": 0.6,
            "THRESH_C2_IAT_VAR_MAX": -0.4,
            "THRESH_C2_BYTES_MAX": 0.3
        }
    }
    
    # Save to src/data/world_model_weights_real.json and root
    with open("src/data/world_model_weights_real.json", "w") as f:
        json.dump(export_data, f, indent=2)
        
    with open("world_model_weights_real.json", "w") as f:
        json.dump(export_data, f, indent=2)
        
    print(f"\n[Export Success] Successfully exported weights to src/data/world_model_weights_real.json ({len(json.dumps(export_data))} bytes)")
    
if __name__ == "__main__":
    main()
