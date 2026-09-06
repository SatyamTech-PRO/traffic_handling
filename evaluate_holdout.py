#!/usr/bin/env python3
"""
evaluate_holdout.py - Evaluates world_model_real.pt on a holdout session/time-range
that was not part of training (e.g., File 17: Friday 7/7/2017 and File 4: Tuesday 4/7/2017).
Reports F1, precision, recall, and false positive rate.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from train_world_model import CyberWorldModel, FEATURE_NAMES, WINDOW_SIZE

def load_holdout_states(file_paths, scaler):
    dfs = []
    for fp in file_paths:
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            dfs.append(df)
            
    if not dfs:
        raise FileNotFoundError(f"No holdout state files found in: {file_paths}")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Fill any missing values in feature columns
    X_raw = combined_df[FEATURE_NAMES].fillna(0).values.astype(np.float64)
    X_norm = scaler.transform(X_raw)
    
    # Ground truth labels
    if "attack_fraction" in combined_df.columns:
        y_raw = (combined_df["attack_fraction"] > 0).astype(int).values
    elif "dominant_attack_type" in combined_df.columns:
        y_raw = (combined_df["dominant_attack_type"].astype(str).str.upper() != "BENIGN").astype(int).values
    else:
        raise ValueError("Cannot determine attack labels from holdout dataframe.")
        
    return X_norm, y_raw

def evaluate(model_path="world_model_real.pt", scaler_path="state_scaler.joblib", holdout_files=None, threshold=0.5):
    if holdout_files is None:
        holdout_files = ["states_file17.csv"]
        
    # 1. Load scaler
    scaler_obj = joblib.load(scaler_path)
    scaler = scaler_obj["scaler"] if isinstance(scaler_obj, dict) and "scaler" in scaler_obj else scaler_obj
    
    # 2. Load holdout data
    X_norm, y_raw = load_holdout_states(holdout_files, scaler)
    num_samples = len(X_norm) - WINDOW_SIZE
    if num_samples <= 0:
        raise ValueError("Insufficient holdout windows for evaluation.")
        
    X_seq = []
    y_target = []
    for i in range(num_samples):
        X_seq.append(X_norm[i : i + WINDOW_SIZE])
        y_target.append(y_raw[i + WINDOW_SIZE])
        
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_true = np.array(y_target)
    
    # 3. Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    num_features = len(FEATURE_NAMES)
    hidden_size = checkpoint.get("hidden_size", 64)
    num_layers = checkpoint.get("num_layers", 1)
    
    model = CyberWorldModel(num_features=num_features, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 4. Predict
    with torch.no_grad():
        _, pred_probs = model(X_tensor)
        probs = pred_probs.squeeze().cpu().numpy()
        
    preds = (probs >= threshold).astype(int)
    
    # 5. Compute metrics
    # If all 0 or 1, handle zero_division cleanly
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_evaluated": len(y_true),
        "attack_windows": int(y_true.sum()),
        "threshold": threshold,
        "probs": probs,
        "y_true": y_true
    }

if __name__ == "__main__":
    # Evaluate on Friday session (File 17 - PortScan / DoS / Patator)
    res = evaluate(holdout_files=["states_file17.csv"])
    print(f"F1 Score:               {res['f1']:.4f}")
    print(f"Precision:              {res['precision']:.4f}")
    print(f"Recall:                 {res['recall']:.4f}")
    print(f"False Positive Rate:    {res['fpr']:.4f}")
