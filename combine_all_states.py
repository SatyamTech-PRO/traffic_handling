#!/usr/bin/env python3
"""
combine_all_states.py - Aggregate all processed states_file*.csv into master states.csv,
states.npy, and fit/save state_scaler.joblib.
"""

import sys
import os

import glob
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    # 7 Flow-level indicators
    "total_connections",
    "sum_syn_ack_rst_flags",
    "unique_dest_ports",
    "avg_bytes_per_sec",
    "avg_packets_per_sec",
    "avg_iat_mean",
    "avg_iat_variance",
    # 7 Packet-level indicators
    "pkt_ttl_variance",
    "pkt_tcp_window_avg",
    "pkt_ip_frag_fraction",
    "pkt_payload_mean",
    "pkt_payload_std",
    "pkt_retrans_count",
    "pkt_portscan_delta_std"
]

def combine_states():
    state_files = sorted(list(set(glob.glob("states_file*.csv") + glob.glob("data/states_file*.csv"))))
    if not state_files:
        print("[ERROR] No states_file*.csv found in root or data/ directory!")
        sys.exit(1)

    print(f"Found {len(state_files)} state file(s): {state_files}")
    dfs = []
    for sf in state_files:
        df = pd.read_csv(sf)
        if "source_file" not in df.columns:
            df["source_file"] = os.path.basename(sf)
        print(f"  {sf}: {len(df):,} windows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    if "Timestamp" in combined.columns:
        combined["dt"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
        combined = combined.sort_values("dt").drop(columns=["dt"]).reset_index(drop=True)

    print(f"\nTotal combined windows: {len(combined):,}")
    print(f"Attack breakdown:\n{combined['dominant_attack_type'].value_counts()}")

    # Ensure all feature columns exist and are numeric
    for col in FEATURE_COLS:
        if col not in combined.columns:
            combined[col] = 0.0
        else:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    # Fit StandardScaler (mean=0, std=1 z-score normalization)
    scaler = StandardScaler()
    X = combined[FEATURE_COLS].values
    X_norm = scaler.fit_transform(X)

    # Save normalized columns into dataframe
    for i, col in enumerate(FEATURE_COLS):
        combined[f"norm_{col}"] = X_norm[:, i]

    # Save outputs
    if "attack_fraction" in combined.columns:
        labels = combined["attack_fraction"].values.astype(np.float32)
    else:
        labels = (combined["dominant_attack_type"].astype(str).str.upper() != "BENIGN").values.astype(np.float32)

    combined.to_csv("states.csv", index=False)
    np.save("states.npy", X_norm)
    np.save("labels.npy", labels)
    joblib.dump({"scaler": scaler, "features": FEATURE_COLS}, "state_scaler.joblib")

    print("\n[SUCCESS] Successfully generated:")
    print(f"  - states.csv: {len(combined):,} rows x {len(combined.columns)} columns")
    print(f"  - states.npy: shape {X_norm.shape}, dtype {X_norm.dtype}")
    print(f"  - labels.npy: shape {labels.shape}, dtype {labels.dtype}")
    print(f"  - state_scaler.joblib: fitted scaler on {len(FEATURE_COLS)} features")

if __name__ == "__main__":
    combine_states()
