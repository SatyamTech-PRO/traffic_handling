#!/usr/bin/env python3
"""
=============================================================================
Validation Script: MITRE ATT&CK Stage Mapping Thresholds vs Ground Truth
=============================================================================
This script loads the preprocessed state dataset (states.npy, labels.npy, 
states.csv) and evaluates how accurately the rule-based map_to_mitre_stage() 
function categorizes real network attack windows into standard MITRE tactics.

Outputs:
  1. Confusion matrix / cross-tabulation of Real Label vs Predicted MITRE Stage
  2. Per-attack-category mapping accuracy (% aligned vs % misclassified to Normal)
  3. Actionable diagnostic warnings suggesting data-driven threshold adjustments
     if an attack category is misclassified to 'Normal' > 50% of the time.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# Import the actual mapping function and configured thresholds from train_world_model.py
from train_world_model import (
    map_to_mitre_stage,
    FEATURE_NAMES,
    THRESH_RECON_PORTS,
    THRESH_IMPACT_FLAGS,
    THRESH_ACCESS_FLAGS,
    THRESH_IMPACT_PKTS,
    THRESH_DDOS_PKTS,
    THRESH_BRUTE_BYTES_MAX,
    THRESH_BRUTE_PORTS_MAX,
    THRESH_EXFIL_BYTES,
    THRESH_EXFIL_PORTS_MAX,
    THRESH_LATERAL_CONNS,
    THRESH_LATERAL_PORTS,
    THRESH_C2_IAT_VAR_MAX,
    THRESH_C2_BYTES_MAX,
)

STATES_NPY = "states.npy"
LABELS_NPY = "labels.npy"
STATES_CSV = "states.csv"


def validate_mitre_thresholds():
    # 1. Verification of inputs
    if not os.path.exists(STATES_NPY) or not os.path.exists(STATES_CSV):
        raise FileNotFoundError(
            "Missing 'states.npy' or 'states.csv'. Run preprocess_traffic.py on "
            "the CIC-IDS dataset first before running validation."
        )

    states = np.load(STATES_NPY)
    labels = np.load(LABELS_NPY) if os.path.exists(LABELS_NPY) else np.zeros(len(states))
    df = pd.read_csv(STATES_CSV)

    if "dominant_attack_type" not in df.columns:
        raise ValueError(
            "Column 'dominant_attack_type' not found in states.csv! "
            "Ensure preprocess_traffic.py has been updated to compute and export "
            "the per-window dominant attack type."
        )

    dominant_labels = df["dominant_attack_type"].astype(str).values
    n_windows = len(states)

    if len(dominant_labels) != n_windows:
        raise ValueError(
            f"Row count mismatch: states.npy has {n_windows} rows, "
            f"states.csv has {len(dominant_labels)} rows."
        )

    print("\n" + "=" * 80)
    print(" MITRE ATT&CK THRESHOLD VALIDATION & ACCURACY AUDIT")
    print("=" * 80)
    print(f" Total Analyzed Observation Windows: {n_windows:,}")
    print(f" State Feature Dimensionality      : {states.shape[1]}")
    print("=" * 80 + "\n")

    # 2. Run map_to_mitre_stage on every window
    predicted_stages = [map_to_mitre_stage(states[i]) for i in range(n_windows)]

    # 3. Compute Confusion Table (Real Label vs Predicted MITRE Stage)
    results_df = pd.DataFrame({
        "Real_Label": dominant_labels,
        "Predicted_MITRE_Stage": predicted_stages,
    })

    confusion_counts = (
        results_df.groupby(["Real_Label", "Predicted_MITRE_Stage"])
        .size()
        .reset_index(name="Count")
    )
    confusion_counts.sort_values(by=["Real_Label", "Count"], ascending=[True, False], inplace=True)

    print("--- CONFUSION MAPPING TABLE: REAL LABEL vs PREDICTED MITRE STAGE ---")
    print(f"{'Real Label (dominant)':<26} | {'Predicted MITRE Stage':<24} | {'Count':>7}")
    print("-" * 27 + "|" + "-" * 26 + "|" + "-" * 9)
    for _, row in confusion_counts.iterrows():
        print(f"{row['Real_Label']:<26} | {row['Predicted_MITRE_Stage']:<24} | {row['Count']:>7,}")
    print("-" * 65 + "\n")

    # 4. Per-Category Accuracy & Misclassification Analysis
    print("--- ATTACK CATEGORY CONVERSION SUMMARY ---")
    unique_labels = sorted(list(set(dominant_labels)))
    attack_labels = [l for l in unique_labels if l.strip().upper() != "BENIGN"]

    if not attack_labels:
        print("[WARNING] Dataset contains only 'BENIGN' windows! No attacks to benchmark.")
        return

    # Expected mappings for standard attacks (Real MITRE ATT&CK taxonomy)
    # Volumetric flooding attacks belong to Impact (TA0040, T1498/T1499)
    # Brute-force credential guessing belongs to Initial Access (TA0001, T1110)
    EXPECTED_TARGET_STAGES = {
        "PORTSCAN": ["Reconnaissance"],
        "DDOS": ["Impact (DoS/DDoS)"],
        "DOS HULK": ["Impact (DoS/DDoS)"],
        "DOS SLOWHTTPTEST": ["Impact (DoS/DDoS)"],
        "DOS SLOWLORIS": ["Impact (DoS/DDoS)"],
        "FTP-PATATOR": ["Initial Access"],
        "SSH-PATATOR": ["Initial Access"],
        "BOT": ["Command & Control", "Lateral Movement"],
        "INFILTRATION": ["Initial Access", "Lateral Movement"],
    }

    warnings_needed = []

    for atk in attack_labels:
        mask = dominant_labels == atk
        total_atk_windows = int(np.sum(mask))
        atk_states = states[mask]
        atk_predictions = [predicted_stages[i] for i in range(n_windows) if mask[i]]
        pred_counts = pd.Series(atk_predictions).value_counts()

        normal_count = pred_counts.get("Normal", 0)
        normal_pct = (normal_count / total_atk_windows) * 100.0

        # Check matched against expected MITRE stage
        expected_stages = EXPECTED_TARGET_STAGES.get(atk.upper(), ["Initial Access"])
        matched_count = sum(pred_counts.get(st, 0) for st in expected_stages)
        matched_pct = (matched_count / total_atk_windows) * 100.0

        print(f"\n[Category: {atk}] (Total Windows: {total_atk_windows})")
        for st, c in pred_counts.items():
            pct = (c / total_atk_windows) * 100.0
            print(f"   -> {st:<22}: {c:>4} windows ({pct:5.1f}%)")

        print(f"   Accuracy towards expected {expected_stages}: {matched_pct:5.1f}%")
        print(f"   Misclassified as 'Normal': {normal_pct:5.1f}%")

        # Threshold strictness check (> 50% mapped to Normal)
        if normal_pct > 50.0:
            warnings_needed.append((atk, total_atk_windows, normal_pct, atk_states))

    # 5. Diagnostic Warnings and Precise Suggested Adjustments
    if warnings_needed:
        print("\n" + "=" * 80)
        print(" [!!! ACTIONABLE MITRE THRESHOLD CALIBRATION WARNINGS !!!]")
        print("=" * 80)

        for atk, count, normal_pct, atk_states in warnings_needed:
            print(f"\n[CRITICAL WARNING] Attack '{atk}' is mapped to 'Normal' {normal_pct:.1f}% of the time!")
            print(f"  Reason: The current detection thresholds are too strict for normalized feature distributions.")

            atk_upper = atk.upper()
            if "PORTSCAN" in atk_upper:
                # PortScan is primarily governed by unique_dest_ports (index 2)
                ports_col = atk_states[:, 2]
                mean_ports = float(np.mean(ports_col))
                p50_ports = float(np.percentile(ports_col, 50))
                p75_ports = float(np.percentile(ports_col, 75))
                suggested = round(max(0.1, p50_ports - 0.2), 2)
                
                print(f"  Diagnostics for 'unique_dest_ports' (Feature 2) in '{atk}' windows:")
                print(f"    - Current Threshold  : THRESH_RECON_PORTS = {THRESH_RECON_PORTS}")
                print(f"    - Actual Mean        : {mean_ports:+.3f}")
                print(f"    - Actual 50th %ile   : {p50_ports:+.3f}")
                print(f"    - Actual 75th %ile   : {p75_ports:+.3f}")
                print(f"  => SUGGESTION: Lower THRESH_RECON_PORTS from {THRESH_RECON_PORTS} to {suggested} "
                      f"(or THRESH_RECON_PORTS - 0.3) to capture scanning traffic without false positives.")

            elif "DOS" in atk_upper or "DDOS" in atk_upper or "PATATOR" in atk_upper:
                # DoS / Brute Force is governed by sum_syn_ack_rst_flags (index 1) or bytes/s (index 3)
                flags_col = atk_states[:, 1]
                mean_flags = float(np.mean(flags_col))
                p50_flags = float(np.percentile(flags_col, 50))
                suggested_flags = round(max(0.2, p50_flags - 0.2), 2)

                print(f"  Diagnostics for 'sum_syn_ack_rst_flags' (Feature 1) in '{atk}' windows:")
                print(f"    - Current Threshold  : THRESH_ACCESS_FLAGS = {THRESH_ACCESS_FLAGS}")
                print(f"    - Actual Mean        : {mean_flags:+.3f}")
                print(f"    - Actual 50th %ile   : {p50_flags:+.3f}")
                print(f"  => SUGGESTION: Lower THRESH_ACCESS_FLAGS from {THRESH_ACCESS_FLAGS} to {suggested_flags} "
                      f"(or THRESH_ACCESS_FLAGS - 0.3).")
            else:
                conns_col = atk_states[:, 0]
                print(f"  Diagnostics for 'total_connections' (Feature 0) in '{atk}' windows:")
                print(f"    - Actual Mean: {float(np.mean(conns_col)):+.3f}, 50th %ile: {float(np.percentile(conns_col, 50)):+.3f}")

        print("=" * 80 + "\n")
    else:
        print("\n[SUCCESS] All attack categories achieve >= 50% detection rate into active MITRE tactics.")
        print("          No thresholds are excessively strict.\n")


if __name__ == "__main__":
    validate_mitre_thresholds()
