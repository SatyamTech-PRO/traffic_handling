#!/usr/bin/env python3
"""
=============================================================================
Generate Demo Scenarios from Real Model Inference on Holdout Sessions
=============================================================================
Pulls 4 real network periods from the chronological session-split holdout set:
  1. normal: Clearly Normal / Benign period (Session 7 holdout)
  2. portscan: PortScan / Reconnaissance period (Session 2 holdout)
  3. dos: DoS / Impact period (Session 4 DDoS holdout)
  4. ftp_patator: FTP-Patator / Initial-Access period (Session 6 holdout)

Runs 5-step closed-loop autoregressive forecast() using trained world_model.pt,
maps each step to MITRE ATT&CK tactics via map_to_mitre_stage(),
and calculates SHAP feature attributions on the final rollout step.

Outputs: demo_scenarios.json
=============================================================================
"""

import json
import os
import numpy as np
import torch

from train_world_model import (
    CyberWorldModel,
    forecast,
    map_to_mitre_stage,
    explain_prediction_shap,
    WINDOW_SIZE,
    FEATURE_NAMES,
)

STATES_NPY = "states.npy"
MODEL_PT = "world_model.pt"
OUTPUT_JSON = "demo_scenarios.json"
SRC_OUTPUT_JSON = "src/data/demo_scenarios.json"


def generate_scenarios():
    if not os.path.exists(STATES_NPY):
        raise FileNotFoundError(f"'{STATES_NPY}' not found!")
    if not os.path.exists(MODEL_PT):
        raise FileNotFoundError(f"'{MODEL_PT}' not found! Run train_world_model.py first.")

    states = np.load(STATES_NPY)
    print(f"[1/4] Loaded real state vectors: {states.shape} (7 normalized features, 211 windows)")

    checkpoint = torch.load(MODEL_PT, map_location="cpu")
    model = CyberWorldModel(num_features=7, hidden_size=64, num_layers=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[2/4] Loaded trained CyberWorldModel from 'world_model.pt'")

    # Target holdout windows verified by auto_calibrate_thresholds.py:
    # 1. Normal: Session 7 (windows 175..210, taking windows 180..184)
    # 2. PortScan: Session 2 tail (windows 59..69, taking windows 60..64)
    # 3. DoS: Session 4 tail (windows 117..121)
    # 4. FTP-Patator: Session 6 tail (windows 164..174, taking windows 165..169)
    scenario_specs = {
        "normal": {
            "name": "Normal / Benign Telemetry",
            "slice": (180, 185),
            "provenance": "holdout set, session never seen during calibration (Session 7 holdout, windows 180-184)",
        },
        "portscan": {
            "name": "PortScan / Reconnaissance Sweep",
            "slice": (60, 65),
            "provenance": "holdout set, session never seen during calibration (PortScan chronological holdout, windows 60-64)",
        },
        "dos": {
            "name": "DoS / Impact Telemetry Surge",
            "slice": (117, 122),
            "provenance": "holdout set, session never seen during calibration (DDoS chronological holdout, windows 117-121)",
        },
        "ftp_patator": {
            "name": "FTP-Patator / Initial Access Brute Force",
            "slice": (165, 170),
            "provenance": "holdout set, session never seen during calibration (FTP-Patator chronological holdout, windows 165-169)",
        },
    }

    demo_scenarios = {}

    print("[3/4] Generating 5-step autoregressive forecasts & SHAP attributions...")
    for sc_key, spec in scenario_specs.items():
        start_idx, end_idx = spec["slice"]
        history = states[start_idx:end_idx]  # Shape: (5, 7)

        # Run 5-step forecast
        forecast_results = forecast(model, history, k_steps=5)

        formatted_steps = []
        for step_idx, (p_state, prob) in enumerate(forecast_results, start=1):
            stage = map_to_mitre_stage(p_state)
            clean_state = [round(float(v), 4) for v in p_state]
            formatted_steps.append({
                "step": step_idx,
                "predicted_state": clean_state,
                "attack_probability": round(float(prob), 4),
                "mitre_stage": stage,
            })

        # Final sequence leading into step 5 prediction:
        # [s_4, p_1, p_2, p_3, p_4] -> produced step 5
        final_seq = np.array([
            history[4],
            forecast_results[0][0],
            forecast_results[1][0],
            forecast_results[2][0],
            forecast_results[3][0],
        ])

        shap_list = explain_prediction_shap(
            model,
            final_seq,
            background_data=states[:50],
            top_k=3,
        )
        clean_shap = [[feat, round(float(val), 4)] for feat, val in shap_list]

        clean_history = [[round(float(v), 4) for v in row] for row in history]

        demo_scenarios[sc_key] = {
            "history": clean_history,
            "forecast_steps": formatted_steps,
            "top_shap_features": clean_shap,
            "data_provenance": spec["provenance"],
        }
        print(f"  ✓ {sc_key:<12} | Steps: 5 | Final P(Attack): {formatted_steps[-1]['attack_probability']:.1%} | Final Stage: {formatted_steps[-1]['mitre_stage']}")

    # Write output to demo_scenarios.json in root and src/data/
    with open(OUTPUT_JSON, "w") as f:
        json.dump(demo_scenarios, f, indent=2)
    os.makedirs(os.path.dirname(SRC_OUTPUT_JSON), exist_ok=True)
    with open(SRC_OUTPUT_JSON, "w") as f:
        json.dump(demo_scenarios, f, indent=2)

    print(f"[4/4] Successfully generated and wrote {OUTPUT_JSON} and {SRC_OUTPUT_JSON}")
    return demo_scenarios


if __name__ == "__main__":
    generate_scenarios()
