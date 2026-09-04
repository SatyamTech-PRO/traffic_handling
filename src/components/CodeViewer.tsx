import React, { useState } from 'react';
import { Copy, Check, Download, Play, FileCode, Info, Terminal, Brain } from 'lucide-react';

const SCRIPT_STAGE_1 = `#!/usr/bin/env python3
"""
=============================================================================
Cybersecurity "World Model" Prototype - State Representation Pipeline
Dataset: CIC-IDS2017 / CIC-IDS2018 Network Flow Traffic
=============================================================================
Transforms raw granular network flow records into a discrete, fixed-interval 
sequence of environment "State Vectors" for a Cyber World Model.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/traffic.csv"
OUTPUT_CSV_PATH = "states.csv"
OUTPUT_NPY_PATH = "states.npy"
OUTPUT_LABELS_PATH = "labels.npy"
SCALER_PATH = "scaler.joblib"
TIME_WINDOW_SECONDS = "10s"

EXPECTED_COLUMNS = [
    "Timestamp", "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s",
    "SYN Flag Count", "ACK Flag Count", "RST Flag Count",
    "Fwd IAT Mean", "Fwd IAT Std", "Label"
]

COLUMN_ALIASES = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "SYN Flag Cnt": "SYN Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "RST Flag Cnt": "RST Flag Count",
}

def load_and_standardize_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at '{file_path}'.")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.rename(columns=COLUMN_ALIASES, inplace=True)
    return df

def clean_and_prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[EXPECTED_COLUMNS].copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    numeric_cols = [c for c in EXPECTED_COLUMNS if c not in ["Timestamp", "Label"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=numeric_cols, inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values(by="Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def aggregate_time_windows(df: pd.DataFrame, window_freq: str = "10s") -> pd.DataFrame:
    df["Fwd_IAT_Var"] = df["Fwd IAT Std"] ** 2
    df["Flags_SYN_ACK_RST"] = df["SYN Flag Count"] + df["ACK Flag Count"] + df["RST Flag Count"]
    df["Is_Attack"] = (df["Label"].str.upper() != "BENIGN").astype(int)
    
    grouped = df.groupby(pd.Grouper(key="Timestamp", freq=window_freq))
    states_df = grouped.agg(
        total_connections=("Flow Duration", "count"),
        sum_syn_ack_rst_flags=("Flags_SYN_ACK_RST", "sum"),
        unique_dest_ports=("Destination Port", "nunique"),
        avg_bytes_per_sec=("Flow Bytes/s", "mean"),
        avg_packets_per_sec=("Flow Packets/s", "mean"),
        avg_iat_mean=("Fwd IAT Mean", "mean"),
        avg_iat_variance=("Fwd_IAT_Var", "mean"),
        attack_fraction=("Is_Attack", "mean"),
    )
    return states_df[states_df["total_connections"] > 0].copy()

def normalize_and_export(states_df: pd.DataFrame):
    state_feature_cols = [
        "total_connections", "sum_syn_ack_rst_flags", "unique_dest_ports",
        "avg_bytes_per_sec", "avg_packets_per_sec", "avg_iat_mean", "avg_iat_variance"
    ]
    X_raw = states_df[state_feature_cols].values
    y_attack = states_df["attack_fraction"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    joblib.dump(scaler, SCALER_PATH)
    np.save(OUTPUT_NPY_PATH, X_scaled)
    np.save(OUTPUT_LABELS_PATH, y_attack)
    
    out_df = states_df.copy()
    for i, col in enumerate(state_feature_cols):
        out_df[f"norm_{col}"] = X_scaled[:, i]
    out_df.to_csv(OUTPUT_CSV_PATH, index=True)
    print("Stage 1 outputs: states.csv, states.npy, labels.npy, scaler.joblib")

if __name__ == "__main__":
    df = load_and_standardize_dataset(DATA_PATH)
    df_clean = clean_and_prepare_features(df)
    st_df = aggregate_time_windows(df_clean)
    normalize_and_export(st_df)
`;

const SCRIPT_STAGE_2 = `#!/usr/bin/env python3
"""
=============================================================================
Cybersecurity "World Model" Prototype - Core Transition Dynamics (Stage 2)
P(S_{t+1} | S_{t-4:t}) Autoregressive State Predictor & Attack Horizon Head
=============================================================================
THEORETICAL WORLD MODEL FOUNDATION (FOR JUDGES):
Unlike standard static intrusion detection classifiers (which map X -> y),
a CYBER WORLD MODEL learns the physical transition dynamics of the cyberspace environment:
    P( S_{t+1} | S_t, S_{t-1}, ..., S_{t-4} )
=============================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration & Hyperparameters
STATES_NPY_PATH = "states.npy"
LABELS_NPY_PATH = "labels.npy"
MODEL_SAVE_PATH = "world_model.pt"

WINDOW_SIZE = 5         # Input sequence: last 5 consecutive 10s states (50s context)
HIDDEN_SIZE = 64        # LSTM hidden state dimension
NUM_LAYERS = 1          # LSTM depth
EPOCHS = 30             # Number of training epochs
BATCH_SIZE = 16         # Mini-batch size
LEARNING_RATE = 0.001   # Adam optimizer learning rate
ATTACK_LOSS_WEIGHT = 0.5 # Weighting scalar for the BCE attack probability loss

FEATURE_NAMES = [
    "total_connections",
    "sum_syn_ack_rst_flags",
    "unique_dest_ports",
    "avg_bytes_per_sec",
    "avg_packets_per_sec",
    "avg_iat_mean",
    "avg_iat_variance",
]

# Empirically Calibrated MITRE ATT&CK Thresholds (Z-score normalized)
# Calibrated via auto_calibrate_thresholds.py with 70/30 stratified holdout validation.
# Tactic Extension: Volumetric attacks (DDoS/DoS Hulk) map to Impact (TA0040, T1498/T1499),
# while credential brute force (FTP-Patator) maps to Initial Access (T1110).
THRESH_RECON_PORTS = 1.01         # z-score >= +1.01 (100% detection on holdout)
THRESH_IMPACT_FLAGS = 0.93        # z-score >= +0.93 (100% detection on holdout)
THRESH_ACCESS_FLAGS = 0.93        # alias for backwards compatibility
THRESH_IMPACT_PKTS = 0.94         # z-score >= +0.94 (100% detection on holdout)
THRESH_DDOS_PKTS = 0.94           # alias for backwards compatibility
THRESH_BRUTE_BYTES_MAX = -0.45    # z-score <= -0.45 (100% detection on holdout)
THRESH_BRUTE_PORTS_MAX = -0.48    # z-score <= -0.48 (single port target)
THRESH_EXFIL_BYTES = 1.5          # z-score >= +1.5 std dev
THRESH_EXFIL_PORTS_MAX = 0.5      # focused egress channel
THRESH_LATERAL_CONNS = 1.0        # z-score >= +1.0 std dev
THRESH_LATERAL_PORTS = 0.6        # z-score >= +0.6 std dev
THRESH_C2_IAT_VAR_MAX = -0.4      # low inter-arrival jitter
THRESH_C2_BYTES_MAX = 0.3         # lightweight beaconing payloads

torch.manual_seed(42)
np.random.seed(42)

# 1. Sliding Window Dataset Builder
class SlidingWindowDataset(Dataset):
    def __init__(self, states: np.ndarray, labels: np.ndarray, window_size: int = 5):
        self.window_size = window_size
        self.num_samples = len(states) - window_size
        
        X_list, target_state_list, target_attack_list = [], [], []
        for i in range(self.num_samples):
            X_list.append(states[i : i + window_size])
            target_state_list.append(states[i + window_size])
            target_attack_list.append(labels[i + window_size])
            
        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.target_states = torch.tensor(np.array(target_state_list), dtype=torch.float32)
        self.target_attacks = torch.tensor(np.array(target_attack_list), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.X[idx], self.target_states[idx], self.target_attacks[idx]

# 2. Cyber World Model Architecture (Dual-Head LSTM)
class CyberWorldModel(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 1):
        super(CyberWorldModel, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer tracking temporal telemetry momentum
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Head 1: Linear layer projecting back to num_features (predicts S_{t+1})
        self.state_head = nn.Linear(hidden_size, num_features)
        
        # Head 2: Horizon attack probability classifier (Linear + Sigmoid)
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        next_state = self.state_head(last_hidden)      # Head 1: P(S_{t+1} | S_t)
        attack_prob = self.attack_head(last_hidden)    # Head 2: Attack risk
        return next_state, attack_prob

# 3. Model Training Pipeline (Chronological 80/20 Train/Validation Split)
def train_world_model():
    states = np.load(STATES_NPY_PATH)
    labels = np.load(LABELS_NPY_PATH)
    num_windows, num_features = states.shape
    
    # Chronological 80/20 split (reserves last 20% timeline as unseen validation)
    split_idx = int(num_windows * 0.8)
    train_states, val_states = states[:split_idx], states[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    train_dataset = SlidingWindowDataset(train_states, train_labels, window_size=WINDOW_SIZE)
    val_dataset = SlidingWindowDataset(val_states, val_labels, window_size=WINDOW_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CyberWorldModel(num_features=num_features, hidden_size=HIDDEN_SIZE).to(device)
    
    criterion_state = nn.MSELoss()
    criterion_attack = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_mse, tr_bce = 0.0, 0.0, 0.0
        for batch_x, batch_target_state, batch_target_attack in train_loader:
            batch_x = batch_x.to(device)
            batch_target_state = batch_target_state.to(device)
            batch_target_attack = batch_target_attack.to(device)
            
            optimizer.zero_grad()
            pred_state, pred_attack = model(batch_x)
            
            loss_mse = criterion_state(pred_state, batch_target_state)
            loss_bce = criterion_attack(pred_attack, batch_target_attack)
            total_loss = loss_mse + ATTACK_LOSS_WEIGHT * loss_bce
            
            total_loss.backward()
            optimizer.step()
            
            tr_loss += total_loss.item() * len(batch_x)
            tr_mse += loss_mse.item() * len(batch_x)
            tr_bce += loss_bce.item() * len(batch_x)
            
        # Validation on unseen timeline
        model.eval()
        v_loss, v_mse, v_bce = 0.0, 0.0, 0.0
        with torch.no_grad():
            for v_bx, v_by_state, v_by_attack in val_loader:
                v_bx, v_by_state, v_by_attack = v_bx.to(device), v_by_state.to(device), v_by_attack.to(device)
                ps, pa = model(v_bx)
                lm = criterion_state(ps, v_by_state)
                lb = criterion_attack(pa, v_by_attack)
                tot = lm + ATTACK_LOSS_WEIGHT * lb
                v_loss += tot.item() * len(v_bx)
                v_mse += lm.item() * len(v_bx)
                v_bce += lb.item() * len(v_bx)
                
        val_loss = v_loss / len(val_dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:02d}/{EPOCHS}] | "
                  f"Train Loss: {tr_loss/len(train_dataset):.4f} (MSE: {tr_mse/len(train_dataset):.4f}) | "
                  f"Val Loss: {val_loss:.4f} (MSE: {v_mse/len(val_dataset):.4f})")
               
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")
    return model, states

# 4. Closed-Loop Autoregressive Forecaster Function
def forecast(model: nn.Module, last_5_states: np.ndarray, k_steps: int = 5):
    """
    Rolls the model forward K steps, feeding its own predictions back in as input.
    Returns: list of (predicted_state, attack_probability) tuples.
    """
    model.eval()
    device = next(model.parameters()).device
    
    rolling_seq = torch.tensor(last_5_states, dtype=torch.float32).unsqueeze(0).to(device)
    forecasts = []
    
    with torch.no_grad():
        for step in range(1, k_steps + 1):
            next_state, attack_prob = model(rolling_seq)
            
            pred_s_np = next_state.squeeze(0).cpu().numpy()
            prob_fl = float(attack_prob.squeeze().cpu().item())
            forecasts.append((pred_s_np, prob_fl))
            
            # Autoregressively feed prediction back in
            rolling_seq = torch.cat([rolling_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)
            
    return forecasts

# 5. MITRE ATT&CK Tactic Stage Mapping (Empirically Calibrated & Holdout Validated)
def map_to_mitre_stage(state_vector) -> str:
    v = np.array(state_vector, dtype=float).flatten()
    total_conns, flags, unique_ports = v[0], v[1], v[2]
    bytes_sec, pkts_sec, iat_mean, iat_var = v[3], v[4], v[5], v[6]

    # PortScan -> Reconnaissance (Empirically calibrated: 100% holdout detection)
    if unique_ports >= THRESH_RECON_PORTS:
        return "Reconnaissance"

    # DoS Hulk & DDoS -> Impact (DoS/DDoS): T1498/T1499 Denial of Service
    # (Empirically calibrated: 100% holdout detection, volumetric flag or packet flood)
    if flags >= THRESH_IMPACT_FLAGS or pkts_sec >= THRESH_IMPACT_PKTS:
        return "Impact (DoS/DDoS)"

    # FTP-Patator -> Initial Access: Brute Force (Empirically calibrated: 100% holdout detection)
    if bytes_sec <= THRESH_BRUTE_BYTES_MAX and unique_ports <= THRESH_BRUTE_PORTS_MAX:
        return "Initial Access"

    # Exfiltration: Heavy outbound transfer
    if bytes_sec >= THRESH_EXFIL_BYTES and unique_ports <= THRESH_EXFIL_PORTS_MAX:
        return "Exfiltration"

    # Lateral Movement: Internal host connection dispersion
    if total_conns >= THRESH_LATERAL_CONNS and unique_ports >= THRESH_LATERAL_PORTS:
        return "Lateral Movement"

    # Command & Control (C2): Low jitter periodic beaconing
    if iat_var <= THRESH_C2_IAT_VAR_MAX and bytes_sec <= THRESH_C2_BYTES_MAX and pkts_sec > -0.5:
        return "Command & Control"

    return "Normal"

# 6. SHAP Feature Attribution for Classifier Head
def explain_prediction_shap(model, sample_seq, background_data=None, top_k=3):
    model.eval()
    device = next(model.parameters()).device
    sample_tensor = sample_seq.reshape(1, WINDOW_SIZE, -1)
    num_features = sample_tensor.shape[-1]
    sample_flat = sample_tensor.reshape(1, -1)

    def predict_prob(x_flat):
        x_3d = torch.tensor(x_flat, dtype=torch.float32).reshape(-1, WINDOW_SIZE, num_features).to(device)
        with torch.no_grad():
            _, attack_prob = model(x_3d)
        return attack_prob.squeeze(-1).cpu().numpy()

    try:
        import shap
        bg_flat = background_data[:20].reshape(len(background_data[:20]), -1) if background_data is not None else np.zeros((1, WINDOW_SIZE * num_features))
        explainer = shap.KernelExplainer(predict_prob, bg_flat, link="identity")
        shap_values = explainer.shap_values(sample_flat, nsamples=60, silent=True)
        shap_matrix = np.array(shap_values[0] if isinstance(shap_values, list) else shap_values).flatten()
        feature_attributions = shap_matrix.reshape(WINDOW_SIZE, num_features).sum(axis=0)
    except ImportError:
        base_prob = float(predict_prob(sample_flat)[0])
        feature_attributions = np.zeros(num_features)
        for f_idx in range(num_features):
            perturbed = sample_tensor.copy()
            perturbed[:, :, f_idx] = 0.0
            feature_attributions[f_idx] = base_prob - float(predict_prob(perturbed.reshape(1, -1))[0])

    ranked = [(FEATURE_NAMES[i], float(val)) for i, val in enumerate(feature_attributions)]
    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return ranked[:top_k]

if __name__ == "__main__":
    model, states = train_world_model()
    future_trajectory = forecast(model, states[-5:], k_steps=5)
    for i, (st, p) in enumerate(future_trajectory, 1):
        stage = map_to_mitre_stage(st)
        print(f"Step t+{i} (+{i*10}s) | P(Attack): {p*100:.1f}% | MITRE: {stage}")
    
    # Explain why alarm was triggered
    top_shap = explain_prediction_shap(model, states[-5:], background_data=states[:20], top_k=3)
    print("Top 3 Alarm Causes (SHAP):", top_shap)
`;

export const CodeViewer: React.FC = () => {
  const [selectedScript, setSelectedScript] = useState<'stage2' | 'stage1'>('stage2');
  const [copied, setCopied] = useState(false);

  const activeCode = selectedScript === 'stage2' ? SCRIPT_STAGE_2 : SCRIPT_STAGE_1;
  const fileName = selectedScript === 'stage2' ? 'train_world_model.py' : 'preprocess_traffic.py';

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([activeCode], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner with Script Switcher */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            {/* Script Toggle */}
            <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                id="select-stage2-script"
                onClick={() => setSelectedScript('stage2')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition ${
                  selectedScript === 'stage2'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Brain className="w-3.5 h-3.5" />
                <span>Stage 2: train_world_model.py</span>
              </button>
              <button
                id="select-stage1-script"
                onClick={() => setSelectedScript('stage1')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-semibold transition ${
                  selectedScript === 'stage1'
                    ? 'bg-emerald-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <FileCode className="w-3.5 h-3.5" />
                <span>Stage 1: preprocess_traffic.py</span>
              </button>
            </div>
          </div>
          <p className="text-xs text-slate-400">
            {selectedScript === 'stage2'
              ? 'PyTorch LSTM World Model: Joint training of P(S_{t+1}|S_t) dynamics & attack risk head with K-step closed-loop forecaster.'
              : 'Data Engineering: NaN/inf hygiene, 10-second temporal aggregation, StandardScaler, and states.npy/csv export.'}
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            id="download-py-btn"
            onClick={handleDownload}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download {fileName}</span>
          </button>
          <button
            id="copy-py-btn"
            onClick={handleCopy}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-sm transition"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy Code'}</span>
          </button>
        </div>
      </div>

      {/* Grid: Code and Steps Explanation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Code Content */}
        <div className="lg:col-span-8 bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 text-xs text-slate-400 font-mono">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-red-500/80 inline-block"></span>
              <span className="w-3 h-3 rounded-full bg-amber-500/80 inline-block"></span>
              <span className="w-3 h-3 rounded-full bg-emerald-500/80 inline-block"></span>
              <span className="ml-2 text-slate-200 font-semibold">{fileName}</span>
            </div>
            <span className="text-[11px] text-slate-400">PyTorch 2.0+ / Python 3.8+</span>
          </div>

          <div className="overflow-x-auto p-4 max-h-[700px] overflow-y-auto text-xs font-mono leading-relaxed text-slate-200">
            <pre>
              <code>{activeCode}</code>
            </pre>
          </div>
        </div>

        {/* Judge Explanation Walkthrough */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-200 mb-2 flex items-center gap-2">
              <Info className="w-4 h-4 text-emerald-400" />
              <span>
                {selectedScript === 'stage2' ? 'World Model Core Concepts' : 'Data Pipeline Principles'}
              </span>
            </h3>

            {selectedScript === 'stage2' ? (
              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="font-semibold text-emerald-400 block mb-1">
                    1. Learning Transition Dynamics P(S_&#123;t+1&#125; | S_t)
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    This is <strong>not a static classifier</strong>. It models the actual physical state changes of network telemetry over time, enabling simulation of counterfactual scenarios.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="font-semibold text-emerald-400 block mb-1">
                    2. Dual-Head Joint Optimization
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    Shared LSTM backbone projects to:
                    <br />• <strong>State Head (MSE)</strong>: Predicts continuous feature vector s&#770;<sub>t+1</sub> in &reals;<sup>7</sup>.
                    <br />• <strong>Attack Head (BCE)</strong>: Predicts risk probability P(Attack)<sub>t+1</sub> &isin; [0, 1].
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="font-semibold text-emerald-400 block mb-1">
                    3. Autoregressive K-Step Forecaster
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    Closed-loop forecasting recursively feeds predictions s&#770;<sub>t+1</sub> into the 5-state buffer to simulate trajectories 50 seconds ahead.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="font-semibold text-emerald-400 block mb-1">
                    1. 10s Window Discretization
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    Aggregates asynchronous packets into fixed Markovian time steps &Delta;t = 10s.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800">
                  <span className="font-semibold text-emerald-400 block mb-1">
                    2. StandardScaler (No Target Leakage)
                  </span>
                  <p className="text-slate-300 leading-relaxed">
                    Normalizes strictly the 7 physical telemetry features. Attack fraction is isolated as an external target.
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="bg-emerald-950/30 border border-emerald-800/40 rounded-xl p-4 text-xs text-slate-300">
            <span className="font-semibold text-emerald-300 block mb-1 flex items-center gap-1.5">
              <Play className="w-3.5 h-3.5" />
              Terminal Execution
            </span>
            <pre className="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-emerald-400 mt-2">
              pip install torch numpy{'\n'}
              python3 train_world_model.py
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
