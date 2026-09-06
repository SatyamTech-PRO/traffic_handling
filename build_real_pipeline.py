#!/usr/bin/env python3
"""
build_real_pipeline.py
Rebuilds the entire flow-level and packet-level pipeline directly from real CIC-IDS2017 files:
Files: 4, 5, 6 (FTP-Patator / Benign - July 4, 2017)
Files: 10, 11 (DoS Hulk / DoS Slowhttptest / slowloris / Benign - July 5, 2017)
Files: 17, 18 (PortScan / DDoS / Benign - July 7, 2017)

Preserves true session boundaries with a 'source_file' column.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import fsspec
from sklearn.preprocessing import StandardScaler
import joblib

HF_BASE = "hf://datasets/rdpahalavan/CIC-IDS2017"
FLOW_PARQUET = "data/Network-Flows/CICIDS_Flow.parquet"

DATE_FILTER = {
    4: "4/7/2017",
    5: "4/7/2017",
    6: "4/7/2017",
    10: "5/7/2017",
    11: "5/7/2017",
    17: "7/7/2017",
    18: "7/7/2017"
}

TARGET_FILES = [4, 5, 6, 10, 11, 17, 18]

def get_dominant_attack(series):
    attacks = series[series.str.upper() != "BENIGN"]
    if len(attacks) > 0:
        return attacks.value_counts().index[0]
    return "BENIGN"

def build_pipeline():
    print("=" * 80)
    print("REBUILDING CYBER WORLD MODEL PIPELINE DIRECTLY FROM REAL SOURCE FILES")
    print("=" * 80)
    
    fs = fsspec.filesystem("hf")
    
    print(f"\n[1/4] Loading master flow dataset from {FLOW_PARQUET}...")
    t0 = time.time()
    flow_cols = [
        "flow_id", "Timestamp", "destination_port", "Flow Duration",
        "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count",
        "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std", "attack_label"
    ]
    print("Reading flow columns:", flow_cols)
    flows_master = pd.read_parquet(FLOW_PARQUET, columns=flow_cols)
    print(f"Loaded {len(flows_master):,} master flows in {time.time()-t0:.1f}s")
    
    # Index by flow_id for quick filtering
    flows_by_id = flows_master.set_index("flow_id")
    
    all_file_windows = []
    
    for file_id in TARGET_FILES:
        session_name = f"File{file_id}_session"
        day_str = DATE_FILTER[file_id]
        print(f"\n--- Processing {session_name} (Filter: {day_str}) ---")
        t_file = time.time()
        
        # 1. Read packet parquet file metadata and relevant packet columns
        pkt_uri = f"{HF_BASE}/Packet-Fields/Packet_Fields_File_{file_id}.parquet"
        pf_pkt = pq.ParquetFile(fs.open(pkt_uri))
        total_pkts = pf_pkt.metadata.num_rows
        print(f"  Remote parquet: {total_pkts:,} total packets")
        
        # Read up to 250k packets to cover rich telemetry while remaining fast
        sample_size = min(250000, total_pkts)
        pkt_cols = [
            "flow_id", "protocol", "destination_port", "IP len", "IP ihl",
            "IP ttl", "IP flags", "IP frag", "TCP dataofs", "TCP seq", "TCP window"
        ]
        t_read = time.time()
        pkt_table = pf_pkt.read(columns=pkt_cols).slice(0, sample_size)
        df_pkt = pkt_table.to_pandas()
        print(f"  Loaded {len(df_pkt):,} sample packets in {time.time()-t_read:.1f}s")
        
        # 2. Extract unique flow IDs and match with master flows
        unique_flows_in_pkt = df_pkt["flow_id"].unique()
        matched_flows = flows_by_id.loc[flows_by_id.index.intersection(unique_flows_in_pkt)].copy().reset_index()
        
        # Filter to the session day to exclude cross-day stale sockets
        matched_flows = matched_flows[matched_flows["Timestamp"].str.contains(day_str, na=False)].copy()
        if len(matched_flows) == 0:
            print(f"  Warning: No flows matched for day {day_str} in File {file_id}, using all matched flows")
            matched_flows = flows_by_id.loc[flows_by_id.index.intersection(unique_flows_in_pkt)].copy().reset_index()
            
        matched_flows["dt"] = pd.to_datetime(matched_flows["Timestamp"], format="mixed", dayfirst=True)
        matched_flows.sort_values("dt", inplace=True)
        
        flow_attack_counts = matched_flows["attack_label"].value_counts().to_dict()
        print(f"  Matched {len(matched_flows):,} flows on {day_str}. Attacks: {flow_attack_counts}")
        
        # 3. Clean numeric columns for flows
        num_cols = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count", "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std"]
        for c in num_cols:
            matched_flows[c] = pd.to_numeric(matched_flows[c], errors="coerce").fillna(0)
            
        matched_flows["Fwd_IAT_Var"] = matched_flows["Fwd IAT Std"] ** 2
        matched_flows["Flags_SYN_ACK_RST"] = (
            matched_flows["SYN Flag Count"] + matched_flows["ACK Flag Count"] + matched_flows["RST Flag Count"]
        )
        matched_flows["Is_Attack"] = (matched_flows["attack_label"].str.upper() != "BENIGN").astype(int)
        
        # Compute flow-level 10s windows
        grouped_flows = matched_flows.groupby(pd.Grouper(key="dt", freq="10s"))
        flow_windows = grouped_flows.agg(
            total_connections=("Flow Duration", "count"),
            sum_syn_ack_rst_flags=("Flags_SYN_ACK_RST", "sum"),
            unique_dest_ports=("destination_port", "nunique"),
            avg_bytes_per_sec=("Flow Bytes/s", "mean"),
            avg_packets_per_sec=("Flow Packets/s", "mean"),
            avg_iat_mean=("Fwd IAT Mean", "mean"),
            avg_iat_variance=("Fwd_IAT_Var", "mean"),
            attack_fraction=("Is_Attack", "mean"),
            dominant_attack_type=("attack_label", get_dominant_attack),
        )
        flow_windows = flow_windows[flow_windows["total_connections"] > 0].copy()
        
        # 4. Compute packet-level indicators
        df_pkt_matched = df_pkt[df_pkt["flow_id"].isin(matched_flows["flow_id"])].copy()
        flow_time_map = matched_flows.set_index("flow_id")["dt"].to_dict()
        df_pkt_matched["dt"] = df_pkt_matched["flow_id"].map(flow_time_map)
        df_pkt_matched = df_pkt_matched.dropna(subset=["dt"]).sort_values("dt")
        
        ip_len = df_pkt_matched["IP len"].fillna(0)
        ip_ihl = df_pkt_matched["IP ihl"].fillna(5) * 4
        tcp_ofs = df_pkt_matched["TCP dataofs"].fillna(5) * 4
        df_pkt_matched["payload_bytes"] = (ip_len - ip_ihl - tcp_ofs).clip(lower=0)
        
        df_pkt_matched["is_fragment"] = (
            (df_pkt_matched["IP frag"].fillna(0) > 0) | 
            df_pkt_matched["IP flags"].astype(str).str.contains("MF", na=False)
        ).astype(int)
        
        df_pkt_matched["is_duplicate_seq"] = df_pkt_matched.duplicated(subset=["flow_id", "TCP seq"], keep="first").astype(int)
        
        # Group packets into 10s windows
        grouped_pkts = df_pkt_matched.groupby(pd.Grouper(key="dt", freq="10s"))
        
        pkt_records = []
        for win_dt, grp in grouped_pkts:
            if len(grp) == 0:
                continue
            ttl_var = grp["IP ttl"].var()
            if pd.isna(ttl_var):
                ttl_var = 0.0
                
            # Filter specifically to TCP packets for TCP window size
            tcp_grp = grp[grp["protocol"].astype(str).str.lower() == "tcp"]
            if len(tcp_grp) > 0 and tcp_grp["TCP window"].notna().any():
                tcp_win_avg = float(tcp_grp["TCP window"].mean())
            else:
                tcp_win_avg = 0.0
                
            frag_fraction = float(grp["is_fragment"].mean())
            payload_mean = float(grp["payload_bytes"].mean())
            payload_std = float(grp["payload_bytes"].std())
            if pd.isna(payload_std):
                payload_std = 0.0
                
            retrans_count = int(grp["is_duplicate_seq"].sum())
            
            dports = grp["destination_port"].dropna().values
            if len(dports) > 1:
                port_deltas = np.abs(np.diff(dports))
                portscan_delta_std = float(np.std(port_deltas))
            else:
                portscan_delta_std = 0.0
                
            pkt_records.append({
                "dt": win_dt,
                "pkt_ttl_variance": ttl_var,
                "pkt_tcp_window_avg": tcp_win_avg,
                "pkt_ip_frag_fraction": frag_fraction,
                "pkt_payload_mean": payload_mean,
                "pkt_payload_std": payload_std,
                "pkt_retrans_count": retrans_count,
                "pkt_portscan_delta_std": portscan_delta_std
            })
            
        pkt_df = pd.DataFrame(pkt_records).set_index("dt") if pkt_records else pd.DataFrame()
        
        # 5. Join flow windows and packet windows
        if not pkt_df.empty:
            merged_win = flow_windows.join(pkt_df, how="left")
        else:
            merged_win = flow_windows.copy()
            for col in ["pkt_ttl_variance", "pkt_tcp_window_avg", "pkt_ip_frag_fraction", "pkt_payload_mean", "pkt_payload_std", "pkt_retrans_count", "pkt_portscan_delta_std"]:
                merged_win[col] = 0.0
                
        # Fill missing packet metrics with 0.0 or forward fill
        for col in ["pkt_ttl_variance", "pkt_tcp_window_avg", "pkt_ip_frag_fraction", "pkt_payload_mean", "pkt_payload_std", "pkt_retrans_count", "pkt_portscan_delta_std"]:
            merged_win[col] = merged_win[col].fillna(0.0)
            
        merged_win["source_file"] = session_name
        merged_win["file_id"] = file_id
        merged_win.reset_index(inplace=True)
        merged_win.rename(columns={"dt": "Timestamp"}, inplace=True)
        
        print(f"  Produced {len(merged_win):,} 10s windows for {session_name} in {time.time()-t_file:.1f}s")
        print(f"  Window attack breakdown: {merged_win['dominant_attack_type'].value_counts().to_dict()}")
        all_file_windows.append(merged_win)
        
    print("\n" + "=" * 80)
    print("[Step 4] Concatenating all files' windows into master states.csv")
    print("=" * 80)
    combined_states = pd.concat(all_file_windows, ignore_index=True)
    
    # Feature columns (7 flow + 7 packet = 14 dynamic features)
    feature_cols = [
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
    
    print(f"Total concatenated windows: {len(combined_states):,}")
    print("\nWindow count broken down per source file:")
    print(combined_states["source_file"].value_counts())
    
    print("\nWindow count broken down per dominant attack category across all files:")
    print(combined_states["dominant_attack_type"].value_counts())
    
    print("\nCrosstab: source_file vs dominant_attack_type:")
    print(pd.crosstab(combined_states["source_file"], combined_states["dominant_attack_type"]))
    
    # Save states.csv
    combined_states.to_csv("states.csv", index=False)
    print("Saved states.csv")
    
    # Fit StandardScaler on combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_states[feature_cols].values)
    y_labels = combined_states["attack_fraction"].values
    
    np.save("states.npy", X_scaled)
    np.save("labels.npy", y_labels)
    joblib.dump(scaler, "scaler.joblib")
    print(f"Saved states.npy shape: {X_scaled.shape}")
    print(f"Saved labels.npy shape: {y_labels.shape}")
    print("Saved scaler.joblib")
    
    # Inspect feature means and stds
    print("\nScaler Feature Means and Stds:")
    for name, mean, scale in zip(feature_cols, scaler.mean_, scaler.scale_):
        print(f"  {name:25s}: mean={mean:12.4f}, std={scale:12.4f}")
        
    return combined_states

if __name__ == "__main__":
    build_pipeline()
