#!/usr/bin/env python3
"""
process_single_file.py - Process a single CIC-IDS2017 session file into 10-second window states.

Usage:
    python3 process_single_file.py <file_number>
    e.g. python3 process_single_file.py 4
"""

import sys
import os

import argparse
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

TARGET_DAYS = {
    4: "4/7/2017",
    5: "4/7/2017",
    6: "4/7/2017",
    10: "5/7/2017",
    11: "5/7/2017",
    17: "7/7/2017",
    18: "7/7/2017"
}

PKT_COLS = [
    "flow_id", "protocol", "destination_port", "IP len", "IP ihl",
    "IP ttl", "IP flags", "IP frag", "TCP dataofs", "TCP seq", "TCP window", "attack_label"
]

FLOW_COLS = [
    "flow_id", "Timestamp", "destination_port", "Flow Duration",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std", "attack_label"
]

def get_dominant_attack(series):
    non_benign = [x for x in series if str(x).upper() != "BENIGN"]
    if non_benign:
        return pd.Series(non_benign).mode().iloc[0]
    return "BENIGN"

def ensure_master_flows():
    local_flow = "data/Network-Flows/CICIDS_Flow.parquet"
    if not os.path.exists(local_flow):
        print("Downloading master flow file (CICIDS_Flow.parquet)...")
        os.makedirs("data/Network-Flows", exist_ok=True)
        fs = HfFileSystem()
        fs.get_file("datasets/rdpahalavan/CIC-IDS2017/Network-Flows/CICIDS_Flow.parquet", local_flow)
    return local_flow

def load_packet_fields_chunked(file_id, max_packets=250000, chunk_size=50000):
    local_path = f"data/Packet-Fields/Packet_Fields_File_{file_id}.parquet"
    if os.path.exists(local_path):
        print(f"Reading local packet fields from {local_path} in chunks of {chunk_size:,}...")
        pf = pq.ParquetFile(local_path)
    else:
        print(f"Streaming packet fields from HuggingFace for File {file_id} in chunks of {chunk_size:,}...")
        fs = HfFileSystem()
        remote_path = f"datasets/rdpahalavan/CIC-IDS2017/Packet-Fields/Packet_Fields_File_{file_id}.parquet"
        f = fs.open(remote_path, "rb")
        pf = pq.ParquetFile(f)

    total_rows = pf.metadata.num_rows
    print(f"File {file_id} remote parquet total rows: {total_rows:,} across {pf.num_row_groups} row groups")
    
    chunks = []
    loaded_count = 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=chunk_size, columns=PKT_COLS):
        chunk_df = batch.to_pandas()
        chunks.append(chunk_df)
        loaded_count += len(chunk_df)
        if max_packets and loaded_count >= max_packets:
            print(f"Reached chunked limit of {loaded_count:,} packets (capped to preserve memory).")
            break
            
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} packets in {time.time()-t0:.1f}s")
    return df

def process_file(file_id):
    print(f"\n=======================================================")
    print(f"Processing File {file_id} (Target day: {TARGET_DAYS.get(file_id, 'all')})")
    print(f"=======================================================")
    t_start = time.time()

    # 1. Load packet fields in memory-safe chunks
    df_pkt = load_packet_fields_chunked(file_id, max_packets=200000, chunk_size=50000)
    unique_pkt_flows = set(df_pkt["flow_id"].unique())
    print(f"Extracted {len(unique_pkt_flows):,} unique flows from File {file_id} packets.")

    # 2. Match flows from master flow index in streaming chunks
    flow_file = ensure_master_flows()
    day_str = TARGET_DAYS.get(file_id, "")
    day_pattern = rf"0?{day_str}" if day_str else ""
    print(f"Scanning master flows in chunks of 200k (filtering for matching flow IDs and day '{day_str}')...")
    t0 = time.time()
    
    pf_flow = pq.ParquetFile(flow_file)
    matched_flow_chunks = []
    total_scanned_flows = 0
    for batch in pf_flow.iter_batches(batch_size=200000, columns=FLOW_COLS):
        total_scanned_flows += len(batch)
        b_df = batch.to_pandas()
        matched = b_df[b_df["flow_id"].isin(unique_pkt_flows)]
        if len(matched) > 0:
            if day_str:
                day_mask = matched["Timestamp"].astype(str).str.contains(rf"0?{day_str}|{day_str}", regex=True, na=False)
                if day_mask.any():
                    matched = matched[day_mask]
            matched_flow_chunks.append(matched)

    if matched_flow_chunks:
        matched_flows = pd.concat(matched_flow_chunks, ignore_index=True)
        print(f"Scanned {total_scanned_flows:,} master flows, matched {len(matched_flows):,} flows in {time.time()-t0:.1f}s")
    else:
        print("Warning: no overlapping flows found; taking day flows sample...")
        matched_flows = pd.DataFrame(columns=FLOW_COLS)

    matched_flows["dt"] = pd.to_datetime(matched_flows["Timestamp"], format="mixed", dayfirst=True)
    matched_flows = matched_flows.sort_values("dt").reset_index(drop=True)

    attack_counts = matched_flows["attack_label"].value_counts().to_dict()
    print(f"Matched flows attack distribution: {attack_counts}")

    # 4. Clean numeric columns for flows
    num_cols = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count", "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std"]
    for c in num_cols:
        matched_flows[c] = pd.to_numeric(matched_flows[c], errors="coerce").fillna(0)

    matched_flows["Fwd_IAT_Var"] = matched_flows["Fwd IAT Std"] ** 2
    matched_flows["Flags_SYN_ACK_RST"] = (
        matched_flows["SYN Flag Count"] + matched_flows["ACK Flag Count"] + matched_flows["RST Flag Count"]
    )
    matched_flows["Is_Attack"] = (matched_flows["attack_label"].str.upper() != "BENIGN").astype(int)

    # 5. Compute flow-level 10s windows
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
    print(f"Generated {len(flow_windows):,} non-empty 10s flow windows")

    # 6. Process packet-level indicators
    matched_flow_ids = set(matched_flows["flow_id"].unique())
    df_pkt_matched = df_pkt[df_pkt["flow_id"].isin(matched_flow_ids)].copy()
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

    grouped_pkts = df_pkt_matched.groupby(pd.Grouper(key="dt", freq="10s"))
    pkt_records = []
    for win_dt, grp in grouped_pkts:
        if len(grp) == 0:
            continue
        ttl_var = float(grp["IP ttl"].var()) if not pd.isna(grp["IP ttl"].var()) else 0.0
        
        tcp_grp = grp[grp["protocol"].astype(str).str.lower().isin(["tcp", "6"])]
        if len(tcp_grp) > 0 and tcp_grp["TCP window"].notna().any():
            tcp_win_avg = float(tcp_grp["TCP window"].mean())
        else:
            tcp_win_avg = 0.0

        frag_fraction = float(grp["is_fragment"].mean())
        payload_mean = float(grp["payload_bytes"].mean())
        payload_std = float(grp["payload_bytes"].std()) if not pd.isna(grp["payload_bytes"].std()) else 0.0
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

    # 7. Merge flow windows and packet windows
    if not pkt_df.empty:
        merged_win = flow_windows.join(pkt_df, how="left")
    else:
        merged_win = flow_windows.copy()
        for col in [
            "pkt_ttl_variance", "pkt_tcp_window_avg", "pkt_ip_frag_fraction",
            "pkt_payload_mean", "pkt_payload_std", "pkt_retrans_count", "pkt_portscan_delta_std"
        ]:
            merged_win[col] = 0.0

    # Fill any packet columns with 0.0
    for col in [
        "pkt_ttl_variance", "pkt_tcp_window_avg", "pkt_ip_frag_fraction",
        "pkt_payload_mean", "pkt_payload_std", "pkt_retrans_count", "pkt_portscan_delta_std"
    ]:
        if col in merged_win.columns:
            merged_win[col] = merged_win[col].fillna(0.0)

    merged_win["source_file"] = f"File_{file_id}"
    merged_win.reset_index(inplace=True)
    merged_win.rename(columns={"dt": "Timestamp"}, inplace=True)

    # 8. Save output
    out_csv = f"states_file{file_id}.csv"
    merged_win.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(merged_win):,} window states to {out_csv} in {time.time()-t_start:.1f}s")
    print(f"       Attack distribution: {dict(merged_win['dominant_attack_type'].value_counts())}")
    print(f"       Columns ({len(merged_win.columns)}): {list(merged_win.columns)}")
    return out_csv, len(merged_win)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single CIC-IDS2017 file into window states.")
    parser.add_argument("file_id", type=int, help="File number (e.g. 4, 5, 6, 10, 11, 17, 18)")
    args = parser.parse_args()

    process_file(args.file_id)
