#!/usr/bin/env python3
"""
process_file_v2.py - Single self-contained processor for CIC-IDS2017 Packet-Fields files.
Accepts file ID as command-line argument (e.g. python3 process_file_v2.py 5).
Uses selective-column streaming via HfFileSystem (NEVER downloads full packet parquets).
"""

import sys
import os
import gc
import time

for pth in ['/app/applet/.python_packages', '/usr/local/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages']:
    if os.path.exists(pth) and pth not in sys.path:
        sys.path.insert(0, pth)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

# Accept file ID from command-line argument (e.g. python3 process_file_v2.py 5)
if len(sys.argv) > 1:
    try:
        FILE_ID = int(sys.argv[1])
    except ValueError:
        print(f"Invalid file id '{sys.argv[1]}', defaulting to 4")
        FILE_ID = 4
else:
    FILE_ID = 4

# Map files to their primary day in CIC-IDS2017:
# Files 4, 5, 6 -> 4/7/2017 (Tuesday: Brute Force FTP/SSH)
# Files 10, 11 -> 5/7/2017 (Wednesday: DoS)
# Files 17, 18 -> 7/7/2017 (Friday: PortScan & DDoS)
FILE_DAY_MAP = {
    4: "4/7/2017",
    5: "4/7/2017",
    6: "4/7/2017",
    10: "5/7/2017",
    11: "5/7/2017",
    17: "7/7/2017",
    18: "7/7/2017"
}
DAY_STR = FILE_DAY_MAP.get(FILE_ID, "4/7/2017")
FLOW_PARQUET = "data/Network-Flows/CICIDS_Flow.parquet"
OUTPUT_CSV = f"states_file{FILE_ID}.csv"

PKT_COLS = [
    'flow_id', 'protocol', 'destination_port', 'IP len', 'IP ihl',
    'IP ttl', 'IP flags', 'IP frag', 'TCP dataofs', 'TCP seq',
    'TCP window', 'attack_label'
]

FLOW_COLS = [
    "flow_id", "Timestamp", "destination_port", "Flow Duration",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std", "attack_label"
]

def get_dominant_attack(series):
    attacks = series[series.astype(str).str.upper() != "BENIGN"]
    if len(attacks) > 0:
        return attacks.value_counts().index[0]
    return "BENIGN"

def main():
    t0_all = time.time()
    print("=" * 80)
    print(f"CIC-IDS2017 SINGLE-PASS PIPELINE: FILE {FILE_ID} (Target Day: {DAY_STR})")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 2. Master flow parquet (~353MB local, download fresh if missing)
    # -------------------------------------------------------------------------
    if not os.path.exists(FLOW_PARQUET):
        print(f"[1/6] Downloading master flow parquet fresh to {FLOW_PARQUET} (~353MB)...")
        os.makedirs(os.path.dirname(FLOW_PARQUET), exist_ok=True)
        from huggingface_hub import hf_hub_download
        import shutil
        cached_p = hf_hub_download(repo_id="rdpahalavan/CIC-IDS2017", filename="Network-Flows/CICIDS_Flow.parquet", repo_type="dataset")
        shutil.copy(cached_p, FLOW_PARQUET)
        print(f"      Downloaded {FLOW_PARQUET} ({os.path.getsize(FLOW_PARQUET)/1e6:.1f} MB)")
    else:
        print(f"[1/6] Found existing local master flow parquet: {FLOW_PARQUET} ({os.path.getsize(FLOW_PARQUET)/1e6:.1f} MB)")

    print("      Loading master flow columns...")
    t_flow = time.time()
    flows_master = pd.read_parquet(FLOW_PARQUET, columns=FLOW_COLS)
    print(f"      Loaded {len(flows_master):,} master flows in {time.time()-t_flow:.2f}s")

    # -------------------------------------------------------------------------
    # 3. Stream 12 needed columns from File {FILE_ID} Packet-Fields via HfFileSystem
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Streaming 12 selective columns from remote File {FILE_ID} via HfFileSystem...")
    t_pkt = time.time()
    fs = HfFileSystem()
    remote_path = f"datasets/rdpahalavan/CIC-IDS2017/Packet-Fields/Packet_Fields_File_{FILE_ID}.parquet"

    with fs.open(remote_path, 'rb') as f:
        pf = pq.ParquetFile(f)
        print(f"      Remote file metadata: {pf.metadata.num_rows:,} total rows across {pf.num_row_groups} row group(s)")
        table = pf.read(columns=PKT_COLS)
        df_pkt = table.to_pandas()
    print(f"      Streamed {len(df_pkt):,} packets in {time.time()-t_pkt:.2f}s!")

    # -------------------------------------------------------------------------
    # 4. Match packets to flows via flow_id & filter to target day
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Matching packets to flows and filtering to {DAY_STR}...")
    unique_flows_in_pkt = set(df_pkt["flow_id"].unique())
    print(f"      Unique flow_ids in packet sample: {len(unique_flows_in_pkt):,}")

    matched_flows = flows_master[flows_master["flow_id"].isin(unique_flows_in_pkt)].copy()
    del flows_master
    gc.collect()

    day_flows = matched_flows[matched_flows["Timestamp"].astype(str).str.contains(DAY_STR, na=False)].copy()
    if len(day_flows) > 0:
        matched_flows = day_flows
        print(f"      Filtered to day {DAY_STR}: {len(matched_flows):,} active flows")
    else:
        print(f"      Warning: No flows found specifically on {DAY_STR}, using all {len(matched_flows):,} matched flows")

    matched_flows["dt"] = pd.to_datetime(matched_flows["Timestamp"], format="mixed", dayfirst=True)
    matched_flows.sort_values("dt", inplace=True)
    matched_flows.reset_index(drop=True, inplace=True)

    attack_counts = matched_flows["attack_label"].value_counts().to_dict()
    print(f"      Attack distribution: {attack_counts}")

    # -------------------------------------------------------------------------
    # 5. Compute flow-level 7 features for 10-second windows
    # -------------------------------------------------------------------------
    print("\n[4/6] Aggregating flow-level features into 10s windows...")
    num_cols = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count", "ACK Flag Count", "RST Flag Count", "Fwd IAT Mean", "Fwd IAT Std"]
    for c in num_cols:
        matched_flows[c] = pd.to_numeric(matched_flows[c], errors="coerce").fillna(0)

    matched_flows["Fwd_IAT_Var"] = matched_flows["Fwd IAT Std"] ** 2
    matched_flows["Flags_SYN_ACK_RST"] = (
        matched_flows["SYN Flag Count"] + matched_flows["ACK Flag Count"] + matched_flows["RST Flag Count"]
    )
    matched_flows["Is_Attack"] = (matched_flows["attack_label"].astype(str).str.upper() != "BENIGN").astype(int)

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
    print(f"      Formed {len(flow_windows):,} active 10s flow windows")

    # -------------------------------------------------------------------------
    # 6. Compute 7 packet-level features per 10s window
    # -------------------------------------------------------------------------
    print("\n[5/6] Computing 7 packet-level features per 10s window...")
    matched_flow_ids = set(matched_flows["flow_id"].unique())
    df_pkt_matched = df_pkt[df_pkt["flow_id"].isin(matched_flow_ids)].copy()
    del df_pkt
    gc.collect()

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
        ttl_val = grp["IP ttl"].var()
        ttl_var = float(ttl_val) if pd.notna(ttl_val) else 0.0

        tcp_grp = grp[grp["protocol"].astype(str).str.lower().isin(["tcp", "6"])]
        if len(tcp_grp) > 0 and tcp_grp["TCP window"].notna().any():
            twin = tcp_grp["TCP window"].mean()
            tcp_win_avg = float(twin) if pd.notna(twin) else 0.0
        else:
            tcp_win_avg = 0.0

        frag_val = grp["is_fragment"].mean()
        frag_fraction = float(frag_val) if pd.notna(frag_val) else 0.0

        p_mean = grp["payload_bytes"].mean()
        payload_mean = float(p_mean) if pd.notna(p_mean) else 0.0

        p_std = grp["payload_bytes"].std()
        payload_std = float(p_std) if pd.notna(p_std) else 0.0

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
    del df_pkt_matched
    gc.collect()

    # -------------------------------------------------------------------------
    # 7. Merge flow and packet features & save immediately to states_file{N}.csv
    # -------------------------------------------------------------------------
    print("\n[6/6] Merging flow and packet window features...")
    pkt_feature_cols = [
        "pkt_ttl_variance", "pkt_tcp_window_avg", "pkt_ip_frag_fraction",
        "pkt_payload_mean", "pkt_payload_std", "pkt_retrans_count", "pkt_portscan_delta_std"
    ]

    if not pkt_df.empty:
        merged_win = flow_windows.join(pkt_df, how="left")
    else:
        merged_win = flow_windows.copy()
        for col in pkt_feature_cols:
            merged_win[col] = 0.0

    for col in pkt_feature_cols:
        merged_win[col] = merged_win[col].fillna(0.0)

    merged_win["source_file"] = f"File_{FILE_ID}"
    merged_win["file_id"] = FILE_ID
    merged_win.reset_index(inplace=True)
    merged_win.rename(columns={"dt": "Timestamp"}, inplace=True)

    # Save immediately
    merged_win.to_csv(OUTPUT_CSV, index=False)
    t_total = time.time() - t0_all

    print("=" * 80)
    print(f"SUCCESS: Saved {len(merged_win):,} rows to {OUTPUT_CSV}")
    print(f"Total time taken: {t_total:.2f} seconds ({t_total/60:.2f} minutes)")
    print(f"Final Output Row Count: {len(merged_win)}")
    print(f"Attack labels in output: {merged_win['dominant_attack_type'].value_counts().to_dict()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
