#!/usr/bin/env python3
"""
Utility script to generate realistic synthetic CIC-IDS2017 / CIC-IDS2018 traffic.
Useful for quick testing of 'preprocess_traffic.py' and hackathon demo runs.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_dataset(output_path: str = "data/traffic.csv", n_flows: int = 5000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_time = datetime(2017, 7, 4, 9, 0, 0)
    
    records = []
    ports = [80, 443, 22, 53, 8080, 21, 3306, 25, 445]
    attack_types = ["BENIGN", "PortScan", "DoS Hulk", "DDoS", "FTP-Patator"]
    total_seconds = 2100  # 35 minutes (210 10s windows)
    
    for i in range(n_flows):
        delta_sec = (i / n_flows) * total_seconds + random.uniform(0, 4)
        flow_time = start_time + timedelta(seconds=delta_sec)
        
        # Episodic attack timeline (like real network traces):
        # 0-350s (0-5.8m): BENIGN baseline
        # 350-700s (5.8-11.6m): PortScan
        # 700-1050s (11.6-17.5m): BENIGN baseline
        # 1050-1400s (17.5-23.3m): DDoS & DoS Hulk
        # 1400-1750s (23.3-29.1m): FTP-Patator (brute force)
        # 1750-2100s (29.1-35m): BENIGN baseline
        if 350 <= delta_sec < 700:
            attack_label = "PortScan" if random.random() < 0.85 else "BENIGN"
        elif 1050 <= delta_sec < 1225:
            attack_label = "DDoS" if random.random() < 0.85 else "BENIGN"
        elif 1225 <= delta_sec < 1400:
            attack_label = "DoS Hulk" if random.random() < 0.85 else "BENIGN"
        elif 1400 <= delta_sec < 1750:
            attack_label = "FTP-Patator" if random.random() < 0.80 else "BENIGN"
        else:
            attack_label = "BENIGN"
        
        # Attack traffic has distinct characteristics
        if attack_label == "PortScan":
            dst_port = random.randint(1024, 65535)
            duration = random.randint(50, 2000)
            fwd_pkts = random.randint(1, 4)
            bwd_pkts = random.randint(0, 1)
            syn_flags = random.randint(1, 2)
            ack_flags = 0
            rst_flags = random.choice([0, 1])
            bytes_s = random.uniform(100, 1500)
            pkts_s = random.uniform(10, 50)
            iat_mean = random.uniform(10, 200)
            iat_std = random.uniform(5, 50)
        elif attack_label in ["DoS Hulk", "DDoS"]:
            dst_port = random.choice([80, 443, 8080])
            duration = random.randint(200000, 2000000)
            fwd_pkts = random.randint(50, 400)
            bwd_pkts = random.randint(20, 200)
            syn_flags = random.randint(5, 20)
            ack_flags = random.randint(20, 150)
            rst_flags = random.randint(0, 5)
            bytes_s = random.uniform(50000, 500000)
            pkts_s = random.uniform(500, 3000)
            iat_mean = random.uniform(50, 1000)
            iat_std = random.uniform(20, 300)
        elif attack_label == "FTP-Patator":
            # Brute force login on port 21: rapid authentication attempts, low flags, small payload
            dst_port = 21
            duration = random.randint(1000, 30000)
            fwd_pkts = random.randint(3, 8)
            bwd_pkts = random.randint(2, 6)
            syn_flags = 1
            ack_flags = random.randint(1, 3)
            rst_flags = random.choice([0, 1])
            bytes_s = random.uniform(200, 3000)
            pkts_s = random.uniform(5, 45)
            iat_mean = random.uniform(300, 3500)
            iat_std = random.uniform(50, 800)
        else:
            # Normal Benign Traffic
            dst_port = random.choice(ports)
            duration = random.randint(1000, 800000)
            fwd_pkts = random.randint(3, 40)
            bwd_pkts = random.randint(2, 35)
            syn_flags = random.choice([0, 1])
            ack_flags = random.randint(1, 10)
            rst_flags = 0
            bytes_s = random.uniform(500, 35000)
            pkts_s = random.uniform(5, 120)
            iat_mean = random.uniform(500, 15000)
            iat_std = random.uniform(100, 5000)

        # Inject realistic quirks: some UNB datasets have leading spaces in column headers
        records.append({
            "Timestamp": flow_time.strftime("%d/%m/%Y %H:%M:%S"),
            " Destination Port": dst_port,  # intentionally testing leading space tolerance
            "Flow Duration": duration,
            "Total Fwd Packets": fwd_pkts,
            "Total Backward Packets": bwd_pkts,
            "Flow Bytes/s": bytes_s,
            "Flow Packets/s": pkts_s,
            "SYN Flag Count": syn_flags,
            "ACK Flag Count": ack_flags,
            "RST Flag Count": rst_flags,
            "Fwd IAT Mean": iat_mean,
            "Fwd IAT Std": iat_std,
            "Label": attack_label
        })

    # Add a couple of zero-duration flows with inf / NaN to test the robust cleaner
    records.append({
        "Timestamp": (start_time + timedelta(seconds=12)).strftime("%d/%m/%Y %H:%M:%S"),
        " Destination Port": 80,
        "Flow Duration": 0,
        "Total Fwd Packets": 1,
        "Total Backward Packets": 0,
        "Flow Bytes/s": np.inf,
        "Flow Packets/s": np.inf,
        "SYN Flag Count": 1,
        "ACK Flag Count": 0,
        "RST Flag Count": 0,
        "Fwd IAT Mean": np.nan,
        "Fwd IAT Std": 0,
        "Label": "BENIGN"
    })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset with {len(df)} rows at '{output_path}'")

if __name__ == "__main__":
    generate_sample_dataset()
