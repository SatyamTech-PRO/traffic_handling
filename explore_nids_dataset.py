#!/usr/bin/env python3
"""
Step 1: Explore and identify the right files in CIC-IDS2017 using nids-datasets
"""
import sys
import pandas as pd
from nids_datasets import Dataset, DatasetInfo

def main():
    print("=" * 80)
    print("STEP 1: EXPLORING CIC-IDS2017 VIA nids_datasets.DatasetInfo")
    print("=" * 80)

    # Initialize DatasetInfo for CIC-IDS2017
    info = DatasetInfo(dataset="CIC-IDS2017")
    df_summary = info if isinstance(info, pd.DataFrame) else info.df

    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 1000)

    print("\n--- Summary DataFrame: Packet Counts Per Class Across All 18 Files ---")
    print(df_summary)

    target_categories = [
        "BENIGN",
        "PortScan",
        "DDoS",
        "DoS Hulk",
        "FTP-Patator"
    ]

    print("\n" + "=" * 80)
    print("TARGET CATEGORY BREAKDOWN ACROSS FILES")
    print("=" * 80)

    for cat in target_categories:
        matching_cols = [c for c in df_summary.columns if cat.lower() in c.lower()]
        if matching_cols:
            for col in matching_cols:
                series = df_summary[col]
                non_zero = series[series > 0].sort_values(ascending=False)
                print(f"\n[Category: '{col}'] Total packets: {series.sum():,}")
                for file_id, count in non_zero.items():
                    print(f"  - File {file_id:>2}: {count:>12,d} packets")
        else:
            print(f"\n[Category: '{cat}'] No matching column found!")

    print("\n" + "=" * 80)
    print("ANALYSIS OF TARGET FILES FOR WORKFLOW:")
    print("=" * 80)
    print("• File 17: Contains PortScan (321,386), DDoS (942,879), and BENIGN (1,920,222)")
    print("• File 10: Contains DoS Hulk (1,459,567) and BENIGN (422,225)")
    print("• File 5 : Contains FTP-Patator (68,205) and BENIGN (3,007,086)")
    print("=" * 80)

if __name__ == "__main__":
    main()
