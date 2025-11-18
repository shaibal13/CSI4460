#!/usr/bin/env python3

"""
Explanation: Console version of nids_gui that uses real network traffic
*MUST BE ON WINDOWS!
Example:
    python live_nids.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --threshold 0.95
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

from data_loader import load_unsw_kaggle_dataset
from simulator import NIDSSimulator


# -----------------------------
# 1) Build preprocessing that matches UNSW
# -----------------------------

def build_unsw_preprocessing(data_dir: str):
    """
    Build scaler + label encoders based on UNSW_NB15 CSVs.
    We do this here so we don't have to touch existing code.
    """
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training set at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing testing set at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Same choices as in data_loader.py
    cat_cols = ["proto", "service", "state"]
    drop_cols = ["id", "label", "attack_cat"]
    num_cols = [c for c in train_df.columns if c not in drop_cols + cat_cols]

    # Label encoders for categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_vals)
        label_encoders[col] = le

    # Scaler for numeric cols (fit on train)
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].values.astype(np.float32))

    # basic class balance for pos_weight
    y_train = train_df["label"].values.astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))

    # Category sizes for FT-Transformer embeddings
    categories = tuple(len(label_encoders[c].classes_) for c in cat_cols)
    num_cont = len(num_cols)

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "all_cols": train_df.columns.tolist(),
        "pos_weight_ratio": pos_weight_ratio,
        "categories": categories,
        "num_cont": num_cont,
    }


def transform_unsw_row(row: pd.Series,
                       num_cols,
                       cat_cols,
                       scaler: StandardScaler,
                       label_encoders):
    """
    Turn a single UNSW-shaped row into (x_num, x_cat) arrays suitable for NIDSSimulator.predict_single.
    """
    # Numeric: scale -> (1, num_cont) -> (num_cont,)
    num_vals = row[num_cols].astype(float).values.reshape(1, -1)
    num_scaled = scaler.transform(num_vals).astype(np.float32)
    x_num = num_scaled[0]

    # Categorical: label encode -> (1, n_cat) -> (n_cat,)
    cat_arr = np.zeros((1, len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        val = str(row[col])
        le = label_encoders[col]
        if val in le.classes_:
            cat_arr[0, i] = le.transform([val])[0]
        else:
            # unseen category → map to first class
            cat_arr[0, i] = 0
    x_cat = cat_arr[0]

    return x_num, x_cat


# -----------------------------
# 2) Map *one packet* to an UNSW-like feature row
# -----------------------------

def make_empty_unsw_row(all_cols):
    """
    Create a dict with all UNSW columns initialized to neutral values.
    We'll fill what we can from the packet and leave the rest 0.
    """
    row = {c: 0 for c in all_cols}

    # Required non-numeric fields & labels
    row["attack_cat"] = "Normal"
    row["label"] = 0
    row["id"] = 0
    row["proto"] = "-"
    row["service"] = "-"
    row["state"] = "-"

    return row


def packet_to_unsw_row(pkt, all_cols):
    """
    Extremely simplified mapping:
    TODO better mapping that doesn't fill in junk
    """
    row = make_empty_unsw_row(all_cols)

    # Duration: single packet ≈ 0
    row["dur"] = 0.0

    # Protocol: TCP/UDP/other
    if TCP in pkt:
        row["proto"] = "tcp"
    elif UDP in pkt:
        row["proto"] = "udp"
    else:
        row["proto"] = "other"

    # Service: rough guess from ports
    sport = int(pkt[TCP].sport) if TCP in pkt else (int(pkt[UDP].sport) if UDP in pkt else None)
    dport = int(pkt[TCP].dport) if TCP in pkt else (int(pkt[UDP].dport) if UDP in pkt else None)

    if dport in [80, 8080, 443] or sport in [80, 8080, 443]:
        row["service"] = "http"
    else:
        row["service"] = "-"

    # State: we don't track true connection state here; mark as "CON" placeholder
    row["state"] = "CON"

    # Simple packet/byte fields
    length = len(pkt)
    row["spkts"] = 1
    row["dpkts"] = 0
    row["sbytes"] = length
    row["dbytes"] = 0
    row["rate"] = 0.0

    # TTL if we have IP
    if IP in pkt:
        row["sttl"] = int(pkt[IP].ttl)
        row["dttl"] = 0

    # Everything else (flags, counts, complex stats) defaults to 0

    return pd.Series(row)


# -----------------------------
# 3) Main live loop
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to ftt_best.pt (same as GUI)")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with UNSW_NB15_training-set.csv & _testing-set.csv")
    parser.add_argument("--iface", default=None,
                        help="Interface name to sniff on (Scapy); omit for default")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold to call something an attack")
    args = parser.parse_args()

    # 1) Build preprocessing to match training
    prep = build_unsw_preprocessing(args.data_dir)
    num_cols = prep["num_cols"]
    cat_cols = prep["cat_cols"]
    scaler = prep["scaler"]
    label_encoders = prep["label_encoders"]
    all_cols = prep["all_cols"]
    pos_weight_ratio = prep["pos_weight_ratio"]
    categories = prep["categories"]
    num_cont = prep["num_cont"]

    # 2) Use existing NIDSSimulator + model weights
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    simulator = NIDSSimulator(
        model_path=args.model_path,
        device=device,
        batch_size=1,
        threshold=args.threshold
    )
    simulator.load_model(categories, num_cont, pos_weight_ratio)

    print(f"[Live NIDS] Device: {device}")
    print(f"[Live NIDS] Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")
    print(f"[Live NIDS] Categories: {categories}")
    print(f"[Live NIDS] Sniffing on interface: {args.iface or 'default'}")
    print(f"[Live NIDS] Using threshold: {args.threshold:.2f}")
    print("Press Ctrl+C to stop.\n")

    def handle_packet(pkt):
        try:
            row = packet_to_unsw_row(pkt, all_cols)
            x_num, x_cat = transform_unsw_row(row, num_cols, cat_cols, scaler, label_encoders)

            # same interface as GUI: simulator expects 1D numpy arrays
            prob, pred = simulator.predict_single(x_num, x_cat)
            is_attack = (pred == 1)

            ts = datetime.now().strftime("%H:%M:%S")
            label = "ATTACK" if is_attack else "NORMAL"
            print(f"[{ts}] prob={prob:.3f}  ->  {label}")
        except Exception as e:
            # Don't crash on weird packets
            print("[Live NIDS] Error processing packet:", e)

    # 3) Start sniffing
    sniff(prn=handle_packet, store=0, iface=args.iface)


if __name__ == "__main__":
    main()
