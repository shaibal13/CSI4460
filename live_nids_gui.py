#!/usr/bin/env python3

"""
Usage (example):
    python live_nids_gui.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --threshold 0.95"
"""

import os
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Import your existing GUI + simulator logic
from nids_gui import NIDSGuiApp, THEME_NAME
from simulator import NIDSSimulator


# -----------------------------
# Preprocessing to match UNSW
# -----------------------------

def build_unsw_preprocessing(data_dir: str):
    """
    Build scaler + label encoders and metadata from UNSW CSVs without touching
    your existing loader.
    """
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cat_cols = ["proto", "service", "state"]
    drop_cols = ["id", "label", "attack_cat"]
    num_cols = [c for c in train_df.columns if c not in drop_cols + cat_cols]

    # Label encoders
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_vals)
        label_encoders[col] = le

    # Scaler for numeric features (fit on train)
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].values.astype(np.float32))

    # Class balance for pos_weight
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
    Turn a single UNSW-shaped row into (x_num, x_cat) 1D numpy arrays.
    """
    # Numeric
    num_vals = row[num_cols].astype(float).values.reshape(1, -1)
    num_scaled = scaler.transform(num_vals).astype(np.float32)
    x_num = num_scaled[0]

    # Categorical
    cat_arr = np.zeros((1, len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        val = str(row[col])
        le = label_encoders[col]
        if val in le.classes_:
            cat_arr[0, i] = le.transform([val])[0]
        else:
            cat_arr[0, i] = 0  # unseen category -> first class
    x_cat = cat_arr[0]

    return x_num, x_cat


# -----------------------------
# Packet → UNSW-like row mapping
# -----------------------------

def make_empty_unsw_row(all_cols):
    """
    Create a dict with all UNSW columns initialized to safe defaults.
    We only fill the fields we can derive from a single packet.
    """
    row = {c: 0 for c in all_cols}
    row["attack_cat"] = "Normal"
    row["label"] = 0
    row["id"] = 0
    row["proto"] = "-"
    row["service"] = "-"
    row["state"] = "-"
    return row


def packet_to_unsw_row(pkt, all_cols):
    """
    Very simplified mapping: treat each packet as a tiny "flow".
    Enough to drive the model live for a demo.
    """
    row = make_empty_unsw_row(all_cols)

    # Duration: single packet ≈ 0
    row["dur"] = 0.0

    # Protocol
    if TCP in pkt:
        row["proto"] = "tcp"
    elif UDP in pkt:
        row["proto"] = "udp"
    else:
        row["proto"] = "other"

    # Service guess based on port
    sport = None
    dport = None
    if TCP in pkt:
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
    elif UDP in pkt:
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)

    if dport in [80, 8080, 443] or sport in [80, 8080, 443]:
        row["service"] = "http"
    else:
        row["service"] = "-"

    # State placeholder
    row["state"] = "CON"

    # Basic sizes
    length = len(pkt)
    row["spkts"] = 1
    row["dpkts"] = 0
    row["sbytes"] = length
    row["dbytes"] = 0
    row["rate"] = 0.0

    if IP in pkt:
        row["sttl"] = int(pkt[IP].ttl)
        row["dttl"] = 0

    return pd.Series(row)


# -----------------------------
# Live GUI subclass
# -----------------------------

class LiveNIDSGuiApp(NIDSGuiApp):
    """
    Same UI as NIDSGuiApp, but _load_data_and_model + run_simulation_loop
    are overridden to use real packets instead of UNSW test rows.
    """

    def _load_data_and_model(self):
        """Initialize model and preprocessing for live use."""
        self.device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") and os.name != "nt") else "cpu"
        self.lbl_status.config(text="Loading Model & Preprocessing...")
        self.root.update()

        try:
            prep = build_unsw_preprocessing(self.args.data_dir)
            self.num_cols = prep["num_cols"]
            self.cat_cols = prep["cat_cols"]
            self.scaler = prep["scaler"]
            self.label_encoders = prep["label_encoders"]
            self.all_cols = prep["all_cols"]
            pos_weight_ratio = prep["pos_weight_ratio"]
            categories = prep["categories"]
            num_cont = prep["num_cont"]

            self.simulator = NIDSSimulator(
                model_path=self.args.model_path,
                device=self.device,
                batch_size=1,
                threshold=self.args.threshold,
            )
            self.simulator.load_model(categories, num_cont, pos_weight_ratio)

            self.lbl_status.config(text="Live mode ready. Press START to sniff.")
        except Exception as e:
            self.lbl_status.config(text=f"Error loading model: {e}", bootstyle="danger")
            raise

    def run_simulation_loop(self):
        """
        Background thread: sniff live packets and push results into the GUI queue.
        """
        iface = getattr(self.args, "iface", None)

        def handle_packet(pkt):
            if not self.is_running:
                return  # ignore packets while paused

            try:
                row = packet_to_unsw_row(pkt, self.all_cols)
                x_num, x_cat = transform_unsw_row(
                    row, self.num_cols, self.cat_cols, self.scaler, self.label_encoders
                )

                start_compute = time.time()
                prob, pred = self.simulator.predict_single(x_num, x_cat)
                compute_time = time.time() - start_compute
                is_attack = (pred == 1)

                packet_data = {
                    "id": int(time.time() * 1000) % 1_000_000,
                    "prob": prob,
                    "pred": pred,
                    "is_attack": is_attack,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "compute_time": compute_time,
                }
                self.data_queue.put(packet_data)
            except Exception as e:
                print("[Live GUI] Error processing packet:", e)

        # sniff will block this thread but GUI stays responsive because Tk runs on main thread
        sniff(
            prn=handle_packet,
            store=0,
            iface=iface,
            stop_filter=lambda p: not self.is_running,
        )


def main():
    parser = argparse.ArgumentParser(description="Live NIDS GUI")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to ftt_best.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dir with UNSW_NB15_training-set.csv/testing-set.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Attack probability threshold")
    parser.add_argument("--iface", type=str, default=None,
                        help="Interface name for sniffing, e.g., 'Wi-Fi'")
    args = parser.parse_args()

    app_style = tb.Window(themename=THEME_NAME)
    app = LiveNIDSGuiApp(app_style, args)
    app_style.mainloop()


if __name__ == "__main__":
    main()
