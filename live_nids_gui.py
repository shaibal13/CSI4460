#!/usr/bin/env python3

"""
Usage:
    python live_nids_gui.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --threshold 0.95
"""

import os
import time
import argparse
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP

import ttkbootstrap as tb

from nids_gui import NIDSGuiApp, THEME_NAME
from simulator import NIDSSimulator
from quarantine import QuarantineManager


# -----------------------------
# 1) Flow Management Logic
# -----------------------------

class Flow:
    """Represents a connection state to track statistics over time."""
    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto

        self.start_time = time.time()
        self.last_seen = time.time()

        # Basic Counters
        self.spkts = 0
        self.dpkts = 0
        self.sbytes = 0
        self.dbytes = 0

        # Features for UNSW-NB15
        self.sttl = 0
        self.dttl = 0

        self.service = self._identify_service()

    def _identify_service(self):
        ports = [self.src_port, self.dst_port]
        if 80 in ports or 8080 in ports: return "http"
        if 443 in ports: return "http"
        if 21 in ports: return "ftp"
        if 22 in ports: return "ssh"
        if 53 in ports: return "dns"
        return "-"

    def update(self, pkt, is_source_to_dest):
        self.last_seen = time.time()
        length = len(pkt)

        ttl = 0
        if IP in pkt:
            ttl = pkt[IP].ttl

        if is_source_to_dest:
            self.spkts += 1
            self.sbytes += length
            if ttl > 0: self.sttl = ttl
        else:
            self.dpkts += 1
            self.dbytes += length
            if ttl > 0: self.dttl = ttl

    def to_feature_row(self, all_cols):
        current_dur = max(0.000001, self.last_seen - self.start_time)

        row = {c: 0 for c in all_cols}

        # 1. Identity & Categorical
        row["proto"] = self.proto
        row["service"] = self.service
        row["state"] = "CON"

        # 2. Basic Flow Stats
        row["dur"] = current_dur
        row["spkts"] = self.spkts
        row["dpkts"] = self.dpkts
        row["sbytes"] = self.sbytes
        row["dbytes"] = self.dbytes

        # Rate: Packets per second
        row["rate"] = (self.spkts + self.dpkts) / current_dur

        # Sload/Dload: Bits per second
        row["sload"] = (self.sbytes * 8) / current_dur
        row["dload"] = (self.dbytes * 8) / current_dur

        # Smean/Dmean: Mean packet size
        row["smean"] = self.sbytes / max(1, self.spkts)
        row["dmean"] = self.dbytes / max(1, self.dpkts)

        row["sttl"] = self.sttl
        row["dttl"] = self.dttl

        # Defaults
        row["attack_cat"] = "Normal"
        row["label"] = 0
        row["id"] = 0

        return pd.Series(row)

    def __str__(self):
        return f"{self.src_ip}:{self.src_port} > {self.dst_ip}:{self.dst_port}"


class FlowManager:
    """Manages active flows and handles bidirectional mapping."""

    def __init__(self, timeout=10.0):
        self.flows = {}
        self.timeout = timeout
        self.lock = threading.Lock()

    def process_packet(self, pkt):
        if not (IP in pkt): return None

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto_num = pkt[IP].proto
        proto_str = "tcp" if TCP in pkt else ("udp" if UDP in pkt else "other")

        sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
        dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

        forward_key = (src_ip, dst_ip, sport, dport, proto_num)
        reverse_key = (dst_ip, src_ip, dport, sport, proto_num)

        with self.lock:
            self._cleanup()
            flow = None
            is_src = True

            if forward_key in self.flows:
                flow = self.flows[forward_key]
            elif reverse_key in self.flows:
                flow = self.flows[reverse_key]
                is_src = False
            else:
                flow = Flow(src_ip, dst_ip, sport, dport, proto_str)
                self.flows[forward_key] = flow

            flow.update(pkt, is_src)
            return flow

    def _cleanup(self):
        now = time.time()
        keys_to_remove = [k for k, v in self.flows.items() if now - v.last_seen > self.timeout]
        for k in keys_to_remove:
            del self.flows[k]


# -----------------------------
# 2) Preprocessing
# -----------------------------

def build_unsw_preprocessing(data_dir: str):
    """
    Build scaler + label encoders based on UNSW_NB15 CSVs.
    Combines Train + Test to prevent size mismatch errors.
    """
    train_path = os.path.join(data_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cat_cols = ["proto", "service", "state"]
    drop_cols = ["id", "label", "attack_cat"]
    num_cols = [c for c in train_df.columns if c not in drop_cols + cat_cols]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fix: Fit on combined data
        all_vals = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(all_vals)
        label_encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].values.astype(np.float32))

    y_train = train_df["label"].values.astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))

    # Replicate data_loader.py logic for category sizes
    X_train_cat_temp = train_df[cat_cols].apply(lambda x: label_encoders[x.name].transform(x.astype(str)))
    X_test_cat_temp = test_df[cat_cols].apply(lambda x: label_encoders[x.name].transform(x.astype(str)))
    all_cat = np.vstack([X_train_cat_temp.values, X_test_cat_temp.values])
    categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(len(cat_cols)))

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "all_cols": train_df.columns.tolist(),
        "pos_weight_ratio": pos_weight_ratio,
        "categories": categories,
        "num_cont": len(num_cols),
    }


def transform_unsw_row(row: pd.Series, num_cols, cat_cols, scaler, label_encoders):
    num_vals = row[num_cols].astype(float).values.reshape(1, -1)
    num_vals = np.nan_to_num(num_vals, posinf=0, neginf=0)  # Safety
    num_scaled = scaler.transform(num_vals).astype(np.float32)
    x_num = num_scaled[0]

    cat_arr = np.zeros((1, len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        val = str(row[col])
        le = label_encoders[col]
        if val in le.classes_:
            cat_arr[0, i] = le.transform([val])[0]
        else:
            cat_arr[0, i] = 0
    x_cat = cat_arr[0]

    return x_num, x_cat


# -----------------------------
# 3) Live GUI Subclass
# -----------------------------

class LiveNIDSGuiApp(NIDSGuiApp):
    """
    Overrides NIDSGuiApp to use FlowManager and Live Sniffing.
    """

    def _load_data_and_model(self):
        """Initialize model and preprocessing for live use."""
        self.device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") and os.name != "nt") else "cpu"
        self.lbl_status.config(text="Loading Model & Preprocessing...")
        self.root.update()

        try:
            # Load Preprocessing
            prep = build_unsw_preprocessing(self.args.data_dir)
            self.num_cols = prep["num_cols"]
            self.cat_cols = prep["cat_cols"]
            self.scaler = prep["scaler"]
            self.label_encoders = prep["label_encoders"]
            self.all_cols = prep["all_cols"]

            # Load Model
            self.simulator = NIDSSimulator(
                model_path=self.args.model_path,
                device=self.device,
                batch_size=1,
                threshold=self.args.threshold,
            )
            self.simulator.load_model(prep["categories"], prep["num_cont"], prep["pos_weight_ratio"])

            # Initialize Flow Manager
            self.flow_manager = FlowManager(timeout=10.0)

            self.quarantine = QuarantineManager(enable_real_blocking=False, log_path="demo_quarantine.log")

            self.lbl_status.config(text="Live Flow Monitor Ready. Press START.")
        except Exception as e:
            self.lbl_status.config(text=f"Error loading: {e}", bootstyle="danger")
            print(e)  # Print to console for debug
            raise

    def run_simulation_loop(self):
        """
        Background thread: sniff live packets, aggregate flows, push to GUI.
        """
        iface = getattr(self.args, "iface", None)

        def handle_packet(pkt):
            if not self.is_running:
                return

            try:
                # 1. Aggregate into Flow
                flow = self.flow_manager.process_packet(pkt)
                if flow is None: return

                # 2. Prepare Features
                row = flow.to_feature_row(self.all_cols)
                x_num, x_cat = transform_unsw_row(
                    row, self.num_cols, self.cat_cols, self.scaler, self.label_encoders
                )

                # 3. Predict
                start_compute = time.time()
                prob, pred = self.simulator.predict_single(x_num, x_cat)
                compute_time = time.time() - start_compute
                is_attack = (pred == 1)

                if is_attack:
                    attacker_ip = flow.src_ip
                    self.quarantine.block_ip(attacker_ip, reason=f"Malicious Probability: {prob:.4f}")

                # 4. Push to GUI
                packet_data = {
                    "id": str(flow),
                    "prob": prob,
                    "pred": pred,
                    "is_attack": is_attack,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "compute_time": compute_time,
                }
                self.data_queue.put(packet_data)

            except Exception as e:
                print(f"[Live GUI] Error: {e}")

        # Start Sniffing
        sniff(
            prn=handle_packet,
            store=0,
            iface=iface,
            stop_filter=lambda p: not self.is_running,
        )

    def _setup_ui(self):
        """
        Override UI setup to make the 'ID' column wider for IP addresses.
        """
        super()._setup_ui()
        # Widen the ID column to fit "192.168.1.1:1234 > ..."
        self.tree_flow.column("id", width=250)
        self.tree_quarantine.column("id", width=250)
        self.tree_flow.heading("id", text="Flow (Src > Dst)")
        self.tree_quarantine.heading("id", text="Flow (Src > Dst)")


def main():
    parser = argparse.ArgumentParser(description="Live NIDS GUI with Flow Aggregation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to ftt_best.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dir with UNSW_NB15 CSVs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Attack probability threshold")
    parser.add_argument("--iface", type=str, default=None,
                        help="Interface name for sniffing")
    args = parser.parse_args()

    app_style = tb.Window(themename=THEME_NAME)
    app = LiveNIDSGuiApp(app_style, args)
    app_style.mainloop()


if __name__ == "__main__":
    main()
