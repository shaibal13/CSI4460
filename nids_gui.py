#!/usr/bin/env python3

'''
    Example usage:
        python nids_gui.py --model_path results/ftt/ftt_best.pt --data_dir Datasets/UNSW --packet_rate 10
'''

import os
import time
import threading
import queue
import argparse
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Import your existing logic
from data_loader import load_unsw_kaggle_dataset
from simulator import NIDSSimulator

# --- Config ---
THEME_NAME = "darkly"
MAX_ROWS = 50


class NIDSGuiApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("NIDS | FT-Transformer Real-Time Monitor")
        self.root.geometry("1400x800")
        self.args = args

        # Threading and Control
        self.is_running = False
        self.simulation_thread = None
        self.data_queue = queue.Queue()

        # Stats
        self.total_packets = 0
        self.attack_count = 0

        self._setup_ui()
        self._load_data_and_model()

    def _setup_ui(self):
        """Constructs the 2-pane UI layout."""

        # 1. Header / Status Bar
        header_frame = tb.Frame(self.root, padding=10, bootstyle="secondary")
        header_frame.pack(fill=X, side=TOP)

        self.lbl_status = tb.Label(header_frame, text="System Ready", font=("Consolas", 12, "bold"),
                                   bootstyle="inverse-secondary")
        self.lbl_status.pack(side=LEFT, padx=10)

        self.btn_start = tb.Button(header_frame, text="START MONITORING", bootstyle="success",
                                   command=self.toggle_simulation)
        self.btn_start.pack(side=RIGHT, padx=10)

        # 2. Stats Dashboard
        stats_frame = tb.Frame(self.root, padding=10)
        stats_frame.pack(fill=X, side=TOP)

        self.meter_total = self._create_stat_card(stats_frame, "Total Packets", "0", "primary")
        self.meter_attacks = self._create_stat_card(stats_frame, "Threats Detected", "0", "danger")
        self.meter_rate = self._create_stat_card(stats_frame, "Processing Time", "0ms", "info")

        self.paned_window = tb.Panedwindow(self.root, orient=HORIZONTAL, bootstyle="light")
        self.paned_window.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # --- LEFT PANE: Network Flow ---
        left_frame = tb.Labelframe(self.paned_window, text="  Live Network Traffic Flow  ", padding=10,
                                   bootstyle="info")
        self.paned_window.add(left_frame, weight=2)

        # Network Flow
        cols_flow = ("id", "time", "prob", "status")
        self.tree_flow = tb.Treeview(left_frame, columns=cols_flow, show='headings', bootstyle="info", height=20)
        self.tree_flow.heading("id", text="Packet ID")
        self.tree_flow.heading("time", text="Timestamp")
        self.tree_flow.heading("prob", text="Conf. Score")
        self.tree_flow.heading("status", text="Classification")

        self.tree_flow.column("id", width=80)
        self.tree_flow.column("time", width=120)
        self.tree_flow.column("prob", width=100)
        self.tree_flow.column("status", width=150)
        self.tree_flow.pack(fill=BOTH, expand=True)

        # --- RIGHT PANE: Quarantine ---
        right_frame = tb.Labelframe(self.paned_window, text="  QUARANTINE / ALERTS  ", padding=10, bootstyle="danger")
        self.paned_window.add(right_frame, weight=1)

        # Quarantine Flow
        cols_alert = ("id", "time", "prob", "action")
        self.tree_quarantine = tb.Treeview(right_frame, columns=cols_alert, show='headings', bootstyle="danger",
                                           height=20)
        self.tree_quarantine.heading("id", text="ID")
        self.tree_quarantine.heading("time", text="Time")
        self.tree_quarantine.heading("prob", text="Confidence")
        self.tree_quarantine.heading("action", text="Action")

        self.tree_quarantine.column("id", width=70)
        self.tree_quarantine.column("time", width=100)
        self.tree_quarantine.column("prob", width=80)
        self.tree_quarantine.column("action", width=100)
        self.tree_quarantine.pack(fill=BOTH, expand=True)

    def _create_stat_card(self, parent, title, value, style):
        """Helper to create a visual stat block."""
        frame = tb.Frame(parent, padding=15, bootstyle="dark")
        frame.pack(side=LEFT, fill=X, expand=True, padx=5)

        lbl_title = tb.Label(frame, text=title, font=("Helvetica", 10), bootstyle="secondary")
        lbl_title.pack()
        lbl_val = tb.Label(frame, text=value, font=("Helvetica", 20, "bold"), bootstyle=style)
        lbl_val.pack()
        return lbl_val

    def _load_data_and_model(self):
        """Initialize the model and data"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lbl_status.config(text="Loading Data & Model...")
        self.root.update()

        try:
            # Load Data
            (X_train_full, X_test_full, X_train_num, X_test_num,
             X_train_cat, X_test_cat, y_train, y_test,
             num_cols, cat_cols, scaler) = load_unsw_kaggle_dataset(self.args.data_dir)

            self.X_num = X_test_num
            self.X_cat = X_test_cat
            self.y_true = y_test

            # Model Setup
            all_cat = np.vstack([X_train_cat, X_test_cat]) if X_test_cat.shape[0] > 0 else X_train_cat
            categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(X_train_cat.shape[1]))
            num_cont = X_train_num.shape[1]

            n_pos = int((y_train == 1).sum())
            n_neg = int((y_train == 0).sum())
            pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))

            self.simulator = NIDSSimulator(
                model_path=self.args.model_path,
                device=self.device,
                batch_size=1,
                threshold=self.args.threshold
            )
            self.simulator.load_model(categories, num_cont, pos_weight_ratio)

            self.lbl_status.config(text="System Initialized. Ready to Start.")

        except Exception as e:
            self.lbl_status.config(text=f"Error: {str(e)}", bootstyle="danger")
            print(e)

    def toggle_simulation(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="RESUME MONITORING", bootstyle="success")
            self.lbl_status.config(text="Paused")
        else:
            self.is_running = True
            self.btn_start.config(text="STOP MONITORING", bootstyle="danger")
            self.lbl_status.config(text="Monitoring Active", bootstyle="success")

            self.simulation_thread = threading.Thread(target=self.run_simulation_loop, daemon=True)
            self.simulation_thread.start()

            self.root.after(100, self.process_queue)

    def run_simulation_loop(self):
        """Background thread that runs prediction logic"""
        indices = np.random.permutation(len(self.X_num))

        packet_delay = 1.0 / self.args.packet_rate

        for idx in indices:
            if not self.is_running:
                break

            start_compute = time.time()

            x_num = self.X_num[idx]
            x_cat = self.X_cat[idx]

            # Predict
            prob, pred = self.simulator.predict_single(x_num, x_cat)

            is_attack = (pred == 1)
            true_label = self.y_true[idx] if self.y_true is not None else None

            compute_time = time.time() - start_compute

            # Send data to UI
            packet_data = {
                'id': idx,
                'prob': prob,
                'pred': pred,
                'is_attack': is_attack,
                'time': datetime.now().strftime("%H:%M:%S"),
                'compute_time': compute_time
            }
            self.data_queue.put(packet_data)

            # Simulated Network Rate
            elapsed = time.time() - start_compute
            sleep_time = max(0, packet_delay - elapsed)
            time.sleep(sleep_time)

    def process_queue(self):
        """Main thread to update UI from the queue"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get_nowait()
                self._update_display(data)
        finally:
            if self.is_running:
                self.root.after(50, self.process_queue)

    def _update_display(self, data):
        self.total_packets += 1

        # 1. Update Network Flow
        status_text = "NORMAL"
        row_tag = "normal"

        if data['is_attack']:
            self.attack_count += 1
            status_text = "ATTACK DETECTED"
            row_tag = "attack"

            # 2. Update Quarantine
            self.tree_quarantine.insert("", 0, values=(
                data['id'],
                data['time'],
                f"{data['prob']:.4f}",
                "QUARANTINED"
            ))
            if len(self.tree_quarantine.get_children()) > MAX_ROWS:
                self.tree_quarantine.delete(self.tree_quarantine.get_children()[-1])

        # Insert into Flow
        self.tree_flow.insert("", 0, values=(
            data['id'],
            data['time'],
            f"{data['prob']:.4f}",
            status_text
        ), tags=(row_tag,))

        self.tree_flow.tag_configure("normal", foreground="#00ff00")  # Hacker Green
        self.tree_flow.tag_configure("attack", foreground="#ff4444")  # Alert Red

        if len(self.tree_flow.get_children()) > MAX_ROWS:
            self.tree_flow.delete(self.tree_flow.get_children()[-1])

        self.meter_total.config(text=str(self.total_packets))
        self.meter_attacks.config(text=str(self.attack_count))
        self.meter_rate.config(text=f"{data['compute_time'] * 1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="NIDS GUI Simulator")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--packet_rate", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    app_style = tb.Window(themename=THEME_NAME)

    app = NIDSGuiApp(app_style, args)

    app_style.mainloop()


if __name__ == "__main__":
    main()
    