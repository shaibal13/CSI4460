#!/usr/bin/env python3

import threading
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch

from data_loader import load_unsw_kaggle_dataset
from simulator import NIDSSimulator


class SimulationController:
    def __init__(
        self,
        model_path: str,
        data_dir: str,
        packet_rate: int = 10,
        threshold: float = 0.5,
        max_packets: int = 500,
        max_alerts: int = 200,
    ):
        self.model_path = model_path
        self.data_dir = data_dir
        self.packet_rate = packet_rate
        self.threshold = threshold
        self.max_packets = max_packets
        self.max_alerts = max_alerts

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self._packets: List[Dict] = []
        self._alerts: List[Dict] = []
        self._total_packets = 0
        self._threats_detected = 0
        self._total_compute_ms = 0.0

        self.X_num = None
        self.X_cat = None
        self.y_true = None

        self._load_data_and_model()


# Setup
    def _load_data_and_model(self):
        """
        Load the UNSW Kaggle dataset + initialize NIDSSimulator.
        Mirrors the logic used in nids_gui.py but without any Tkinter.
        """
        print("[SimController] Loading data from:", self.data_dir)

        (X_train_full, X_test_full, X_train_num, X_test_num,
         X_train_cat, X_test_cat, y_train, y_test,
         num_cols, cat_cols, scaler) = load_unsw_kaggle_dataset(self.data_dir)

        self.X_num = X_test_num
        self.X_cat = X_test_cat
        self.y_true = y_test

        if X_train_cat.shape[0] > 0:
            all_cat = np.vstack([X_train_cat, X_test_cat])
            categories = tuple(int(np.max(all_cat[:, i]) + 1) for i in range(X_train_cat.shape[1]))
        else:
            categories = tuple()

        num_cont = X_train_num.shape[1]

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        pos_weight_ratio = max(1.0, n_neg / max(1, n_pos))

        print("[SimController] Initializing NIDSSimulator")
        self.simulator = NIDSSimulator(
            model_path=self.model_path,
            device=self.device,
            batch_size=1,
            threshold=self.threshold,
        )
        self.simulator.load_model(categories, num_cont, pos_weight_ratio)

        print("[SimController] Initialization complete.")

# Control
    def start(self):
        """Start the background simulation loop."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[SimController] Simulation started.")

    def stop(self):
        """Stop the background simulation loop."""
        with self._lock:
            self._running = False
        print("[SimController] Simulation stop requested.")

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

# Simulation loop
    def _run_loop(self):
        """
        Background loop that simulates network packets, runs predictions,
        and stores results in memory.
        """
        if self.X_num is None or self.X_cat is None:
            print("[SimController] No data loaded; cannot run.")
            with self._lock:
                self._running = False
            return

        packet_delay = 1.0 / max(1, self.packet_rate)
        indices = np.random.permutation(len(self.X_num))

        for idx in indices:
            if not self._is_running():
                break

            start_compute = time.time()

            x_num = self.X_num[idx]
            x_cat = self.X_cat[idx]

            prob, pred = self.simulator.predict_single(x_num, x_cat)
            is_attack = (pred == 1)
            compute_time = time.time() - start_compute

            packet_time = datetime.now().strftime("%H:%M:%S")

            with self._lock:
                self._total_packets += 1
                self._total_compute_ms += compute_time * 1000.0
                if is_attack:
                    self._threats_detected += 1

                packet = {
                    "id": int(idx),
                    "time": packet_time,
                    "prob": float(prob),
                    "pred": int(pred),
                    "is_attack": bool(is_attack),
                    "compute_time": float(compute_time),
                }
                self._packets.append(packet)
                if len(self._packets) > self.max_packets:
                    self._packets = self._packets[-self.max_packets:]

                if is_attack:
                    alert = {
                        "id": int(idx),
                        "time": packet_time,
                        "prob": float(prob),
                        "action": "QUARANTINED",
                    }
                    self._alerts.append(alert)
                    if len(self._alerts) > self.max_alerts:
                        self._alerts = self._alerts[-self.max_alerts:]

            time.sleep(packet_delay)

        with self._lock:
            self._running = False
        print("[SimController] Simulation loop finished.")

# Info methods used by the Flask API
    def get_status(self) -> Dict:
        """Return stats for the top metrics cards."""
        with self._lock:
            if self._total_packets > 0:
                avg_ms = self._total_compute_ms / self._total_packets
            else:
                avg_ms = 0.0

            status_str = "monitoring active" if self._running else "monitoring stopped"

            return {
                "status": status_str,
                "total_packets": self._total_packets,
                "threats_detected": self._threats_detected,
                "avg_processing_ms": avg_ms,
            }

    def get_packets(self, limit: int = 100) -> List[Dict]:
        """Return the most recent packets, newest first."""
        with self._lock:
            return list(reversed(self._packets[-limit:]))

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Return the most recent alerts, newest first."""
        with self._lock:
            return list(reversed(self._alerts[-limit:]))
