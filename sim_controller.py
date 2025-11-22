#!/usr/bin/env python3

import threading
import time
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import torch

from data_loader import load_unsw_kaggle_dataset
from simulator import NIDSSimulator
from quarantine import QuarantineManager


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

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        self._packets: List[Dict] = []
        self._alerts: List[Dict] = []
        self._total_packets = 0
        self._threats_detected = 0
        self._processing_times: List[float] = []

        self._data_loaded = False
        self._idx = 0

        self.X_num = None
        self.X_cat = None
        self.y_true = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nids: Optional[NIDSSimulator] = None

        # Safe, demo-mode quarantine manager
        self.quarantine = QuarantineManager(enable_real_blocking=False)

        self._load_data_and_model()

    def _load_data_and_model(self) -> None:
        print("[SimController] Loading data from:", self.data_dir)
        try:
            (
                X_train_full,
                X_test_full,
                X_train_num,
                X_test_num,
                X_train_cat,
                X_test_cat,
                y_train,
                y_test,
                num_cols,
                cat_cols,
                scaler,
            ) = load_unsw_kaggle_dataset(self.data_dir)
        except Exception as e:
            print("[SimController] Failed to load dataset:", e)
            return

        self.X_num = X_test_num
        self.X_cat = X_test_cat
        self.y_true = y_test

        categories = []
        if X_train_cat is not None and X_train_cat.size > 0:
            for j in range(X_train_cat.shape[1]):
                max_val = int(X_train_cat[:, j].max())
                categories.append(max_val + 1)

        num_cont = X_train_num.shape[1] if X_train_num is not None else 0

        print("[SimController] Initializing NIDSSimulator")
        self.nids = NIDSSimulator(
            model_path=self.model_path,
            device=self.device,
            batch_size=512,
            threshold=self.threshold,
        )
        self.nids.load_model(categories=categories, num_cont=num_cont)

        self._data_loaded = True
        print("[SimController] Initialization complete.")

    def _generate_ip_for_index(self, idx: int) -> str:
        a = 10
        b = 0
        c = (idx // 256) % 256
        d = (idx % 256) or 1
        return f"{a}.{b}.{c}.{d}"

    def _run_loop(self) -> None:
        if not self._data_loaded or self.nids is None or self.X_num is None:
            print("[SimController] No data loaded; cannot run.")
            return

        print("[SimController] Simulation started.")
        n_samples = self.X_num.shape[0]
        target_interval = 1.0 / float(self.packet_rate) if self.packet_rate > 0 else 0.1

        while not self._stop_event.is_set():
            start_loop = time.time()

            idx = self._idx % n_samples
            self._idx += 1

            x_num = self.X_num[idx]
            if self.X_cat is not None and self.X_cat.size > 0:
                x_cat = self.X_cat[idx]
            else:
                x_cat = np.empty((0,), dtype=np.int64)

            ip = self._generate_ip_for_index(idx)
            timestamp = datetime.utcnow().isoformat()

            t0 = time.time()
            prob, pred = self.nids.predict_single(x_num, x_cat)
            processing_ms = (time.time() - t0) * 1000.0
            is_attack = bool(pred == 1)

            with self._lock:
                self._total_packets += 1
                self._processing_times.append(processing_ms)
                if len(self._processing_times) > 1000:
                    self._processing_times = self._processing_times[-1000:]

                packet = {
                    "id": int(idx),
                    "time": timestamp,
                    "ip": ip,
                    "prob": float(prob),
                    "pred": int(pred),
                    "is_attack": is_attack,
                    "compute_time": float(processing_ms),
                }
                self._packets.append(packet)
                if len(self._packets) > self.max_packets:
                    self._packets = self._packets[-self.max_packets:]

            if is_attack:
                self.quarantine.block_ip(ip, reason="NIDS classified packet as attack")
                with self._lock:
                    self._threats_detected += 1
                    alert = {
                        "id": int(idx),
                        "time": timestamp,
                        "ip": ip,
                        "prob": float(prob),
                        "action": f"QUARANTINED (logged {ip})",
                    }
                    self._alerts.append(alert)
                    if len(self._alerts) > self.max_alerts:
                        self._alerts = self._alerts[-self.max_alerts:]

            elapsed = time.time() - start_loop
            sleep_for = target_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        print("[SimController] Simulation loop finished.")

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._data_loaded or self.nids is None:
                print("[SimController] Cannot start; data/model not loaded.")
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._running = True
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            print("[SimController] Simulation stop requested.")
            self._stop_event.set()
            thread = self._thread
            self._thread = None
            self._running = False

        if thread is not None:
            thread.join()

    def get_status(self) -> Dict:
        with self._lock:
            if self._processing_times:
                avg_ms = float(sum(self._processing_times) / len(self._processing_times))
            else:
                avg_ms = 0.0
            status_str = "running" if self._running else "stopped"
            return {
                "status": status_str,
                "total_packets": self._total_packets,
                "threats_detected": self._threats_detected,
                "avg_processing_ms": avg_ms,
            }

    def get_packets(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            return list(reversed(self._packets[-limit:]))

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            return list(reversed(self._alerts[-limit:]))
