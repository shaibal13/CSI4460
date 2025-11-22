#!/usr/bin/env python3

import ipaddress
import platform
import subprocess
from datetime import datetime
from pathlib import Path


class QuarantineManager:
    def __init__(self, enable_real_blocking: bool = False, log_path: str = "quarantine.log"):
        self.enable_real_blocking = enable_real_blocking
        self.log_path = Path(log_path)
        self.blocked_ips = set()

    def _log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        line = f"{timestamp} {message}\n"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
        print(f"[Quarantine] {message}")

    def _valid_ip(self, ip: str) -> bool:
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def _block_linux(self, ip: str) -> None:
        cmd = ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
        self._log(f"(Linux) Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            self._log(f"Failed to apply iptables rule for {ip}: {e}")

    def _block_macos(self, ip: str) -> None:
        rule = f"block drop from {ip} to any"
        self._log(f"(macOS) pf rule example: {rule}")
        # For safety we DO NOT run pfctl here by default.
        # A real deployment would maintain a dedicated pf anchor/table
        # and use: sudo pfctl -t blocked_ips -T add {ip}

    def block_ip(self, ip: str, reason: str = "model-detected-attack") -> None:
        if not self._valid_ip(ip):
            self._log(f"Not blocking invalid IP: {ip} (reason: {reason})")
            return

        if ip in self.blocked_ips:
            return

        self.blocked_ips.add(ip)
        self._log(f"Quarantining IP {ip} (reason: {reason})")

        if not self.enable_real_blocking:
            self._log(f"Real OS-level blocking is DISABLED (demo mode).")
            return

        system = platform.system().lower()
        if system == "linux":
            self._block_linux(ip)
        elif system == "darwin":
            self._block_macos(ip)
        else:
            self._log(f"OS '{system}' not supported for automatic firewall rules.")
