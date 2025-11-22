#!/usr/bin/env python3

import os
from flask import Flask, jsonify, render_template
from sim_controller_ensa import SimulationController

#CONFIG
MODEL_PATH = os.path.join("models", "ftt_best.pt")
DATA_DIR = os.path.join("Datasets", "UNSW")
PACKET_RATE = 10
THRESHOLD = 0.5        

app = Flask(__name__)

sim = SimulationController(
    model_path=MODEL_PATH,
    data_dir=DATA_DIR,
    packet_rate=PACKET_RATE,
    threshold=THRESHOLD,
)

sim.start()


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return overall stats for the top cards."""
    status = sim.get_status()
    return jsonify(status)


@app.route("/api/packets")
def api_packets():
    """Return recent packets for the 'Live Network Traffic Flow' table."""
    packets = sim.get_packets(limit=100)
    frontend_packets = [
        {
            "id": p["id"],
            "timestamp": p["time"],
            "confidence": p["prob"],
            "classification": "ATTACK" if p["is_attack"] else "BENIGN",
        }
        for p in packets
    ]
    return jsonify(frontend_packets)


@app.route("/api/alerts")
def api_alerts():
    """Return recent alerts for the 'Quarantine / Alerts' table."""
    alerts = sim.get_alerts(limit=100)
    frontend_alerts = [
        {
            "id": a["id"],
            "timestamp": a["time"],
            "confidence": a["prob"],
            "action": a["action"],
        }
        for a in alerts
    ]
    return jsonify(frontend_alerts)


@app.route("/api/start", methods=["POST"])
def api_start():
    """(Optional) endpoint to start the simulation."""
    sim.start()
    return jsonify({"ok": True, "status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """(Optional) endpoint to stop the simulation."""
    sim.stop()
    return jsonify({"ok": True, "status": "stopped"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)