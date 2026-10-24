"""
app.py - the "localhost" server. ESP32 nodes POST readings over Wi-Fi,
this computes the source location and serves the live dashboard.

Run:  python app.py        then open  http://localhost:5000
"""
import csv
import json
import os
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import localize as L

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
BASELINE_PATH = os.path.join(DATA, "baseline.json")
LOG_PATH = os.path.join(DATA, "readings_log.csv")
REAL_CSV = os.path.join(DATA, "real_samples.csv")

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
lock = threading.Lock()

cfg = L.load_config()
IDS, PTS = L.node_arrays(cfg)
readings = {k: deque(maxlen=200) for k in IDS}   # node -> recent readings
baseline = {}                                      # node -> {mv,temp,hum}
if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)


def log_row(r):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "node", "mv", "raw", "temp", "hum", "rssi"])
        w.writerow([round(r["t"], 2), r["node"], r["mv"], r.get("raw"), r.get("temp"), r.get("hum"), r.get("rssi")])


def recent(node, window):
    now = time.time()
    return [r for r in readings[node] if now - r["t"] <= window]


def node_snapshot():
    """Smoothed latest value + signal-above-baseline for every node."""
    snap, sig = {}, []
    for k in IDS:
        rs = recent(k, cfg["smooth_window_seconds"])
        last = readings[k][-1] if readings[k] else None
        online = bool(last) and (time.time() - last["t"] <= cfg["stale_seconds"])
        d = {"x": cfg["nodes"][k]["x"], "y": cfg["nodes"][k]["y"], "online": online,
             "age": None if not last else round(time.time() - last["t"], 1),
             "temp": last.get("temp") if last else None, "hum": last.get("hum") if last else None,
             "mv": None, "baseline": None, "signal": None, "rssi": last.get("rssi") if last else None}
        mv = None
        if rs:
            # Weight recent readings higher for immediate response to gas spikes
            weights = np.linspace(0.6, 1.0, len(rs))
            weights /= weights.sum()
            mv = float(np.sum([r["mv"] * w for r, w in zip(rs, weights)]))
            d["mv"] = round(mv, 1)
        elif last and online:
            mv = float(last["mv"])
            d["mv"] = round(mv, 1)

        if mv is not None:
            b = baseline.get(k)
            if b is not None:
                corr = 0.0
                if last.get("temp") is not None and b.get("temp") is not None:
                    corr += cfg["temp_coeff_mv_per_c"] * (last["temp"] - b["temp"])
                if last.get("hum") is not None and b.get("hum") is not None:
                    corr += cfg["hum_coeff_mv_per_pct"] * (last["hum"] - b["hum"])
                d["baseline"] = round(b["mv"], 1)
                d["signal"] = round(max(mv - b["mv"] - corr, 0.0), 1)
        snap[k] = d
        sig.append(d["signal"] if (d["online"] and d["signal"] is not None) else None)
    return snap, sig


def classify_hazard(sig, nodes_snap):
    """Classify the detected pollution into severity tiers with actionable info.

    MQ-135 detects: NH3, NOx, alcohol, benzene, smoke, CO2.
    Signal thresholds are calibrated against typical MQ-135 response curves.
    """
    s = np.array(sig, float)
    total = float(s.sum())
    peak = float(s.max())

    # Severity tiers (mV rise above clean-air baseline)
    # These map real MQ-135 response to human-understandable risk levels
    if total < 12:
        return {
            "level": "safe",
            "severity": 0,
            "label": "Clean Air",
            "color": "#059669",
            "icon": "✅",
            "headline": "Air Quality Normal",
            "detail": "All sensor readings are at or near baseline. No elevated gas concentrations detected in the monitored area.",
            "action": "No action required. System is monitoring normally.",
            "gas_hint": None,
            "peak_mv": round(peak, 1),
            "total_mv": round(total, 1),
        }

    # Determine likely gas type from signal pattern and temperature
    gas_hints = []
    avg_temp = np.mean([n["temp"] for n in nodes_snap.values()
                        if n.get("temp") is not None] or [25])
    avg_hum = np.mean([n["hum"] for n in nodes_snap.values()
                       if n.get("hum") is not None] or [50])

    if avg_temp > 30:
        gas_hints.append("VOCs (volatile organic compounds from heat)")
    if avg_hum > 75:
        gas_hints.append("Humidity-amplified sensor response")
    # Generic MQ-135 gas detection labels
    gas_hints.append("Smoke / Combustion products")
    gas_hints.append("NH₃ (ammonia)")
    gas_hints.append("CO₂ (carbon dioxide)")
    gas_hints.append("Alcohol / Benzene vapours")

    if total < 40:
        return {
            "level": "low",
            "severity": 1,
            "label": "Trace Detection",
            "color": "#eab308",
            "icon": "🟡",
            "headline": "Faint Gas Trace Detected",
            "detail": f"A weak signal of {peak:.0f} mV (peak) has been picked up. This could be a distant or dissipating source, or background fluctuation. Total network signal: {total:.0f} mV.",
            "action": "Monitor the trend. If signal rises steadily, investigate the area near the dominant node. No immediate action needed.",
            "gas_hint": gas_hints[0] if gas_hints else "General air-quality gas (MQ-135)",
            "possible_gases": gas_hints[:3],
            "peak_mv": round(peak, 1),
            "total_mv": round(total, 1),
        }

    elif total < 120:
        return {
            "level": "moderate",
            "severity": 2,
            "label": "Moderate Pollution",
            "color": "#f97316",
            "icon": "🟠",
            "headline": "Active Gas Source — Moderate Level",
            "detail": f"Significant gas concentration detected. Peak node reads {peak:.0f} mV above baseline (total network: {total:.0f} mV). Source is likely within the sensor triangle.",
            "action": "Investigate the estimated source location. Open windows for ventilation. If source is an incense stick or test gas, this is expected. For unknown sources: check for gas leaks, smouldering materials, or chemical spills.",
            "gas_hint": gas_hints[0] if gas_hints else "General air-quality gas (MQ-135)",
            "possible_gases": gas_hints[:4],
            "peak_mv": round(peak, 1),
            "total_mv": round(total, 1),
        }

    elif total < 350:
        return {
            "level": "high",
            "severity": 3,
            "label": "High Concentration",
            "color": "#ef4444",
            "icon": "🔴",
            "headline": "Strong Gas Source — High Alert",
            "detail": f"High gas concentration! Peak sensor reads {peak:.0f} mV, total network: {total:.0f} mV. The source appears strong and close to the sensor field.",
            "action": "⚠️ Evacuate sensitive personnel from the area. Open all windows immediately. Identify and remove the source. If unknown origin: check for fire, gas leaks, or chemical spills. Consider shutting off HVAC to prevent spread.",
            "gas_hint": gas_hints[0] if gas_hints else "General air-quality gas (MQ-135)",
            "possible_gases": gas_hints[:4],
            "peak_mv": round(peak, 1),
            "total_mv": round(total, 1),
        }

    else:
        return {
            "level": "critical",
            "severity": 4,
            "label": "Critical — Dangerous",
            "color": "#dc2626",
            "icon": "🚨",
            "headline": "CRITICAL: Dangerous Gas Levels",
            "detail": f"Extremely high readings! Peak: {peak:.0f} mV, total: {total:.0f} mV. Sensor is near or at saturation. This indicates a very strong, close, or concentrated gas source.",
            "action": "🚨 EVACUATE the area immediately! Do NOT attempt to locate the source. Open doors/windows from outside if possible. Call emergency services if source is unknown. Do NOT use electrical switches near the area (spark risk with flammable gas).",
            "gas_hint": "Multiple gases / high concentration",
            "possible_gases": gas_hints,
            "peak_mv": round(peak, 1),
            "total_mv": round(total, 1),
        }


def compute_estimates(sig, nodes_snap=None):
    out = {"centroid": None, "trilateration": None, "ai": None, "final": None, "status": "",
           "hazard": classify_hazard([0, 0, 0], {}) if nodes_snap is None else None}

    if any(s is None for s in sig):
        missing = [k for k, s in zip(IDS, sig) if s is None]
        out["status"] = "Waiting for node(s): " + ", ".join(missing) + (
            " (set baseline?)" if not baseline else "")
        out["hazard"] = {
            "level": "unknown", "severity": -1, "label": "Sensors Offline",
            "color": "#64748b", "icon": "⏳",
            "headline": "Waiting for Sensor Data",
            "detail": "Not all nodes are reporting or baseline is missing. Cannot assess air quality yet.",
            "action": "Ensure all ESP32 nodes are powered on and connected to the same Wi-Fi network. Set a clean-air baseline if not done.",
            "gas_hint": None, "peak_mv": 0, "total_mv": 0
        }
        return out

    s = np.array(sig, float)
    out["hazard"] = classify_hazard(sig, nodes_snap or {})

    if s.sum() < cfg["min_total_signal_mv"]:
        out["status"] = "Air looks clean - no pollution source detected"
        return out

    cen = L.weighted_centroid(PTS, s)
    tri = L.trilaterate(PTS, s, n=cfg["path_loss_exponent"], r0=cfg["source_core_m"])
    tri_xy = None if tri is None else (tri[0], tri[1])
    ai = L.ai_predict(IDS, s)
    fin = L.fuse(tri_xy, ai, cen)

    def pack(p):
        if p is None:
            return None
        lat, lng = L.xy_to_latlng(p[0], p[1], cfg)
        return {"x": round(p[0], 2), "y": round(p[1], 2), "lat": round(lat, 7), "lng": round(lng, 7)}

    out.update(centroid=pack(cen), trilateration=pack(tri_xy), ai=pack(ai),
               final=pack(fin) if fin else None)
    out["method"] = fin[2] if fin else None
    out["status"] = "Pollution source located"
    return out



@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/reading", methods=["POST"])
def api_reading():
    j = request.get_json(force=True, silent=True) or {}
    node = str(j.get("node", ""))
    if node not in readings or "mv" not in j:
        return jsonify(error="unknown node or missing mv"), 400
    r = {"t": time.time(), "node": node, "mv": float(j["mv"]), "raw": j.get("raw"),
         "temp": j.get("temp"), "hum": j.get("hum"), "rssi": j.get("rssi")}
    with lock:
        readings[node].append(r)
        log_row(r)
    return jsonify(ok=True)


@app.route("/api/state")
def api_state():
    with lock:
        snap, sig = node_snapshot()
        est = compute_estimates(sig, snap)
        samples_count = 0
        if os.path.exists(REAL_CSV):
            try:
                with open(REAL_CSV) as f:
                    samples_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                pass
    return jsonify(nodes=snap, estimates=est, baseline_set=bool(baseline),
                   exponent=cfg["path_loss_exponent"], smooth_window_seconds=cfg["smooth_window_seconds"],
                   samples_count=samples_count, time=time.time())


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global cfg
    if request.method == "POST":
        j = request.get_json(force=True, silent=True) or {}
        with lock:
            if "smooth_window_seconds" in j:
                cfg["smooth_window_seconds"] = max(0.2, float(j["smooth_window_seconds"]))
            if "min_total_signal_mv" in j:
                cfg["min_total_signal_mv"] = max(1.0, float(j["min_total_signal_mv"]))
            if "stale_seconds" in j:
                cfg["stale_seconds"] = max(1.0, float(j["stale_seconds"]))
            try:
                with open(L.CONFIG_PATH, "w") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass
        return jsonify(ok=True, config=cfg)
    return jsonify(cfg)


@app.route("/api/history")
def api_history():
    with lock:
        out = {k: list(readings[k]) for k in IDS}
    return jsonify(history=out)


@app.route("/api/baseline", methods=["POST"])
def api_baseline():
    """Call this in CLEAN air. Stores the average of the last ~10 s as 'zero'."""
    with lock:
        new = {}
        for k in IDS:
            rs = recent(k, 10)
            if len(rs) < 3:
                return jsonify(error=f"node {k} has too few recent readings"), 400
            new[k] = {"mv": float(np.mean([r["mv"] for r in rs])),
                      "temp": rs[-1].get("temp"), "hum": rs[-1].get("hum")}
        baseline.clear()
        baseline.update(new)
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
    return jsonify(ok=True, baseline=baseline)


@app.route("/api/baseline/clear", methods=["POST"])
def api_baseline_clear():
    with lock:
        baseline.clear()
        if os.path.exists(BASELINE_PATH):
            try:
                os.remove(BASELINE_PATH)
            except Exception:
                pass
    return jsonify(ok=True)


@app.route("/api/label", methods=["POST"])
def api_label():
    """Save current signals + the TRUE source position -> real training sample."""
    j = request.get_json(force=True, silent=True) or {}
    try:
        x, y = float(j["x"]), float(j["y"])
    except (KeyError, ValueError, TypeError):
        return jsonify(error="send numeric x and y"), 400
    with lock:
        snap, sig = node_snapshot()
    if any(s is None for s in sig):
        return jsonify(error="not all nodes online / baseline missing"), 400
    new = not os.path.exists(REAL_CSV)
    with open(REAL_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["x", "y"] + ["s_" + k for k in IDS])
        w.writerow([x, y] + sig)
    n = sum(1 for _ in open(REAL_CSV)) - 1
    return jsonify(ok=True, samples=n)


if __name__ == "__main__":
    print("Nodes:", {k: (v["x"], v["y"]) for k, v in cfg["nodes"].items()})
    print("AI model:", "loaded" if L.load_model() else "NOT trained yet (run python train.py)")
    app.run(host="0.0.0.0", port=5000, threaded=True)
