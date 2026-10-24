"""
simulate_nodes.py - pretend to be the 3 ESP32 nodes (test the server with no hardware).

  python simulate_nodes.py --src 2.5 1.5            # fixed source at (2.5, 1.5) m
  python simulate_nodes.py --move                   # source walks in a circle
"""
import argparse
import math
import random
import time

import numpy as np
import requests

import localize as L

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:5000")
ap.add_argument("--src", nargs=2, type=float, default=[2.5, 1.5])
ap.add_argument("--amp", type=float, default=600.0, help="source strength A (mV*m^n)")
ap.add_argument("--move", action="store_true")
ap.add_argument("--warmup", type=int, default=12, help="clean-air seconds before baseline")
ap.add_argument("--duration", type=int, default=40)
a = ap.parse_args()

cfg = L.load_config()
ids, pts = L.node_arrays(cfg)
n, r0 = cfg["path_loss_exponent"], cfg["source_core_m"]
clean = {k: 650 + 40 * i for i, k in enumerate(ids)}    # different clean-air mV per node


def post(node, mv):
    requests.post(a.url + "/api/reading", json={
        "node": node, "mv": mv, "raw": int(mv), "temp": 27 + random.random(),
        "hum": 55 + random.random() * 3, "rssi": -55}, timeout=3)


print("Sending clean air ...")
for _ in range(a.warmup):
    for k in ids:
        post(k, clean[k] + random.gauss(0, 3))
    time.sleep(1)
print("Set baseline:", requests.post(a.url + "/api/baseline", timeout=3).json().get("ok"))

t0 = time.time()
while time.time() - t0 < a.duration:
    t = time.time() - t0
    src = np.array([2.0 + 1.5 * math.cos(t / 4), 1.3 + 1.0 * math.sin(t / 4)]) if a.move else np.array(a.src)
    s = L.forward_signal(pts, src, a.amp, n, r0) * np.random.lognormal(0, 0.12, len(ids))
    for k, sv in zip(ids, s):
        post(k, clean[k] + sv + random.gauss(0, 3))
    print(f"t={t:4.0f}s true source = ({src[0]:.2f}, {src[1]:.2f})")
    time.sleep(1)
