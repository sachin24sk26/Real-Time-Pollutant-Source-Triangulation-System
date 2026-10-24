"""
localize.py - all the maths of the project in one place.

Three estimators (same idea as the slides):
  1. weighted_centroid : x = sum(Ci*Xi)/sum(Ci)           (simple baseline)
  2. trilaterate       : concentration -> distance model, then least-squares
                         intersection of the 3 distance "circles"
  3. ai_predict        : RandomForest trained on simulated + your real samples

Signal model (gas spreading from a point source):
        s_i = A / (d_i^2 + r0^2)^(n/2)
    s_i : sensor rise above clean-air baseline (mV)
    d_i : distance node i <-> source (m)
    A   : source strength,  n : path-loss exponent (~2, fitted in train.py)
    r0  : small "core" radius so the formula does not blow up near the source
"""
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
MODEL_PATH = os.path.join(HERE, "data", "model.joblib")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def node_arrays(cfg):
    ids = list(cfg["nodes"].keys())
    pts = np.array([[cfg["nodes"][k]["x"], cfg["nodes"][k]["y"]] for k in ids], float)
    return ids, pts


def bounds_of(pts, margin_frac=0.25):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    m = np.maximum((hi - lo) * margin_frac, 0.5)
    return lo - m, hi + m


def forward_signal(pts, src, A, n, r0):
    """Model: signal each node would read for a source at `src`."""
    d2 = ((pts - np.asarray(src)) ** 2).sum(axis=1)
    return A / (d2 + r0 ** 2) ** (n / 2.0)


# ---------------------------------------------------------------- 1. centroid
def weighted_centroid(pts, s):
    s = np.asarray(s, float)
    tot = s.sum()
    if tot <= 0:
        return None
    x = float((pts[:, 0] * s).sum() / tot)
    y = float((pts[:, 1] * s).sum() / tot)
    return x, y


# ------------------------------------------------------------ 2. trilateration
def trilaterate(pts, s, n=2.0, r0=0.3, lam=0.02):
    """
    Solve for (x, y, lnA) from  ln(s_i) = lnA - n * ln(d_eff_i).
    Unknown source strength A is solved together with the position, so no
    per-node distance calibration is needed. 3 nodes -> 3 equations, 3 unknowns;
    a 4th node makes it over-determined (more accurate).
    A weak pull (lam) toward the centroid removes the mirror-image ambiguity.
    Returns (x, y, residual_cost) or None.
    """
    s = np.asarray(s, float)
    ok = s > 1e-6
    if ok.sum() < 3:
        return None
    P, S = pts[ok], s[ok]
    lnS = np.log(S)
    c0 = weighted_centroid(P, S)
    lo, hi = bounds_of(pts)

    def resid(p):
        x, y, lnA = p
        d2 = (P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2 + r0 ** 2
        model = lnA - (n / 2.0) * np.log(d2)
        reg = lam * np.array([x - c0[0], y - c0[1]])
        return np.concatenate([model - lnS, reg])

    starts = [c0] + [(0.5 * (c0[0] + px), 0.5 * (c0[1] + py)) for px, py in P]
    best = None
    for sx, sy in starts:
        lnA0 = float(np.mean(lnS + (n / 2.0) * np.log(((P - [sx, sy]) ** 2).sum(1) + r0 ** 2)))
        try:
            r = least_squares(
                resid, [sx, sy, lnA0],
                bounds=([lo[0], lo[1], -10], [hi[0], hi[1], 25]),
                loss="soft_l1", f_scale=0.3,
            )
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    return float(best.x[0]), float(best.x[1]), float(best.cost)


# ------------------------------------------------------------------- 3. AI
def features(s):
    """Scale-free features: share of total signal per node + log(total)."""
    s = np.maximum(np.asarray(s, float), 0.0)
    tot = s.sum()
    if tot <= 0:
        return None
    return np.concatenate([s / tot, [np.log1p(tot)]])


def simulate_training_set(pts, n, r0, n_samples, rng, noise_rel=0.15, noise_abs=4.0):
    lo, hi = bounds_of(pts)
    X, Y = [], []
    while len(X) < n_samples:
        src = rng.uniform(lo, hi)
        A = float(np.exp(rng.uniform(np.log(100), np.log(1500))))
        s = forward_signal(pts, src, A, n, r0)
        s = s * rng.lognormal(0, noise_rel, size=len(s)) + rng.normal(0, noise_abs, size=len(s))
        s = np.maximum(s, 0)
        f = features(s)
        if f is None:
            continue
        X.append(f)
        Y.append(src)
    return np.array(X), np.array(Y)


_model_cache = {"mtime": None, "bundle": None}


def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None
    mt = os.path.getmtime(MODEL_PATH)
    if _model_cache["mtime"] != mt:
        _model_cache["bundle"] = joblib.load(MODEL_PATH)
        _model_cache["mtime"] = mt
    return _model_cache["bundle"]


def ai_predict(node_ids, s):
    b = load_model()
    if b is None or b["nodes"] != list(node_ids):
        return None
    f = features(s)
    if f is None:
        return None
    x, y = b["model"].predict([f])[0]
    return float(x), float(y)


def fuse(tri, ai, cen):
    """Final answer: mean of trilateration and AI when both exist."""
    if tri is not None and ai is not None:
        return 0.5 * (tri[0] + ai[0]), 0.5 * (tri[1] + ai[1]), "trilateration + AI"
    if tri is not None:
        return tri[0], tri[1], "trilateration"
    if ai is not None:
        return ai[0], ai[1], "AI"
    if cen is not None:
        return cen[0], cen[1], "centroid"
    return None


def xy_to_latlng(x, y, cfg):
    lat0, lng0 = cfg["origin_lat"], cfg["origin_lng"]
    lat = lat0 + y / 111320.0
    lng = lng0 + x / (111320.0 * math.cos(math.radians(lat0)))
    return lat, lng
