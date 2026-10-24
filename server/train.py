"""
train.py - trains the AI localizer.

Step 1 (always): pre-train a RandomForest on simulated plumes.
Step 2 (optional, recommended): if data/real_samples.csv exists (you create it
        with the dashboard's "Log sample" button while a source sits at a KNOWN
        position), the path-loss exponent n is fitted to your real data and the
        real samples are blended into training with extra weight.

Run:  python train.py
"""
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import localize as L

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_CSV = os.path.join(HERE, "data", "real_samples.csv")
REAL_WEIGHT = 20


def load_real(ids):
    if not os.path.exists(REAL_CSV):
        return None, None
    S, T = [], []
    with open(REAL_CSV) as f:
        for row in csv.DictReader(f):
            try:
                S.append([float(row["s_" + k]) for k in ids])
                T.append([float(row["x"]), float(row["y"])])
            except (KeyError, ValueError):
                pass
    if not S:
        return None, None
    return np.array(S), np.array(T)


def fit_exponent(pts, S, T, r0):
    best_n, best_err = None, 1e9
    for n in np.arange(0.8, 4.01, 0.2):
        errs = []
        for s, t in zip(S, T):
            r = L.trilaterate(pts, s, n=n, r0=r0)
            if r is not None:
                errs.append(np.hypot(r[0] - t[0], r[1] - t[1]))
        if len(errs) >= 5 and np.mean(errs) < best_err:
            best_n, best_err = float(n), float(np.mean(errs))
    return best_n, best_err


def main():
    cfg = L.load_config()
    ids, pts = L.node_arrays(cfg)
    n, r0 = cfg["path_loss_exponent"], cfg["source_core_m"]
    rng = np.random.default_rng(42)

    S_real, T_real = load_real(ids)
    if S_real is not None and len(S_real) >= 10:
        n_fit, err = fit_exponent(pts, S_real, T_real, r0)
        if n_fit:
            print(f"Fitted path-loss exponent from {len(S_real)} real samples: n={n_fit} (trilat err {err:.2f} m)")
            n = n_fit
            cfg["path_loss_exponent"] = n
            with open(L.CONFIG_PATH, "w") as f:
                import json
                json.dump(cfg, f, indent=2)

    X, Y = L.simulate_training_set(pts, n, r0, 20000, rng)
    w = np.ones(len(X))
    if S_real is not None and len(S_real):
        Xr = np.array([L.features(s) for s in S_real if L.features(s) is not None])
        Tr = np.array([t for s, t in zip(S_real, T_real) if L.features(s) is not None])
        X = np.vstack([X, Xr])
        Y = np.vstack([Y, Tr])
        w = np.concatenate([w, np.full(len(Xr), REAL_WEIGHT)])
        print(f"Blended {len(Xr)} real samples (weight x{REAL_WEIGHT})")

    Xtr, Xte, Ytr, Yte, wtr, _ = train_test_split(X, Y, w, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=8, n_jobs=-1, random_state=1)
    model.fit(Xtr, Ytr, sample_weight=wtr)
    err = np.hypot(*(model.predict(Xte) - Yte).T)
    print(f"Hold-out error  mean={err.mean():.2f} m   median={np.median(err):.2f} m   (n={n})")

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    joblib.dump({"model": model, "nodes": ids, "n": n}, L.MODEL_PATH)
    print("Saved", L.MODEL_PATH)


if __name__ == "__main__":
    main()
