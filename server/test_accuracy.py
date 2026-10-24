"""
test_accuracy.py - offline Monte-Carlo test of the 3 estimators (no hardware, no server).
Sources are dropped at random spots INSIDE the node triangle with sensor noise added.
"""
import argparse

import numpy as np

import localize as L

ap = argparse.ArgumentParser()
ap.add_argument("--true-n", type=float, default=None, help="real-world exponent (model mismatch test)")
ap.add_argument("--outside", action="store_true", help="drop sources anywhere in the expanded area, not only inside the triangle")
args = ap.parse_args()

cfg = L.load_config()
ids, pts = L.node_arrays(cfg)
n, r0 = cfg["path_loss_exponent"], cfg["source_core_m"]
rng = np.random.default_rng(7)

# random points inside the triangle (barycentric sampling)
def inside_triangle(P):
    u, v = rng.random(2)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    return P[0] + u * (P[1] - P[0]) + v * (P[2] - P[0])

errs = {"centroid": [], "trilateration": [], "ai": [], "final": []}
for _ in range(500):
    if args.outside:
        lo, hi = L.bounds_of(pts)
        src = rng.uniform(lo, hi)
    else:
        src = inside_triangle(pts)
    A = float(np.exp(rng.uniform(np.log(200), np.log(1200))))
    s = L.forward_signal(pts, src, A, args.true_n or n, r0)
    s = np.maximum(s * rng.lognormal(0, 0.15, len(s)) + rng.normal(0, 4, len(s)), 0)
    if s.sum() < cfg["min_total_signal_mv"]:
        continue
    cen = L.weighted_centroid(pts, s)
    tri = L.trilaterate(pts, s, n=n, r0=r0)
    tri_xy = None if tri is None else tri[:2]
    ai = L.ai_predict(ids, s)
    fin = L.fuse(tri_xy, ai, cen)
    for name, p in [("centroid", cen), ("trilateration", tri_xy), ("ai", ai), ("final", fin)]:
        if p is not None:
            errs[name].append(np.hypot(p[0] - src[0], p[1] - src[1]))

side = np.linalg.norm(pts[1] - pts[0])
print(f"true n={args.true_n or n} (model assumes {n}), sources {'anywhere' if args.outside else 'inside triangle'}")
print(f"Triangle side ~{side:.1f} m, noise: 15% multiplicative + 4 mV, simulated data\n")
print(f"{'method':<15}{'mean err (m)':>14}{'median (m)':>12}{'90th pct (m)':>14}{'samples':>9}")
for k, v in errs.items():
    if v:
        v = np.array(v)
        print(f"{k:<15}{v.mean():>14.2f}{np.median(v):>12.2f}{np.percentile(v, 90):>14.2f}{len(v):>9}")
