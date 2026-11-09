#!/usr/bin/env python3
"""
Step-1 numerical handoff check (Aaron brief, 2026-11-02, section 4.1).

For each development sky:
  A. Regenerate Linh's uniform-RA nulls from the recorded seed exactly as
     run_ra_scrambling.py does (default_rng(seed); one rng.uniform(0, 360, N)
     per row). If the saved scrambled_reco_ra_deg.npy is present, assert the
     regenerated rows are bit-identical to it.
  B. Recompute C_ab(theta) with pair_counter.py on the observed sky and every
     null, and assert exact equality with the saved results/counts_<id>.npz.
  C. Assert sum over the 15 unordered channel pairs == pooled count at every
     threshold, for observed + every null (and that a > b entries are zero).
  D. Independent direct pair calculation (vectorized haversine over all i<j,
     no KD-tree, no 3D vectors) on the observed sky and on one saved draw
     (row --draw-row), compared exactly with pair_counter.py.

Reads only reco_ra_deg, reco_dec_deg, reco_energy_channel (+ n_events/seed
metadata). Truth columns are never loaded.

Usage (from repo root):
  python code/step1_handoff_check.py --out results/step1_handoff_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_counter import THRESHOLDS_DEG, pair_counts_by_channel  # noqa: E402

SKY_DIRS = {
    "26392": "aaron_pair_counter_data_2026-10-24/uniform_26392",
    "14265": "aaron_pair_counter_data_2026-10-24/uniform_14265",
    "55289": "aaron_dev_extension_2026-10-31/uniform_55289",
    "192235": "aaron_dev_extension_2026-10-31/uniform_192235",
    "3332": "aaron_dev_extension_2026-10-31/uniform_3332",
}
OBS = ("reco_ra_deg", "reco_dec_deg", "reco_energy_channel")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def regenerate_nulls(seed: int, n_rows: int, n_events: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, n_events))
    for i in range(n_rows):
        out[i] = rng.uniform(0.0, 360.0, size=n_events)
    return out


def direct_counts(ra_deg, dec_deg, ch, thresholds=THRESHOLDS_DEG, block=512):
    """All i<j pairs, haversine on RA/Dec directly. Independent of pair_counter."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    ch = np.asarray(ch, np.int64)
    thr = np.asarray(thresholds, float)
    n = ra.size
    C = np.zeros((thr.size, 5, 5), np.int64)
    cosd = np.cos(dec)
    for s in range(0, n, block):
        i = np.arange(s, min(s + block, n))
        # only j > i
        h = (np.sin((dec[None, :] - dec[i, None]) / 2) ** 2
             + cosd[i, None] * cosd[None, :] * np.sin((ra[None, :] - ra[i, None]) / 2) ** 2)
        sep = np.degrees(2 * np.arcsin(np.minimum(1.0, np.sqrt(h))))
        upper = np.arange(n)[None, :] > i[:, None]
        ii, jj = np.nonzero(upper & (sep <= thr.max()))
        sp = sep[ii, jj]
        a = np.minimum(ch[i[ii]], ch[jj])
        b = np.maximum(ch[i[ii]], ch[jj])
        flat = a * 5 + b
        for t, th in enumerate(thr):
            m = sp <= th
            C[t] += np.bincount(flat[m], minlength=25).reshape(5, 5)
    return C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("results/step1_handoff_check.json"))
    p.add_argument("--draw-row", type=int, default=0, help="saved draw for the direct check")
    p.add_argument("--sims", nargs="+", default=list(SKY_DIRS))
    a = p.parse_args()

    seeds = json.loads((a.root / "aaron_dev_extension_2026-10-31/splits/"
                        "h1_dev_ra_null_seeds_2026-10-31.json").read_text())["development_null_runs"]
    report = {"check": "step1_numerical_handoff", "thresholds_deg": list(THRESHOLDS_DEG),
              "draw_row_direct_check": a.draw_row,
              "software": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__,
                           "pair_counter_sha256": sha256(Path(__file__).parent / "pair_counter.py")},
              "skies": {}}
    all_ok = True
    for sid in a.sims:
        t0 = time.time()
        d = a.root / SKY_DIRS[sid]
        summ = json.loads((d / "summary.json").read_text())
        with np.load(d / "event_inventory.npz", allow_pickle=False) as inv:
            ra, dec, ch = (np.asarray(inv[k]) for k in OBS)
        saved = np.load(a.root / f"results/counts_{sid}.npz")
        n_rows = saved["C_null"].shape[0]
        assert summ["seed"] == seeds[sid]["seed"] and summ["n_events"] == ra.size

        nulls = regenerate_nulls(summ["seed"], n_rows, ra.size)
        r = {"n_events": int(ra.size), "seed": int(summ["seed"]), "n_null": int(n_rows),
             "event_inventory_sha256": sha256(d / "event_inventory.npz")}
        npy = d / "scrambled_reco_ra_deg.npy"
        if npy.exists():
            r["A_regenerated_equals_saved_npy"] = bool(
                np.array_equal(np.load(npy, mmap_mode="r")[:n_rows], nulls))
        else:
            r["A_regenerated_equals_saved_npy"] = "npy not present here (regenerated from seed)"

        C_obs = pair_counts_by_channel(ra, dec, ch)
        C_null = np.stack([pair_counts_by_channel(nulls[i], dec, ch) for i in range(n_rows)])
        r["B_obs_equals_saved"] = bool(np.array_equal(C_obs, saved["C_obs"]))
        r["B_nulls_equal_saved"] = int(sum(np.array_equal(C_null[i], saved["C_null"][i])
                                           for i in range(n_rows)))

        allC = np.concatenate([C_obs[None], C_null])
        iu = np.triu_indices(5)
        lower_zero = bool(np.all(allC[:, :, np.tril_indices(5, -1)[0], np.tril_indices(5, -1)[1]] == 0))
        sum15 = allC[:, :, iu[0], iu[1]].sum(-1)
        pooled_direct = allC.sum(axis=(2, 3))
        r["C_sum15_equals_pooled_all_thresholds"] = bool(np.array_equal(sum15, pooled_direct)) and lower_zero
        r["pooled_obs"] = pooled_direct[0].tolist()

        D_obs = direct_counts(ra, dec, ch)
        D_draw = direct_counts(nulls[a.draw_row], dec, ch)
        r["D_direct_equals_counter_obs"] = bool(np.array_equal(D_obs, C_obs))
        r["D_direct_equals_counter_draw"] = bool(np.array_equal(D_draw, C_null[a.draw_row]))
        r["D_pooled_direct_draw"] = D_draw.sum(axis=(1, 2)).tolist()

        ok = (r["A_regenerated_equals_saved_npy"] is not False and r["B_obs_equals_saved"]
              and r["B_nulls_equal_saved"] == n_rows and r["C_sum15_equals_pooled_all_thresholds"]
              and r["D_direct_equals_counter_obs"] and r["D_direct_equals_counter_draw"])
        r["PASS"] = bool(ok)
        r["runtime_s"] = round(time.time() - t0, 1)
        all_ok &= ok
        report["skies"][sid] = r
        print(f"sky {sid}: B {r['B_nulls_equal_saved']}/{n_rows} nulls + obs={r['B_obs_equals_saved']}, "
              f"C sum15==pooled {r['C_sum15_equals_pooled_all_thresholds']}, "
              f"D direct obs={r['D_direct_equals_counter_obs']} draw{a.draw_row}={r['D_direct_equals_counter_draw']}, "
              f"A {r['A_regenerated_equals_saved_npy']} -> {'PASS' if ok else 'FAIL'} ({r['runtime_s']} s)",
              flush=True)
    report["ALL_PASS"] = bool(all_ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print("\nSTEP-1 CHECK PASS" if all_ok else "\nSTEP-1 CHECK FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
