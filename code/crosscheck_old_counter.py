#!/usr/bin/env python3
"""
Cross-check pair_counter.py against the older pair-count code
(scripts/analyze_ra_pair_counts.py on analysis/scrambling-background, from the
dfe378f era; still present unchanged at bac169d).

Compares all 15 channel-pair counts x 5 thresholds, plus the pooled count, on
the observed sky and every null realization.

Usage (from ~/energy_pairs):
  python code/crosscheck_old_counter.py \
      --old ~/scrambling_background/scripts/analyze_ra_pair_counts.py \
      --data aaron_pair_counter_data_2026-10-24 --sims 26392 14265
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_counter import THRESHOLDS_DEG, pair_counts_by_channel  # noqa: E402


def load_old(path: Path):
    spec = importlib.util.spec_from_file_location("old_pair_counts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_statistics


def old_as_C(compute_statistics, ra, dec, ch):
    st = compute_statistics(ra, dec, ch, list(THRESHOLDS_DEG))
    C = np.zeros((len(THRESHOLDS_DEG), 5, 5), np.int64)
    for a in range(5):
        for b in range(a, 5):
            C[:, a, b] = st[f"ch{a}-ch{b}"]
    return C, st["all"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--sims", nargs="+", default=["26392", "14265"])
    p.add_argument("--max-null", type=int, default=None)
    args = p.parse_args()

    compute_statistics = load_old(args.old)
    all_ok = True
    for sid in args.sims:
        d = args.data / f"uniform_{sid}"
        with np.load(d / "event_inventory.npz") as inv:
            ra0, dec, ch = inv["reco_ra_deg"], inv["reco_dec_deg"], inv["reco_energy_channel"]
        null = np.load(d / "scrambled_reco_ra_deg.npy", mmap_mode="r")
        if args.max_null:
            null = null[: args.max_null]
        skies = [ra0] + [null[i] for i in range(null.shape[0])]
        n_ok = n_pooled_ok = 0
        first_bad = None
        t0 = time.time()
        for k, ra in enumerate(skies):
            C_old, all_old = old_as_C(compute_statistics, ra, dec, ch)
            C_new = pair_counts_by_channel(ra, dec, ch)
            if np.array_equal(C_old, C_new):
                n_ok += 1
            elif first_bad is None:
                first_bad = (k, np.argwhere(C_old != C_new)[:5].tolist())
            n_pooled_ok += np.array_equal(all_old, C_new.sum(axis=(1, 2)))
        ok = n_ok == len(skies) and n_pooled_ok == len(skies)
        all_ok &= ok
        print(f"sky {sid}: {len(skies)} skies (observed + {len(skies) - 1} nulls): "
              f"channel-pair counts identical {n_ok}/{len(skies)}, pooled identical "
              f"{n_pooled_ok}/{len(skies)}, first mismatch {first_bad}, {time.time() - t0:.0f} s "
              f"-> {'PASS' if ok else 'FAIL'}")
    print("\nCROSS-CHECK PASS" if all_ok else "\nCROSS-CHECK FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
