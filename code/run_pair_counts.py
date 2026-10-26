#!/usr/bin/env python3
"""
Compute energy-resolved pair counts C_ab(theta) for one mixed sky and for
each of Linh's uniform-RA null realizations.

Reads ONLY reco_ra_deg, reco_dec_deg, reco_energy_channel from
event_inventory.npz, and the scrambled RA rows. Truth columns are not loaded.

Usage:
  python run_pair_counts.py --null-dir <.../uniform_26392> --out counts_26392.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from pair_counter import (
    N_CHANNELS,
    THRESHOLDS_DEG,
    channel_pair_labels,
    pair_counts_by_channel,
    pooled_counts,
)

OBSERVABLE_FIELDS = ("reco_ra_deg", "reco_dec_deg", "reco_energy_channel")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--null-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-null", type=int, default=None,
                   help="use only the first N null rows (quick test)")
    args = p.parse_args()

    inv_path = args.null_dir / "event_inventory.npz"
    ra_path = args.null_dir / "scrambled_reco_ra_deg.npy"
    summary = json.loads((args.null_dir / "summary.json").read_text())

    with np.load(inv_path, allow_pickle=False) as d:
        ra, dec, ch = (np.asarray(d[k]) for k in OBSERVABLE_FIELDS)
    null_ra = np.load(ra_path, mmap_mode="r")
    if args.max_null:
        null_ra = null_ra[: args.max_null]

    # producer -> consumer consistency checks
    assert null_ra.shape[1] == ra.size == summary["n_events"], "event count mismatch"
    assert null_ra.shape[0] <= summary["n_scrambles"]

    t0 = time.time()
    C_obs = pair_counts_by_channel(ra, dec, ch)
    C_null = np.empty((null_ra.shape[0],) + C_obs.shape, dtype=np.int64)
    for i in range(null_ra.shape[0]):
        C_null[i] = pair_counts_by_channel(null_ra[i], dec, ch)
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        C_obs=C_obs,
        C_null=C_null,
        thresholds_deg=np.asarray(THRESHOLDS_DEG),
        channel_pairs=np.asarray(channel_pair_labels(N_CHANNELS)),
    )
    meta = {
        "global_sim_id": summary["global_sim_id"],
        "null_method": summary["method"],
        "null_seed": summary["seed"],
        "n_events": int(ra.size),
        "n_null_used": int(null_ra.shape[0]),
        "thresholds_deg": list(THRESHOLDS_DEG),
        "separation_rule": "pair counted if separation <= theta",
        "fields_read": list(OBSERVABLE_FIELDS),
        "inputs_sha256": {inv_path.name: sha256(inv_path), ra_path.name: sha256(ra_path)},
        "runtime_s": round(elapsed, 1),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    po, pn = pooled_counts(C_obs), pooled_counts(C_null)
    print(f"sky {summary['global_sim_id']}: {ra.size} events, "
          f"{null_ra.shape[0]} nulls, {elapsed:.1f} s")
    print(" theta   observed   null mean   null sd   (development diagnostic only)")
    for t, theta in enumerate(THRESHOLDS_DEG):
        print(f" {theta:4.1f}  {po[t]:9d}  {pn[:, t].mean():10.1f}  {pn[:, t].std(ddof=1):8.1f}")
    print(f"saved {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
