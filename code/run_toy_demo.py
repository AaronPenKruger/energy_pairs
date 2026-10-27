#!/usr/bin/env python3
"""
Runnable small synthetic example (TOY skies, not the IceCube simulator).

For a few values of f_multiplet (toy analog of g_comp):
  1. write a toy sky in Linh's event_inventory.npz format,
  2. make uniform-RA nulls by calling LINH's draw_uniform_ra (no second sampler),
  3. run the unchanged run_pair_counts.py on it,
  4. print pooled and high-channel excesses as a development diagnostic.

Usage (from ~/energy_pairs):
  python code/run_toy_demo.py --out toy --n-null 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthetic_sky import ToySkyConfig, write_toy_sky  # noqa: E402

DEFAULT_LINH = HERE.parent / "aaron_uniform_ra_handoff_2026-10-21" / "code" / "run_ra_scrambling.py"


def load_linh_sampler(path: Path):
    spec = importlib.util.spec_from_file_location("linh_run_ra_scrambling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.draw_uniform_ra


def make_nulls(sky_dir: Path, n_null: int, seed: int, draw_uniform_ra, sim_id: int):
    with np.load(sky_dir / "event_inventory.npz") as d:
        ra = d["reco_ra_deg"]
    rng = np.random.default_rng(seed)
    mask = np.ones(ra.size, bool)
    null = np.empty((n_null, ra.size))
    for i in range(n_null):
        null[i] = draw_uniform_ra(ra, mask, rng)
    np.save(sky_dir / "scrambled_reco_ra_deg.npy", null)
    # same keys run_pair_counts.py reads from Linh's summary.json
    s = json.loads((sky_dir / "summary.json").read_text())
    s.update({"global_sim_id": sim_id, "method": "uniform-full-ra (Linh draw_uniform_ra)",
              "seed": seed, "n_scrambles": n_null, "n_events": int(ra.size)})
    (sky_dir / "summary.json").write_text(json.dumps(s, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("toy"))
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--f-multiplet", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    p.add_argument("--multiplet-size", type=int, default=3)
    p.add_argument("--n-astro", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--linh-sampler", type=Path, default=DEFAULT_LINH)
    args = p.parse_args()

    draw_uniform_ra = load_linh_sampler(args.linh_sampler)
    rows = []
    for k, f in enumerate(args.f_multiplet):
        sim_id = 900000 + k
        sky = args.out / f"toy_{sim_id}"
        cfg = ToySkyConfig(seed=args.seed * 1000 + k, n_astro=args.n_astro,
                           f_multiplet=f, multiplet_size=args.multiplet_size)
        write_toy_sky(cfg, sky)
        make_nulls(sky, args.n_null, seed=args.seed * 1000 + 500 + k,
                   draw_uniform_ra=draw_uniform_ra, sim_id=sim_id)
        out = args.out / f"counts_{sim_id}.npz"
        subprocess.run([sys.executable, str(HERE / "run_pair_counts.py"),
                        "--null-dir", str(sky), "--out", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        d = np.load(out)
        meta = json.loads((sky / "summary.json").read_text())
        co, cn = d["C_obs"], d["C_null"]
        pooled_o, pooled_n = co.sum((1, 2)), cn.sum((2, 3))
        # pairs where at least one event is above channel 0 (diagnostic only)
        hi_o = pooled_o - co[:, 0, 0]
        hi_n = pooled_n - cn[:, :, 0, 0]
        z = lambda o, n: (o - n.mean(0)) / n.std(0, ddof=1)
        rows.append((f, meta["n_multiplet_sources"], meta["n_astro_in_multiplets"],
                     z(pooled_o, pooled_n), z(hi_o, hi_n)))

    print("TOY skies: n_astro=%d, multiplet size=%d, %d nulls each (development diagnostic only)"
          % (args.n_astro, args.multiplet_size, args.n_null))
    print(" f_mult  n_mult_src  astro_in_mult |  pooled z @0.5,1,2 deg   |  ch>=1 pairs z @0.5,1,2 deg")
    for f, nm, na, zp, zh in rows:
        print(f"  {f:4.2f}  {nm:9d}  {na:13d} | {zp[0]:6.1f} {zp[1]:6.1f} {zp[2]:6.1f}       | "
              f"{zh[0]:6.1f} {zh[1]:6.1f} {zh[2]:6.1f}")
    print("\nz = (observed - null mean)/null sd. Not a calibrated significance; toy skies only.")


if __name__ == "__main__":
    main()
