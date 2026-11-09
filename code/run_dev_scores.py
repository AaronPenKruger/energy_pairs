#!/usr/bin/env python3
"""
Development diagnostics for the frozen scores (NOT calibration, NOT evaluation).

For each development sky: build the reference (separate stream), save its
fitted numbers to frozen/reference_<id>.npz, and score the observed sky and
its 1000 existing development nulls (from results/counts_<id>.npz).
Development p-values use the same formula Linh will use, but these nulls are
development draws, not calibration draws, and 26392/14265 share one null seed.

Usage: python code/run_dev_scores.py --out results/dev_scores.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustering_scores as cs
from step1_handoff_check import SKY_DIRS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/dev_scores.json"))
    p.add_argument("--extra", nargs="*", default=[], help="name=dir:counts.npz (e.g. toy skies)")
    a = p.parse_args()
    cfg = cs.load_config()
    skies = {k: (Path(v), Path(f"results/counts_{k}.npz")) for k, v in SKY_DIRS.items()}
    for e in a.extra:
        name, rest = e.split("=", 1); d, c = rest.split(":")
        skies[name] = (Path(d), Path(c))
    res = {"config_sha256": cfg["config_sha256"], "status": "development diagnostic only", "skies": {}}
    for sid, (d, cpath) in skies.items():
        t0 = time.time()
        with np.load(d / "event_inventory.npz", allow_pickle=False) as inv:
            ev = {k: np.asarray(inv[k]) for k in cs.OBSERVABLE_FIELDS}
        dec, ch = ev["reco_dec_deg"].astype(float), ev["reco_energy_channel"].astype(np.int64)
        ref = cs.get_reference(dec, ch, cfg)
        np.savez_compressed(Path("frozen") / f"reference_{sid}.npz",
            inventory_key=np.uint64(cs.inventory_key(dec, ch)),
            pooled_mu=ref["pooled_mu"], pooled_sd=ref["pooled_sd"], dense=ref["dense"],
            T_mu=ref["T_mu"], T_sd=ref["T_sd"],
            **{f"cell_mu_t{t}": ref["cell_mu"][t] for t in range(5)},
            **{f"cell_den_t{t}": ref["cell_den"][t] for t in range(5)},
            **{f"cell_keep_t{t}": ref["cell_keep"][t] for t in range(5)})
        counts = np.load(cpath)
        obs = cs.score_both(ev, cfg)
        r = {"n_events": int(dec.size), "inventory_key": str(cs.inventory_key(dec, ch))}
        for m in ("pooled", "energy"):
            nul = np.array([cs.per_scale(Cn, ref, m).max() for Cn in counts["C_null"]])
            zs = cs.per_scale(counts["C_obs"], ref, m)
            r[m] = {"S_obs": obs[m], "argmax_theta_deg": float(cfg["thresholds_deg"][int(zs.argmax())]),
                    "per_scale": np.round(zs, 3).tolist(),
                    "dev_p": float((1 + (nul >= obs[m]).sum()) / (nul.size + 1)),
                    "null_mean": float(nul.mean()), "null_sd": float(nul.std(ddof=1)),
                    "null_q95": float(np.quantile(nul, 0.95)), "n_null": int(nul.size),
                    "n_null_ties_with_obs": int((nul == obs[m]).sum())}
            assert np.isclose(cs.per_scale(counts["C_obs"], ref, m).max(), obs[m])
        r["dense_cells_per_theta"] = ref["dense"].sum(1).tolist()
        r["runtime_s"] = round(time.time() - t0, 1)
        res["skies"][sid] = r
        print(f"{sid:>8}: pooled S={obs['pooled']:6.2f} p={r['pooled']['dev_p']:.3f} "
              f"(theta {r['pooled']['argmax_theta_deg']})   energy S={obs['energy']:6.2f} "
              f"p={r['energy']['dev_p']:.3f} (theta {r['energy']['argmax_theta_deg']})  "
              f"dense/theta {r['dense_cells_per_theta']}  {r['runtime_s']} s", flush=True)
    a.out.write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
