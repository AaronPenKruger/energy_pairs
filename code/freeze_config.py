#!/usr/bin/env python3
"""Write frozen/score_config.json with its self-hash. Run once; never edit by hand."""
import hashlib, json, sys
from pathlib import Path
cfg = {
    "name": "energy_granularity_scores_v1",
    "frozen_date": "2026-11-09",
    "thresholds_deg": [0.5, 1.0, 2.0, 3.0, 5.0],
    "n_channels": 5,
    "fields_read": ["reco_ra_deg", "reco_dec_deg", "reco_energy_channel"],
    "pair_rule": "unique unordered pairs i<j with angular separation <= theta",
    "reference": {
        "n_draws": 1000,
        "law": "independent RA ~ Uniform[0,360) per event; Dec and channel fixed",
        "entropy": 20261109,
        "stream": "SeedSequence(entropy, spawn_key=(inventory_key,)); events in (channel, Dec) order",
        "note": "separate from Linh's default_rng(int seed) null/calibration streams",
    },
    "pooled": {"statistic": "z_t=(N_t-mu_t)/sd_t", "combine": "max over thresholds"},
    "energy": {
        "min_ref_mean": 5.0,
        "var_floor": 1.0,
        "sparse_rule": "channel pairs with reference mean < min_ref_mean at threshold t are summed into one sparse cell; sparse cell dropped if its reference mean is 0",
        "statistic": "z_kt=(C_kt-mu_kt)/sqrt(sd_kt^2+var_floor); T_t=sum_k z_kt; Z_t=(T_t-mu_T)/sd_T",
        "combine": "max over thresholds",
    },
    "calibration_rule_for_linh": "p=(1+#{null>=obs})/(B+1); recompute full score incl. max over thresholds on every null",
}
cfg["config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
out = Path(sys.argv[1] if len(sys.argv) > 1 else "frozen/score_config.json")
if out.exists():
    sys.exit(f"{out} exists; refusing to overwrite a frozen config")
out.write_text(json.dumps(cfg, indent=2) + "\n")
print(out, cfg["config_sha256"])
