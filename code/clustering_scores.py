"""
Two frozen clustering scores for the energy-resolved granularity study.

    score(events, method, config) -> float      method in {"pooled", "energy"}

Both scores read ONLY reco_ra_deg, reco_dec_deg, reco_energy_channel.
Truth columns (component, source_index, true_*, psf_deg) are never read.

------------------------------------------------------------------------------
Reference ensemble (fitted numbers)
------------------------------------------------------------------------------
Centering/scaling come from R reference draws per sky. Each reference draw
gives every event an independent RA ~ Uniform[0, 360) and keeps its Dec and
energy channel, i.e. the same law as Linh's uniform-RA null. Because a
uniform-RA draw changes only RA, the reference depends only on the sky's
(Dec, channel) inventory, so the original sky and every randomized version
of it get the SAME reference and are scored identically.

The reference stream is separate from every null/calibration stream:
    SeedSequence(entropy=config["reference"]["entropy"],
                 spawn_key=(inventory_key,))
where inventory_key is a 64-bit hash of the event inventory sorted by
(channel, Dec). Linh's nulls use default_rng(<int seed>) (no spawn key),
which is a different stream. Do not use calibration draws as reference.

Events are put in canonical (channel, Dec) order before drawing, so the
reference (and the score) do not depend on event order.

------------------------------------------------------------------------------
Pooled score
------------------------------------------------------------------------------
    N_t      = pooled pair count at threshold t
    z_t      = (N_t - mu_t) / sd_t                 (mu, sd over reference draws)
    S_pool   = max_t z_t                           t over 0.5, 1, 2, 3, 5 deg

------------------------------------------------------------------------------
Energy-resolved score
------------------------------------------------------------------------------
For each threshold t, the 15 unordered channel pairs (a<=b) are split into
"dense" cells (reference mean >= min_ref_mean) and one merged "sparse" cell
that sums all remaining channel pairs at that t (dropped if its reference
mean is exactly 0).

    z_kt     = (C_kt - mu_kt) / sqrt(sd_kt^2 + var_floor)
    T_t      = sum_k z_kt                          (equal weights)
    Z_t      = (T_t - mu_T_t) / sd_T_t             (mu_T, sd_T over the same reference draws)
    S_energy = max_t Z_t

Both scores are therefore maxima of a reference-standardized statistic over
the same five scales. The only difference is whether pairs are split by the
two events' energy channels. Equal weights mean a cell with a quiet
background (e.g. (1,2)) counts as much as (0,0), which is how energy enters.

Calibration (Linh): p = (1 + #{null S >= observed S}) / (B + 1), repeating the
full score (including the max over t) on every null draw. Use the same
reference for a sky's observed and null draws (automatic: same inventory).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pair_counter import N_CHANNELS, THRESHOLDS_DEG, pair_counts_by_channel

OBSERVABLE_FIELDS = ("reco_ra_deg", "reco_dec_deg", "reco_energy_channel")
IU = np.triu_indices(N_CHANNELS)          # 15 channel pairs, a <= b
CONFIG_PATH = Path(__file__).resolve().parent.parent / "frozen" / "score_config.json"

_REF_CACHE: dict = {}


# ----------------------------------------------------------------------------- config
def load_config(path: str | Path = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text())
    body = {k: v for k, v in cfg.items() if k != "config_sha256"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if cfg.get("config_sha256") != h:
        raise ValueError("score_config.json was modified after freezing (hash mismatch)")
    if tuple(cfg["thresholds_deg"]) != tuple(THRESHOLDS_DEG):
        raise ValueError("config thresholds differ from pair_counter thresholds")
    return cfg


# ----------------------------------------------------------------------------- inputs
def _observables(events):
    """Accepts a dict / npz-like with the three reco fields. Reads nothing else."""
    ra = np.asarray(events["reco_ra_deg"], float)
    dec = np.asarray(events["reco_dec_deg"], float)
    ch = np.asarray(events["reco_energy_channel"]).astype(np.int64)
    return ra, dec, ch


def inventory_key(dec, ch) -> int:
    """64-bit key of the RA-invariant inventory, independent of event order."""
    order = np.lexsort((np.asarray(dec, float), np.asarray(ch, np.int64)))
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(ch, np.int64)[order]).tobytes())
    h.update(np.ascontiguousarray(np.asarray(dec, np.float64)[order]).tobytes())
    return int.from_bytes(h.digest()[:8], "little")


# ----------------------------------------------------------------------------- reference
def reference_counts(dec, ch, cfg) -> np.ndarray:
    """(R, T, 5, 5) pair counts on R uniform-RA reference draws."""
    order = np.lexsort((dec, ch))
    dec_c, ch_c = dec[order], ch[order]
    ss = np.random.SeedSequence(entropy=cfg["reference"]["entropy"],
                                spawn_key=(inventory_key(dec, ch),))
    rng = np.random.default_rng(ss)
    R = cfg["reference"]["n_draws"]
    out = np.empty((R, len(THRESHOLDS_DEG), N_CHANNELS, N_CHANNELS), np.int64)
    for r in range(R):
        out[r] = pair_counts_by_channel(rng.uniform(0.0, 360.0, size=dec.size), dec_c, ch_c)
    return out


def _cells(C, dense):
    """C (..., T, 5, 5) -> list over t of (..., K_t) cell counts: dense cells + merged sparse."""
    flat = C[..., IU[0], IU[1]]                        # (..., T, 15)
    out = []
    for t in range(flat.shape[-2]):
        f = flat[..., t, :]
        d = f[..., dense[t]]
        s = f[..., ~dense[t]].sum(-1, keepdims=True)
        out.append(np.concatenate([d, s], -1))
    return out


def fit_reference(Cref, cfg) -> dict:
    """Fitted numbers from reference counts only."""
    e = cfg["energy"]
    pooled = Cref.sum(axis=(-2, -1)).astype(float)                       # (R, T)
    ref = {"pooled_mu": pooled.mean(0), "pooled_sd": pooled.std(0, ddof=1)}
    flat = Cref[..., IU[0], IU[1]].astype(float)                         # (R, T, 15)
    dense = flat.mean(0) >= e["min_ref_mean"]                            # (T, 15)
    ref["dense"] = dense
    cells = _cells(Cref.astype(float), dense)
    mu_k, den_k, keep, muT, sdT = [], [], [], [], []
    for t, c in enumerate(cells):
        mu = c.mean(0)
        sd = c.std(0, ddof=1)
        k = mu > 0                       # drop an all-zero sparse cell
        den = np.sqrt(sd ** 2 + e["var_floor"])
        T = ((c - mu) / den)[:, k].sum(-1)
        mu_k.append(mu); den_k.append(den); keep.append(k)
        muT.append(T.mean()); sdT.append(T.std(ddof=1))
    ref.update(cell_mu=mu_k, cell_den=den_k, cell_keep=keep,
               T_mu=np.array(muT), T_sd=np.array(sdT))
    for name in ("pooled_sd", "T_sd"):
        if np.any(ref[name] <= 0):
            raise ValueError(f"degenerate reference: {name} has zero entries")
    return ref


def get_reference(dec, ch, cfg) -> dict:
    key = (cfg["config_sha256"], inventory_key(dec, ch), dec.size)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = fit_reference(reference_counts(dec, ch, cfg), cfg)
    return _REF_CACHE[key]


# ----------------------------------------------------------------------------- scores
def per_scale(C, ref, method) -> np.ndarray:
    """Reference-standardized statistic per threshold, shape (T,)."""
    if method == "pooled":
        return (C.sum(axis=(-2, -1)) - ref["pooled_mu"]) / ref["pooled_sd"]
    if method == "energy":
        cells = _cells(C.astype(float), ref["dense"])
        T = np.array([((c - ref["cell_mu"][t]) / ref["cell_den"][t])[ref["cell_keep"][t]].sum()
                      for t, c in enumerate(cells)])
        return (T - ref["T_mu"]) / ref["T_sd"]
    raise ValueError("method must be 'pooled' or 'energy'")


def score(events, method: str, config: dict | None = None) -> float:
    """Frozen score of one sky. Larger = more clustered."""
    cfg = config or load_config()
    ra, dec, ch = _observables(events)
    C = pair_counts_by_channel(ra, dec, ch)
    return float(per_scale(C, get_reference(dec, ch, cfg), method).max())


def score_both(events, config: dict | None = None) -> dict:
    cfg = config or load_config()
    ra, dec, ch = _observables(events)
    C = pair_counts_by_channel(ra, dec, ch)
    ref = get_reference(dec, ch, cfg)
    return {m: float(per_scale(C, ref, m).max()) for m in ("pooled", "energy")}


def score_from_counts(C, dec, ch, method, config: dict | None = None) -> float:
    """Same score from precomputed C (T,5,5) on this sky's inventory (fast path for nulls)."""
    cfg = config or load_config()
    return float(per_scale(C, get_reference(np.asarray(dec, float),
                                            np.asarray(ch, np.int64), cfg), method).max())


def energy_contributions(C, ref) -> dict:
    """For the figure: per-channel-pair z (dense cells) and sparse-cell z, per threshold."""
    out = []
    for t in range(C.shape[0]):
        flat = C[t][IU].astype(float)
        dense = ref["dense"][t]
        z = np.full(15, np.nan)
        z[dense] = (flat[dense] - ref["cell_mu"][t][:-1]) / ref["cell_den"][t][:-1]
        sparse_z = ((flat[~dense].sum() - ref["cell_mu"][t][-1]) / ref["cell_den"][t][-1]
                    if ref["cell_keep"][t][-1] else np.nan)
        out.append({"z_dense": z, "z_sparse": sparse_z, "sparse_pairs": np.flatnonzero(~dense)})
    return out


def score_sky_and_draws(events, ra_draws, config: dict | None = None) -> dict:
    """
    Calibration helper for Linh. events: one sky's inventory; ra_draws: (B, N)
    null RA rows for the SAME event order. Returns observed scores and the full
    score recomputed (incl. max over thresholds) on every draw.
    """
    cfg = config or load_config()
    ra, dec, ch = _observables(events)
    ra_draws = np.asarray(ra_draws, float)
    if ra_draws.ndim != 2 or ra_draws.shape[1] != ra.size:
        raise ValueError("ra_draws must have shape (B, n_events) in the sky's event order")
    ref = get_reference(dec, ch, cfg)
    out = {"observed": {}, "null": {}}
    C_obs = pair_counts_by_channel(ra, dec, ch)
    C_null = [pair_counts_by_channel(r, dec, ch) for r in ra_draws]
    for m in ("pooled", "energy"):
        out["observed"][m] = float(per_scale(C_obs, ref, m).max())
        out["null"][m] = np.array([per_scale(C, ref, m).max() for C in C_null])
        out.setdefault("p", {})[m] = float((1 + (out["null"][m] >= out["observed"][m]).sum())
                                           / (len(C_null) + 1))
    return out
