"""
TOY synthetic skies for developing and sanity-checking pair-count scores.

NOT the IceCube simulator and NOT a production simulation. Every number here is
a rough, hard-coded approximation taken from the two exposed development
skies (26392, 14265). Use these skies only to check that code and score ideas
behave sensibly. Never use them for sensitivity claims.

A toy sky has N_total events (default 3760):
  - n_astro astrophysical events. A fraction `f_multiplet` of them come from
    multiplet sources of `multiplet_size` events each; the rest are singlets.
    `f_multiplet` is a toy analog of g_comp (truth descriptor only).
  - the remaining atmospheric events, scattered with no source structure.

Positions:
  - atmospheric events: RA uniform, Dec drawn from a piecewise-constant
    density in sin(Dec) matching the dev skies (more events near the horizon).
  - sources: RA uniform, Dec from the same sin(Dec) density (toy choice).
  - astrophysical events: source position smeared by a 2D Gaussian PSF with
    per-axis sigma ASTRO_PSF_SIGMA_DEG (median error ~0.4 deg as in dev skies).
    Smeared events falling below DEC_MIN_DEG are redrawn.
Energy channels are drawn independently per event from component-specific
fractions (astro harder than atmo).

Output, in Linh's format so all downstream code runs unchanged:
  <out>/event_inventory.npz : period, reco_log10_energy, reco_energy_channel,
                              reco_ra_deg, reco_dec_deg, angular_uncertainty_deg
  <out>/truth.npz           : component, source_index (-1 for atmo), plus the
                              true source positions. KEPT SEPARATE so a score
                              can never read truth by accident.
  <out>/summary.json        : all generator settings and the seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEC_MIN_DEG = 5.0

# Relative event counts in 9 equal-width bins of sin(Dec) over [sin 5 deg, 1],
# from the two dev skies combined (all events). Approximate.
SIN_DEC_WEIGHTS = (1863, 1324, 1101, 931, 642, 537, 485, 357, 280)

# Energy-channel fractions, approximate, from the dev skies (truth used only to
# set toy parameters).
ATMO_CHANNEL_FRAC = (0.925, 0.071, 0.0035, 0.0005, 0.0)
ASTRO_CHANNEL_FRAC = (0.63, 0.28, 0.072, 0.013, 0.005)

ASTRO_PSF_SIGMA_DEG = 0.34
CHANNEL_EDGES_LOG10_GEV = np.log10(
    [1.0e4, 3.981e4, 1.585e5, 6.310e5, 2.512e6, 1.0e7]
)


@dataclass
class ToySkyConfig:
    seed: int
    n_total: int = 3760
    n_astro: int = 80
    f_multiplet: float = 0.0          # fraction of astro events in multiplets
    multiplet_size: int = 3           # events per multiplet source
    astro_psf_sigma_deg: float = ASTRO_PSF_SIGMA_DEG
    atmo_channel_frac: tuple = field(default=ATMO_CHANNEL_FRAC)
    astro_channel_frac: tuple = field(default=ASTRO_CHANNEL_FRAC)


def _draw_dec(n, rng):
    w = np.asarray(SIN_DEC_WEIGHTS, float)
    edges = np.linspace(np.sin(np.radians(DEC_MIN_DEG)), 1.0, w.size + 1)
    b = rng.choice(w.size, size=n, p=w / w.sum())
    s = rng.uniform(edges[b], edges[b + 1])
    return np.degrees(np.arcsin(s))


def _smear(ra0, dec0, sigma_deg, rng):
    """Offset each (ra0, dec0) by a 2D Gaussian error on the sphere."""
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    s = np.column_stack([np.cos(dec0) * np.cos(ra0),
                         np.cos(dec0) * np.sin(ra0),
                         np.sin(dec0)])
    e_ra = np.column_stack([-np.sin(ra0), np.cos(ra0), np.zeros_like(ra0)])
    e_dec = np.cross(s, e_ra)
    dx, dy = np.radians(sigma_deg) * rng.standard_normal((2, ra0.size))
    r = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(r[:, None] > 0, (dx[:, None] * e_ra + dy[:, None] * e_dec) / r[:, None], 0.0)
    v = np.cos(r)[:, None] * s + np.sin(r)[:, None] * u
    dec = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    ra = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    return ra, dec


def _channels(n, frac, rng):
    p = np.asarray(frac, float)
    return rng.choice(p.size, size=n, p=p / p.sum()).astype(np.int64)


def _log10e_in_channel(ch, rng):
    lo = CHANNEL_EDGES_LOG10_GEV[ch]
    hi = CHANNEL_EDGES_LOG10_GEV[ch + 1]
    return rng.uniform(lo, hi)


def make_toy_sky(cfg: ToySkyConfig):
    """Returns (observables dict, truth dict)."""
    if not 0 <= cfg.f_multiplet <= 1:
        raise ValueError("f_multiplet must be in [0, 1]")
    if cfg.multiplet_size < 2:
        raise ValueError("multiplet_size must be >= 2")
    if cfg.n_astro > cfg.n_total:
        raise ValueError("n_astro > n_total")
    rng = np.random.default_rng(cfg.seed)

    n_mult_src = int(round(cfg.f_multiplet * cfg.n_astro / cfg.multiplet_size))
    n_in_mult = n_mult_src * cfg.multiplet_size
    if n_in_mult > cfg.n_astro:
        n_mult_src -= 1
        n_in_mult = n_mult_src * cfg.multiplet_size
    n_singlets = cfg.n_astro - n_in_mult
    n_sources = n_mult_src + n_singlets

    src_ra = rng.uniform(0, 360, n_sources)
    src_dec = _draw_dec(n_sources, rng)
    src_of_event = np.concatenate([
        np.repeat(np.arange(n_mult_src), cfg.multiplet_size),
        n_mult_src + np.arange(n_singlets),
    ]).astype(np.int64)

    a_ra, a_dec = _smear(src_ra[src_of_event], src_dec[src_of_event],
                         cfg.astro_psf_sigma_deg, rng)
    bad = a_dec < DEC_MIN_DEG
    while bad.any():  # redraw smeared events that left the footprint
        a_ra[bad], a_dec[bad] = _smear(src_ra[src_of_event[bad]], src_dec[src_of_event[bad]],
                                       cfg.astro_psf_sigma_deg, rng)
        bad = a_dec < DEC_MIN_DEG

    n_atmo = cfg.n_total - cfg.n_astro
    t_ra = rng.uniform(0, 360, n_atmo)
    t_dec = _draw_dec(n_atmo, rng)

    ch = np.concatenate([_channels(cfg.n_astro, cfg.astro_channel_frac, rng),
                         _channels(n_atmo, cfg.atmo_channel_frac, rng)])
    ra = np.concatenate([a_ra, t_ra])
    dec = np.concatenate([a_dec, t_dec])

    # shuffle so event order carries no component information
    p = rng.permutation(cfg.n_total)
    component = np.array(["astro"] * cfg.n_astro + ["atmo"] * n_atmo, dtype="U16")
    source_index = np.concatenate([src_of_event, np.full(n_atmo, -1, np.int64)])

    obs = {
        "period": np.full(cfg.n_total, "TOY", dtype="U4"),
        "reco_log10_energy": _log10e_in_channel(ch, rng)[p],
        "reco_energy_channel": ch[p],
        "reco_ra_deg": ra[p],
        "reco_dec_deg": dec[p],
        "angular_uncertainty_deg": np.full(cfg.n_total, cfg.astro_psf_sigma_deg)[p],
    }
    truth = {
        "component": component[p],
        "source_index": source_index[p],
        "source_ra_deg": src_ra,
        "source_dec_deg": src_dec,
        "source_is_multiplet": np.arange(n_sources) < n_mult_src,
    }
    return obs, truth


def write_toy_sky(cfg: ToySkyConfig, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs, truth = make_toy_sky(cfg)
    np.savez_compressed(out_dir / "event_inventory.npz", **obs)
    np.savez_compressed(out_dir / "truth.npz", **truth)
    meta = asdict(cfg)
    meta.update({
        "kind": "TOY synthetic sky (not the IceCube simulator)",
        "n_multiplet_sources": int(truth["source_is_multiplet"].sum()),
        "n_astro_in_multiplets": int(np.isin(truth["source_index"],
                                             np.flatnonzero(truth["source_is_multiplet"])).sum()),
        "dec_min_deg": DEC_MIN_DEG,
        "sin_dec_weights": list(SIN_DEC_WEIGHTS),
    })
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    return obs, truth
