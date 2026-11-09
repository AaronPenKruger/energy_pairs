#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_events(path: Path, component: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = [
            "period",
            "reco_log10_energy",
            "reco_energy_channel",
            "reco_ra_deg",
            "reco_dec_deg",
            "angular_uncertainty_deg",
        ]

        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")

        out = {key: np.asarray(data[key]) for key in required}

    n = len(out["reco_ra_deg"])

    for key, value in out.items():
        if len(value) != n:
            raise ValueError(
                f"{path}: field {key!r} has length {len(value)}, expected {n}"
            )

    out["component"] = np.full(n, component, dtype="U16")

    return out


def concatenate_events(
    astro: dict[str, np.ndarray],
    atmo: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    keys = astro.keys()

    return {
        key: np.concatenate([astro[key], atmo[key]])
        for key in keys
    }

def draw_uniform_ra(
    original_ra: np.ndarray,
    scramble_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Independently draw reconstructed RA from the frozen full-circle
    conditional acceptance for events selected by scramble_mask.

    Events outside scramble_mask remain completely unchanged.
    """
    randomized = np.asarray(original_ra, dtype=float).copy()
    scramble_mask = np.asarray(scramble_mask, dtype=bool)

    if scramble_mask.shape != randomized.shape:
        raise ValueError("scramble_mask shape does not match RA array")

    n_randomized = int(np.count_nonzero(scramble_mask))

    randomized[scramble_mask] = rng.uniform(
        0.0,
        360.0,
        size=n_randomized,
    )

    return randomized


def scramble_ra_within_period(
    original_ra: np.ndarray,
    periods: np.ndarray,
    scramble_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute reconstructed RA within each IceCube detector period for only
    the events selected by scramble_mask.

    Events outside scramble_mask remain completely unchanged.
    """
    scrambled = np.asarray(original_ra, dtype=float).copy()
    scramble_mask = np.asarray(scramble_mask, dtype=bool)

    if scramble_mask.shape != scrambled.shape:
        raise ValueError("scramble_mask shape does not match RA array")

    for period in np.unique(periods):
        idx = np.flatnonzero(
            (periods == period) & scramble_mask
        )

        if idx.size <= 1:
            continue

        scrambled[idx] = original_ra[idx][
            rng.permutation(idx.size)
        ]

    return scrambled


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replay-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--n-scrambles",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261003,
    )

    parser.add_argument(
        "--global-sim-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "combined",
            "astro-only",
            "astro-plus-fixed-atmo",
        ],
        default="combined",
    )

    parser.add_argument(
        "--method",
        choices=[
            "permute-within-period",
            "uniform-full-ra",
        ],
        default="permute-within-period",
    )

    args = parser.parse_args()

    if args.n_scrambles <= 0:
        raise ValueError("--n-scrambles must be positive")

    replay_dir = args.replay_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {out_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    astro = load_events(
        replay_dir / "astro_events.npz",
        "astro",
    )

    atmo = load_events(
        replay_dir / "atmo_events.npz",
        "atmo",
    )

    if args.mode == "combined":
        events = concatenate_events(astro, atmo)
        scramble_mask = np.ones(
            len(events["reco_ra_deg"]),
            dtype=bool,
        )

    elif args.mode == "astro-only":
        events = astro
        scramble_mask = np.ones(
            len(events["reco_ra_deg"]),
            dtype=bool,
        )

    elif args.mode == "astro-plus-fixed-atmo":
        events = concatenate_events(astro, atmo)

        scramble_mask = (
            np.asarray(events["component"]) == "astro"
        )

    else:
        raise RuntimeError(
            f"Unhandled scrambling mode: {args.mode}"
        )

    ra = np.asarray(events["reco_ra_deg"], dtype=float)
    dec = np.asarray(events["reco_dec_deg"], dtype=float)
    periods = np.asarray(events["period"]).astype("U")
    channels = np.asarray(events["reco_energy_channel"], dtype=int)
    loge = np.asarray(events["reco_log10_energy"], dtype=float)
    angerr = np.asarray(events["angular_uncertainty_deg"], dtype=float)

    n_events = len(ra)

    if n_events == 0:
        raise RuntimeError("No events available for scrambling")

    if np.any(~np.isfinite(ra)):
        raise ValueError("Non-finite reconstructed RA values found")
    if np.any(~np.isfinite(dec)):
        raise ValueError("Non-finite reconstructed declinations found")
    if np.any((ra < 0.0) | (ra >= 360.0)):
        raise ValueError("Reconstructed RA outside [0, 360) degrees")

    rng = np.random.default_rng(args.seed)

    scrambled_ra = np.empty(
        (args.n_scrambles, n_events),
        dtype=np.float64,
    )

    print(f"events:       {n_events}", flush=True)
    print(f"astro:        {len(astro['reco_ra_deg'])}", flush=True)
    print(f"atmosphere:   {len(atmo['reco_ra_deg'])}", flush=True)
    print(f"scrambles:    {args.n_scrambles}", flush=True)
    print(f"seed:         {args.seed}", flush=True)

    unique_periods, period_counts = np.unique(periods, return_counts=True)

    print("period counts:", flush=True)
    for period, count in zip(unique_periods, period_counts):
        print(f"  {period}: {count}", flush=True)

    print(f"mode:         {args.mode}", flush=True)
    print(f"method:       {args.method}", flush=True)
    print(
        f"scrambled:    {np.count_nonzero(scramble_mask)}",
        flush=True,
    )
    print(
        f"fixed:        {np.count_nonzero(~scramble_mask)}",
        flush=True,
    )

    for i in range(args.n_scrambles):
        if args.method == "permute-within-period":
            scrambled_ra[i] = scramble_ra_within_period(
                ra,
                periods,
                scramble_mask,
                rng,
            )

        elif args.method == "uniform-full-ra":
            scrambled_ra[i] = draw_uniform_ra(
                ra,
                scramble_mask,
                rng,
            )

        else:
            raise RuntimeError(
                f"Unhandled scrambling method: {args.method}"
            )

    # Strong validation:
    # 1. Permutation method preserves the scrambled RA multiset
    #    within each detector period.
    # 2. Uniform-RA method produces finite RA values in [0, 360).
    # 3. Events outside scramble_mask remain exactly unchanged.

    scrambled_ra_multiset_preserved = True
    randomized_ra_valid = True
    fixed_ra_unchanged = True

    for i in range(args.n_scrambles):

        if args.method == "permute-within-period":
            for period in unique_periods:
                scrambled_idx = np.flatnonzero(
                    (periods == period) & scramble_mask
                )

                if scrambled_idx.size > 0:
                    if not np.array_equal(
                        np.sort(scrambled_ra[i, scrambled_idx]),
                        np.sort(ra[scrambled_idx]),
                    ):
                        scrambled_ra_multiset_preserved = False
                        break

        elif args.method == "uniform-full-ra":
            randomized_values = scrambled_ra[i, scramble_mask]

            if np.any(~np.isfinite(randomized_values)):
                randomized_ra_valid = False

            if np.any(
                (randomized_values < 0.0)
                | (randomized_values >= 360.0)
            ):
                randomized_ra_valid = False

        fixed_idx = np.flatnonzero(~scramble_mask)

        if fixed_idx.size > 0:
            if not np.array_equal(
                scrambled_ra[i, fixed_idx],
                ra[fixed_idx],
            ):
                fixed_ra_unchanged = False

        if (
            not scrambled_ra_multiset_preserved
            or not randomized_ra_valid
            or not fixed_ra_unchanged
        ):
            break

    if (
        args.method == "permute-within-period"
        and not scrambled_ra_multiset_preserved
    ):
        raise RuntimeError(
            "Detector-period RA multiset was not preserved "
            "for the scrambled component"
        )

    if (
        args.method == "uniform-full-ra"
        and not randomized_ra_valid
    ):
        raise RuntimeError(
            "Uniform-RA randomization produced invalid RA values"
        )

    if not fixed_ra_unchanged:
        raise RuntimeError(
            "Fixed-event RA values changed during scrambling"
        )

    np.save(
        out_dir / "scrambled_reco_ra_deg.npy",
        scrambled_ra,
        allow_pickle=False,
    )

    # Save the unchanged event inventory once.
    np.savez_compressed(
        out_dir / "event_inventory.npz",
        period=periods,
        reco_log10_energy=loge,
        reco_energy_channel=channels,
        reco_ra_deg=ra,
        reco_dec_deg=dec,
        angular_uncertainty_deg=angerr,
        component=np.asarray(events["component"]).astype("U"),
    )

    summary = {
        "global_sim_id": int(args.global_sim_id),
        "method": str(args.method),
        "seed": int(args.seed),
        "n_scrambles": int(args.n_scrambles),
        "n_events": int(n_events),
        "n_astro_events": int(len(astro["reco_ra_deg"])),
        "n_atmo_events": int(len(atmo["reco_ra_deg"])),
        "period_counts": {
            str(period): int(count)
            for period, count in zip(unique_periods, period_counts)
        },
        "non_ra_event_inventory_preserved": True,
        "detector_period_ra_multiset_preserved": (
            bool(scrambled_ra_multiset_preserved)
            if args.method == "permute-within-period"
            else None
        ),
        "uniform_ra_values_valid": (
            bool(randomized_ra_valid)
            if args.method == "uniform-full-ra"
            else None
        ),
        "fixed_ra_unchanged": bool(
            fixed_ra_unchanged
        ),
        "analysis_mode": args.mode,
        "n_scrambled_events":
            int(
                np.count_nonzero(scramble_mask)
            ),
        "n_fixed_events":
            int(
                np.count_nonzero(~scramble_mask)
            ),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print()
    print("PASS", flush=True)

    if args.method == "permute-within-period":
        print(
            "RA multiset preserved within each detector period.",
            flush=True,
        )

    elif args.method == "uniform-full-ra":
        print(
            "Independent full-circle uniform RA draws are valid.",
            flush=True,
        )

    if np.count_nonzero(~scramble_mask) > 0:
        print(
            "Fixed-event RA values remained exactly unchanged.",
            flush=True,
        )
    else:
        print(
            "No fixed events; all events were randomized.",
            flush=True,
        )
    print(
        "Declination, energy, channel, detector period, "
        "and angular uncertainty unchanged.",
        flush=True,
    )
    print(f"Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()