# Energy-resolved pair counter (Aaron), 2026-10-26

Step 2 of the energy-information study: one shared pair counter `C_ab(theta)`.
No scores yet. Nothing here uses truth labels.

## Files

- `code/pair_counter.py`: the counter. `pair_counts_by_channel(ra, dec, channel)` returns
  `C[t, a, b]` with shape (5 thresholds, 5, 5), filled only for a <= b.
  The pooled count is `C.sum(axis=(1, 2))`.
- `code/test_pair_counter.py`: 11 checks, including direct brute-force agreement,
  no self-pairs or double counts, degrees not radians, RA wraparound, poles,
  channel sum = pooled, event-order invariance, pooled unchanged by energy shuffles.
- `code/run_pair_counts.py`: runs the counter on one sky and all of Linh's uniform-RA nulls.
- `code/run_pair_counts.slurm`: SFSU Polaris wrapper (runs tests, then both skies).
- `results/counts_<sim>.npz`: `C_obs` (5,5,5) and `C_null` (1000,5,5,5), plus thresholds and channel pairs.
- `results/counts_<sim>.json`: provenance (input SHA-256s, null seed, fields read, runtime).

- `code/synthetic_sky.py`: TOY sky generator, not the IceCube simulator. It writes Linh's
  `event_inventory.npz` format and keeps truth in a separate `truth.npz`. `f_multiplet` is a toy
  analog of `g_comp`.
- `code/test_synthetic_sky.py`: 5 checks (reproducible, format with no truth columns, multiplet
  bookkeeping, PSF scale, astro harder than atmo).
- `code/run_toy_demo.py`: the runnable synthetic example. It generates toy skies, makes nulls with
  Linh's own `draw_uniform_ra`, and runs the unchanged `run_pair_counts.py`.
- `PREDICTION.md`: prediction note, written before any score design.

## Conventions

- Thresholds 0.5, 1, 2, 3, 5 deg. A pair counts if its separation is <= theta.
- Positions are converted to 3D unit vectors and searched with a KD-tree, so RA wraparound
  and poles need no special handling.
- Reads only `reco_ra_deg`, `reco_dec_deg`, `reco_energy_channel`.
  It never reads `component`, `source_index`, `true_*`, or `psf_deg`.

## Run on Polaris (from ~/energy_pairs)

    conda activate icecube_clean
    python code/test_pair_counter.py
    mkdir -p logs && sbatch code/run_pair_counts.slurm

About 8 s per sky with 1000 nulls on one CPU.

Toy example (about 10 s):

    python code/test_synthetic_sky.py
    python code/run_toy_demo.py --out toy --n-null 200

The toy's null pair density is within about 3% of the real dev skies at every threshold.

## Status

- Not yet compared against the old `dfe378f` pair-count code. Do this when Linh's Git bundle arrives.
- The pooled-count table printed by `run_pair_counts.py` is a development diagnostic on exposed skies.
  It is not a calibrated result.
