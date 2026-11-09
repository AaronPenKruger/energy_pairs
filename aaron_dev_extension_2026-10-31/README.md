# H1 Development Sky Extension — Aaron Handoff

Date: 2026-10-31
Latest sampler commit: 81e7082

## Purpose

This package extends the two previously shared development skies
(26392 and 14265) with three intermediate-g_comp development skies.

New development skies:

Simulation   g_comp     M1       Events    RA Seed
55289        0.253819   86.99    3592      20316292
192235       0.501072   63.13    4324      20453238
3332         0.742569   146.09   6186      20264335

Each sky has 1000 independent uniform-RA null realizations.
All mixed-sky events are randomized, preserving non-RA event marks.

## Files

replay_<ID>/
  astro_events.npz
  atmo_events.npz
  summary.json

uniform_<ID>/
  event_inventory.npz
  scrambled_reco_ra_deg.npy
  summary.json

splits/
  h1_dev_eval_split_2026-10-31.json
  h1_dev_ra_null_seeds_2026-10-31.json

## Development/Evaluation Rules

Approved development IDs:
26392, 14265, 55289, 192235, 3332

The remaining 251 H1 bank skies are reserved for evaluation.
Their previous-exposure audit remains pending.

Do not tune scores on reserved evaluation skies.

The three new skies were selected using frozen g_comp metadata,
not observed pair-count or clustering outcomes.

They are not matched in expected astrophysical signal M1 or
atmospheric event counts. Differences across these skies must
not be attributed solely to g_comp.

## Validation

All three event-level replays exactly reproduced their frozen
H1 maps.

All three uniform-RA null runs passed internal validation:
- 1000 null realizations per sky
- every event scrambled
- zero fixed events
- non-RA event inventory preserved
- valid reconstructed RA values
- distinct recorded seeds

The original development nulls were not overwritten.

## Scientific Boundaries

Aaron owns pair counting and pooled/energy-resolved score construction.

Linh owns the conditional null sampler, split record, calibration,
and joint evaluation.

Source truth labels are available for diagnostics but should not
enter the primary label-free scores.

These development runs are not final false-alarm calibration
or independent evaluation results.
