# Conditional RA Null Sampler Handoff

Date: 2026-10-21  
Branch: `analysis/scrambling-background`  
Commit: `bac169d` — `Add independent uniform RA conditional null sampler`

## Purpose

This package hands off Linh's validated conditional-RA null sampler to Aaron for integration with Aaron's frozen pooled and energy-resolved clustering scores.

The primary null treats every event in the mixed sky identically and independently redraws reconstructed right ascension as

    RA ~ Uniform(0, 360 deg)

while preserving the reconstructed event inventory and marks, including:

- reconstructed declination
- reconstructed energy
- reconstructed energy channel
- detector period
- angular uncertainty

The existing within-detector-period RA permutation remains a development diagnostic only.

## Frozen H1 configuration

The sampler was checked against the frozen H1/DR2 setup used for the development skies.

Key configuration:

- analysis declination minimum: 5 deg
- source ROI declination minimum: 5 deg
- source generation declination minimum: -5 deg
- atmospheric generation declination minimum: -5 deg
- HEALPix nside: 128
- response physics hash:
  `30475bb70f21eff54f9539d00a506bec71db9571df18d5906b0581d346c8d9c7`

The response and production paths were checked to have no RA-dependent analysis cut. The DR2 directional smearing uses a uniformly random tangent-plane orientation, supporting full-circle rotational symmetry in RA.

## Development validation

Two exposed H1 development skies were tested with 1000 conditional-null realizations each.

### Simulation 26392

- 3760 total events
- 66 astrophysical
- 3694 atmospheric
- 1000 uniform-RA draws
- status: PASS

### Simulation 14265

- 3760 total events
- 86 astrophysical
- 3674 atmospheric
- 1000 uniform-RA draws
- status: PASS

Validation checks included:

- exact preservation of frozen mixed-sky event marks
- analytic concentrated-RA test
- independent uniform-reference comparison
- conditional RA checks by detector period
- conditional RA checks by energy channel

A sparse channel-3 fluctuation in simulation 14265 was followed up with an independent RNG stream. The initial maximum 12-bin deviation was 4.40 sigma; the independent replication gave 1.59 sigma, consistent with a finite Monte Carlo fluctuation rather than persistent RA bias.

See `handoff_summary.json` for the machine-readable validation record.

## Files

`code/run_ra_scrambling.py`
: Conditional-null sampler implementation.

`code/run_ra_scrambling.slurm`
: Slurm wrapper supporting both the diagnostic permutation method and the independent uniform-RA method.

`validation/summary_26392.json`
: Metadata from the 1000-draw uniform-RA run for simulation 26392.

`validation/summary_14265.json`
: Metadata from the 1000-draw uniform-RA run for simulation 14265.

`handoff_summary.json`
: Machine-readable handoff and validation summary.

`git_commit.txt`
: Git provenance for the frozen implementation.

## Aaron handoff

Aaron owns:

- pair counting
- pooled score construction
- energy-resolved score construction
- physical interpretation

The requested return handoff is the frozen pooled and energy-resolved score callable/settings and the minimal event/result interface needed for Linh to run calibration.

Linh will apply the same conditional-null and calibration procedure to both frozen scores.

## Current limitation

Final 5% false-alarm calibration has not yet been established.

That stage requires genuinely independent smooth/null control skies. The historical H0 evidence available locally contains JSON reports, but the referenced historical bank is not accessible from Linh's account.

The sampler itself is considered validated; the independent-control-sky issue remains a calibration-stage dependency.
