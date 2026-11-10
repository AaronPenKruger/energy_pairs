# Handoff: frozen clustering scores v1 (Aaron -> Linh), 2026-11-09

Development only. No calibration, held-out scoring, sensitivity curve or energy-advantage claim.
Only the five development skies (26392, 14265, 55289, 192235, 3332) and labeled toy skies were used.

## 1. Numerical cross-check (brief step 1): PASS
`python code/step1_handoff_check.py` -> `results/step1_handoff_check.json`
- Nulls regenerated from Linh's recorded seeds (`default_rng(seed)`, one `uniform(0,360,N)` per row) reproduce
  every saved count: observed + 1000/1000 nulls, all five skies, exact.
- Sum of the 15 unordered channel pairs == pooled count, every threshold, observed and every null; a>b entries zero.
- Independent direct calculation (vectorized haversine over all i<j, no KD-tree) == counter on the observed sky
  and on saved draw 0, all five skies, exact.
- Earlier: exact agreement with the historical `dfe378f` counter on 26392/14265 (2,002 skies). For the three new skies,
  run on Polaris (needs the scrambled .npy and the old script):
  `python code/crosscheck_old_counter.py --old <path>/analyze_ra_pair_counts.py --data aaron_dev_extension_2026-10-31 --sims 55289 192235 3332`

## 2. The two scores
Both read only `reco_ra_deg, reco_dec_deg, reco_energy_channel`. Thresholds t in {0.5, 1, 2, 3, 5} deg.

**Reference ensemble.** R = 1000 draws per sky, each event RA ~ U[0,360), Dec and channel fixed (same law as the null).
Stream: `SeedSequence(entropy=20261109, spawn_key=(inventory_key,))`, events in (channel, Dec) order; separate from
`default_rng(<int>)` null/calibration streams. Because only RA changes, a sky and all its randomized versions share the
same reference: original and randomized inventories are scored identically, and event order does not matter.

**Pooled:** z_t = (N_t - mu_t)/sd_t;  S_pool = max_t z_t

**Energy-resolved:** at each t, channel pairs with reference mean >= 5 are separate cells; the rest are summed into one
sparse cell (dropped if its reference mean is 0).
z_kt = (C_kt - mu_kt)/sqrt(sd_kt^2 + 1);  T_t = sum_k z_kt;  Z_t = (T_t - mu_T,t)/sd_T,t;  S_E = max_t Z_t

Same scales, same max, same kind of standardization; the only difference is splitting pairs by the two events' channels.

## 3. What the energy score emphasizes, and why
At small angles the uniform-RA background sits almost entirely in (0,0) and (0,1); every other channel pair has fewer
than 5 expected chance pairs and falls into one merged "sparse" cell. With equal weights, that cell counts as much as
(0,0), so a handful of close pairs in which one event is in channel 2 or higher, or both events are above channel 0, moves the
score as much as dozens of extra low-energy pairs. The score therefore emphasizes spatially close associations among
the harder events, where atmospheric chance pairs are rare, and de-emphasizes soft multiplets whose pairs land in (0,0).
On the dev skies this shows directly: the single 85-event source in 14265 puts many pairs into the sparse cell (z = 75)
and the energy score exceeds pooled, while the bright but soft sources in 3332 put their excess into (0,0) and (0,2),
which equal weighting dilutes, so pooled is higher. The price is that when the signal is soft, the energy score spends
weight on cells that do not respond. That is the trade-off Linh's independent evaluation should measure.

## 4. Development diagnostics (`results/dev_scores.json`, NOT calibrated)
| sky | g_comp | M1 expected astro | events (astro / atmo, realized) | S_pool (dev p) | S_E (dev p) |
|---|---|---|---|---|---|
| 26392 | 0.03 | 85.5 | 3760 (66 / 3694) | 0.78 (0.564) | 0.18 (0.829) |
| 55289 | 0.25 | 87.0 | 3592 (83 / 3509) | 0.81 (0.526) | 0.51 (0.680) |
| 192235 | 0.50 | 63.1 | 4324 (47 / 4277) | -0.20 (0.915) | -0.60 (0.975) |
| 3332 | 0.74 | 146.1 | 6186 (169 / 6017) | 11.42 (0.001) | 7.68 (0.001) |
| 14265 | 0.99 | 87.4 | 3760 (86 / 3674) | 67.48 (0.001) | 97.32 (0.001) |
Dev nulls are development draws (26392/14265 share a seed). Null score distributions look alike across skies
(mean 0.90-0.97, sd 0.82-0.90). No ties with the observed energy score. The dev skies are unmatched (astro counts and
backgrounds differ); no sensitivity statement follows from them. M1 (expected astro) is from the Nov 1 record
(`development_sky_results`), saved in `results/expected_astro_M1.json`; annotation only, never read by a score.

## 5. Figure
`figures/channel_contributions_dev.png` (+ `.json` post-hoc truth notes), made with
`python code/make_contribution_figure.py --expected-astro results/expected_astro_M1.json`. `source_index` used only after scoring.

## 6. Frozen artifacts
- `frozen/score_config.json` sha256 a96140b7866b07cc... (self-hash `config_sha256` 6e7dc3ab4281...)
- `frozen/reference_<id>.npz`: fitted mu, sd, dense mask, T_mu, T_sd per dev sky (and toy skies)
- `code/clustering_scores.py`, `code/pair_counter.py` at the commit carrying this file
- Seeds: reference entropy 20261109; dev nulls per `splits/h1_dev_ra_null_seeds_2026-10-31.json`

## 7. How Linh calls it
```python
import sys; sys.path.insert(0, "code")
import clustering_scores as cs
cfg = cs.load_config()                         # refuses a modified config
out = cs.score_sky_and_draws(events, ra_draws, cfg)   # ra_draws (B, N), same event order
out["observed"]["energy"], out["null"]["energy"], out["p"]["energy"]
```
p = (1 + #{null >= obs})/(B + 1), full score (incl. max over t) on every draw. Reference costs ~6-15 s per sky
once (cached per inventory); each extra draw ~8-18 ms.
**Joint check:** pick one saved draw, both run `cs.score(dict(events, reco_ra_deg=draw), m, cfg)` and compare to all digits.

## 8. Tests
`python -m pytest -q code/` -> 24 passed (11 counter + 8 score + 5 synthetic/toy). Score tests cover config tamper guard,
truth columns ignored, event-order invariance, original vs randomized inventory sharing the reference, reference
reproducing frozen numbers, reference stream differing from null stream, calibration helper == score().

## Addendum (2026-11-09, Polaris)
- Historical-counter cross-check for 55289, 192235, 3332 (crosscheck_old_counter.py vs
  scrambling_background analysis/scrambling-background scripts/analyze_ra_pair_counts.py):
  1001/1001 identical per sky (observed + 1000 nulls), all 15 channel pairs and pooled. PASS.
  With the earlier 26392/14265 check: all five dev skies agree exactly with the old counter.
- On Polaris, step1_handoff_check.py check A (regenerated nulls == saved scrambled_reco_ra_deg.npy): True for all five skies.
- pytest on Polaris (icecube_clean): 24 passed.

## Addendum (2026-11-09, Polaris)
- Historical-counter cross-check for 55289, 192235, 3332 (crosscheck_old_counter.py vs
  scrambling_background analysis/scrambling-background scripts/analyze_ra_pair_counts.py):
  1001/1001 identical per sky (observed + 1000 nulls), all 15 channel pairs and pooled. PASS.
  With the earlier 26392/14265 check: all five dev skies agree exactly with the old counter.
- On Polaris, step1_handoff_check.py check A (regenerated nulls == saved scrambled_reco_ra_deg.npy): True for all five skies.
- pytest on Polaris (icecube_clean): 24 passed.
