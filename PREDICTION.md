# Prediction note: where energy information should help

Author: Aaron · Written: 2026-10-26, before any score is frozen or evaluated.
Status: DRAFT. Rewrite in your own words, then commit to git so the timestamp is fixed.

Basis: only the two exposed development skies (26392, 14265) and their 1000 uniform-RA nulls.
No evaluation or unexposed sky has been looked at.

## 1. Expected result

The energy-resolved score should detect source clustering at **lower `g_comp`** than the pooled score.
The paired gain in detection probability at 5% false alarms should be:

- **largest at intermediate `g_comp`**, where astrophysical events form many small multiplets
  (doublets to a few events per source);
- **near zero at high `g_comp`**, where one or a few bright sources dominate and both methods
  detect almost every sky (14265 is this case);
- **near zero at very low `g_comp`**, where there are almost no same-source pairs to find, so both
  methods stay at the 5% false-alarm rate (26392 is this case).

Sketched as detection probability against `g_comp`, both curves start at about 5% and end near 100%.
The energy-resolved curve should rise earlier.

## 2. Physical reasoning

- **Background pairs are almost all low-energy.** About 92% of atmospheric events are in channel 0,
  and channels 3–4 hold at most 2 events per sky. Chance pairs therefore sit overwhelmingly in (0,0).
  At 1°, the uniform-RA null gives about 1,320 (0,0) pairs, 220 (0,1), 9 (1,1), 1.2 (1,2),
  and 0.1 or fewer for most pairs involving channels 2–4.
- **Source events are harder.** In the two dev skies, 30–45% of astrophysical events lie above
  channel 0, against about 8% of atmospheric events (truth labels, used for reasoning only).
  Same-source pairs therefore land in channel pairs with small backgrounds much more often than
  chance pairs do.
- **One source pair matters more in a quiet bin.** At 0.5°, the pooled count has a null sd of about
  20 pairs. A detection at 5% needs an excess of roughly 30+ pairs actually within 0.5°. That is like
  one source with about 10–12 events, and no realistic number of doublets reaches it. In (1,1) the
  null is about 2 ± 1.4 pairs, and in (1,2) about 0.3. A few source pairs there are already a large
  deviation.
- **Rough expectation:** an energy-weighted score might need around half as many same-source pairs
  as the pooled score to reach the same detection probability. This is a back-of-envelope estimate
  (Gaussian thinking, small-number bins, guessed pair fractions within θ), not a calculation.
- **Small angles should matter most.** Astrophysical events are better localized than atmospheric
  ones (median true error about 0.4° vs. about 1.0° in these skies). Source pairs should therefore
  show up most at 0.5–1°, not at 5°. These skies show no clear improvement in resolution with
  reconstructed energy channel, so I do not predict a large extra gain from that.

## 3. Where energy may NOT help, or may hurt

- **Bright single source.** If one source supplies most astrophysical events, pooled counts already
  detect it, and energy adds nothing measurable.
- **Signal mostly in channel 0.** In 14265, 60 of 85 events from the main source are in channel 0.
  Where source events are not much harder than the background, splitting into 15 channel pairs
  spreads the signal and adds noise. A badly chosen weighting could make the energy-resolved score
  worse than pooled.
- **Empty and discrete bins.** High-channel bins have null means near zero, so score distributions
  are discrete. Actual false-alarm rates can differ from 5%. Any apparent gain must be checked against
  each method's real false-alarm rate.
- **Soft-spectrum populations.** If a population's astrophysical spectrum is close to atmospheric,
  the energy contrast shrinks, and so should the gain.

## 4. What would contradict this prediction

- A paired difference consistent with zero, with tight uncertainty, at intermediate `g_comp`.
- An energy-resolved score that is less sensitive than pooled at intermediate `g_comp` at matched
  actual false-alarm rates.
- A gain driven mainly by (0,0) pairs rather than by pairs involving channels 1–3, which would mean
  the score is not using energy the way this reasoning assumes.

## 5. Caveats on this note

- Built from two exposed skies at the extremes of `g_comp`. The intermediate regime has not been seen.
- Truth information (component labels, source IDs, true errors) informs this reasoning only. It never
  enters either score.
- Numbers are development diagnostics, not calibrated results.
