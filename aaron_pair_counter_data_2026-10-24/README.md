# Aaron pair-counter data handoff

Development skies:
- 26392
- 14265

## replay_*/

Contains event-level replay products.

Files:
- astro_events.npz
- atmo_events.npz
- summary.json

The astrophysical event files include `source_index`, which can be used later for truth-level one-source vs. many-sources checks. Atmospheric events do not have source IDs.

## uniform_*/

Contains the primary label-free conditional-null products.

Files:
- event_inventory.npz
- scrambled_reco_ra_deg.npy
- summary.json

The `scrambled_reco_ra_deg.npy` arrays contain 1000 independent full-circle uniform-RA null realizations for the corresponding mixed sky.

The null preserves the non-RA event marks in `event_inventory.npz`, including reconstructed declination, reconstructed energy/channel, detector period, and angular uncertainty.

These two skies are exposed development material. Do not treat them as independent evaluation skies.
