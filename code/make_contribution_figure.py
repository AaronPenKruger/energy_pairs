#!/usr/bin/env python3
"""
One figure: how energy-channel pairs contribute to the energy-resolved score on
the SAME sky, for the five development skies, next to both scores' per-scale
statistics.

Left  : z_ab at the threshold the energy score picked (upper triangle, a<=b).
        Hatched cells are merged into the sparse cell; its z is printed in the panel.
Right : per-scale statistic Z_t for pooled and energy (score = max over t).
Text  : event inventory, astro counts, scores and development p.

source_index / component are read ONLY here, after scoring, to annotate which
same-source pairs make the excess. They never enter either score.

Usage: python code/make_contribution_figure.py --out figures/channel_contributions_dev.png \
          [--expected-astro expected_astro.json]   # {"26392": 63.2, ...} if available
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustering_scores as cs
from pair_counter import candidate_pairs, pair_counts_by_channel
from step1_handoff_check import SKY_DIRS

BLUE, MID, RED = "#2a78d6", "#f0efec", "#e34948"
INK, MUTED = "#26251f", "#6b6a63"
CMAP = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
THR = (0.5, 1.0, 2.0, 3.0, 5.0)
ORDER = ["26392", "55289", "192235", "3332", "14265"]   # by g_comp


def posthoc_same_source(sid, d, theta):
    """Truth-only annotation: same-source astro pairs within theta, by channel cell."""
    rep = Path(str(d).replace("uniform_", "replay_"))
    with np.load(rep / "astro_events.npz") as a:
        src = np.asarray(a["source_index"]); ra_a = np.asarray(a["reco_ra_deg"])
    with np.load(d / "event_inventory.npz") as inv:
        ra, dec, ch = inv["reco_ra_deg"], inv["reco_dec_deg"], inv["reco_energy_channel"]
        comp = np.asarray(inv["component"])
    n_a = src.size
    assert np.array_equal(ra[:n_a], ra_a) and np.all(comp[:n_a] == "astro")
    lab = np.full(ra.size, -1); lab[:n_a] = src
    pairs, sep = candidate_pairs(ra, dec, theta)
    m = (sep <= theta) & (lab[pairs[:, 0]] >= 0) & (lab[pairs[:, 0]] == lab[pairs[:, 1]])
    p = pairs[m]
    hi = np.maximum(ch[p[:, 0]], ch[p[:, 1]]) > 0
    vals, cnt = np.unique(src[src >= 0], return_counts=True) if n_a else ([], [])
    top = int(cnt.max()) if n_a else 0
    top_pairs = int(np.sum(lab[p[:, 0]] == vals[cnt.argmax()])) if p.size else 0
    return {"n_astro": int(n_a), "same_src_pairs": int(p.shape[0]),
            "same_src_pairs_above_00": int(hi.sum()), "largest_source_events": top,
            "largest_source_pair_share": (top_pairs / p.shape[0]) if p.shape[0] else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures/channel_contributions_dev.png"))
    ap.add_argument("--expected-astro", type=Path, default=None)
    a = ap.parse_args()
    cfg = cs.load_config()
    gcomp = json.loads(Path("aaron_dev_extension_2026-10-31/splits/h1_dev_eval_split_2026-10-31.json")
                       .read_text())["development_g_comp"]
    dev = json.loads(Path("results/dev_scores.json").read_text())["skies"]
    expected = json.loads(a.expected_astro.read_text()) if a.expected_astro else {}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(12, 15.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(len(ORDER), 3, width_ratios=[1.0, 1.15, 1.1], wspace=0.35, hspace=0.5,
                          top=0.92, bottom=0.10, left=0.07, right=0.98)
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=8)
    notes = {}
    for row, sid in enumerate(ORDER):
        d = Path(SKY_DIRS[sid])
        with np.load(d / "event_inventory.npz") as inv:
            ev = {k: np.asarray(inv[k]) for k in cs.OBSERVABLE_FIELDS}
        dec, ch = ev["reco_dec_deg"].astype(float), ev["reco_energy_channel"].astype(np.int64)
        ref = cs.get_reference(dec, ch, cfg)
        C = pair_counts_by_channel(ev["reco_ra_deg"], dec, ch)
        Zp, Ze = cs.per_scale(C, ref, "pooled"), cs.per_scale(C, ref, "energy")
        t = int(Ze.argmax()); theta = THR[t]
        con = cs.energy_contributions(C, ref)[t]

        # ---- heatmap
        ax = fig.add_subplot(gs[row, 0])
        M = np.full((5, 5), np.nan)
        iu = np.triu_indices(5)
        M[iu] = con["z_dense"]
        ax.imshow(M, cmap=CMAP, norm=norm, origin="upper")
        for k, (i, j) in enumerate(zip(*iu)):
            if np.isnan(con["z_dense"][k]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///",
                                       edgecolor="#b8b7b0", lw=0))
            else:
                z = con["z_dense"][k]
                ax.text(j, i, f"{z:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(z) > 5 else INK)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xlabel("channel b"); ax.set_ylabel("channel a")
        for s in ax.spines.values(): s.set_visible(False)
        sz = con["z_sparse"]
        ax.set_title(f"sky {sid}: z_ab at θ = {theta}°\nsparse cell ({len(con['sparse_pairs'])} pairs merged): "
                     f"z = {sz:.1f}" if np.isfinite(sz) else f"sky {sid}: z_ab at θ = {theta}°",
                     fontsize=9, loc="left")

        # ---- per-scale lines
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.axhline(0, color="#d9d8d2", lw=1)
        ax2.plot(THR, Zp, "-o", color=BLUE, lw=2, ms=6, label="pooled", mec="#fcfcfb", mew=1.5)
        ax2.plot(THR, Ze, "-s", color=RED, lw=2, ms=6, label="energy", mec="#fcfcfb", mew=1.5)
        ax2.set_xscale("log"); ax2.set_xticks(THR); ax2.set_xticklabels([str(x) for x in THR])
        ax2.minorticks_off()
        ax2.set_xlabel("θ (deg)"); ax2.set_ylabel("Z_t (reference-standardized)")
        for s in ("top", "right"): ax2.spines[s].set_visible(False)
        ax2.grid(axis="y", color="#ecebe6", lw=.8)
        if row == 0: ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2)

        # ---- annotation text
        ph = posthoc_same_source(sid, d, theta)
        notes[sid] = ph
        nch = np.bincount(ch, minlength=5)
        exp_s = f"{expected[sid]:.0f}" if sid in expected else "n/a (to add)"
        r = dev[sid]
        txt = (f"g_comp = {gcomp[sid]:.2f}\n"
               f"expected astro count: {exp_s}\n"
               f"events: {dec.size}  (astro {ph['n_astro']}, truth)\n"
               f"per channel: {' / '.join(str(x) for x in nch)}\n\n"
               f"S_pool = {r['pooled']['S_obs']:.2f}  (dev p {r['pooled']['dev_p']:.3f})\n"
               f"S_E    = {r['energy']['S_obs']:.2f}  (dev p {r['energy']['dev_p']:.3f})\n\n"
               f"post hoc, truth labels, θ = {theta}°:\n"
               f"same-source pairs: {ph['same_src_pairs']}"
               f" ({ph['same_src_pairs_above_00']} outside (0,0))\n"
               f"largest source: {ph['largest_source_events']} events,\n"
               f"  {100 * ph['largest_source_pair_share']:.0f}% of same-source pairs")
        ax3 = fig.add_subplot(gs[row, 2]); ax3.axis("off")
        ax3.text(0, 1, txt, va="top", ha="left", family="monospace", fontsize=8.2)

    cax = fig.add_axes([0.09, 0.062, 0.25, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal",
                      extend="both")
    cb.set_label("z_ab = (C_ab − μ_ref) / √(σ²_ref + 1)   (clipped at −4, +8)")
    cb.outline.set_visible(False)
    fig.suptitle("Energy-channel contributions to the energy score, same sky (5 development skies)\n"
                 "Development diagnostic only: dev p uses development nulls, not calibration draws",
                 x=0.07, ha="left", y=0.975, fontsize=11)
    fig.text(0.07, 0.018, f"score config {cfg['config_sha256'][:12]}  ·  hatched = merged into sparse cell "
             f"(reference mean < {cfg['energy']['min_ref_mean']:g})  ·  rows ordered by g_comp",
             fontsize=8, color=MUTED)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor=fig.get_facecolor())
    a.out.with_suffix(".json").write_text(json.dumps(notes, indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
