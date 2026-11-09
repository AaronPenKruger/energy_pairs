"""Tests for the frozen scores. Run: python -m pytest -q code/test_clustering_scores.py"""
import json, sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustering_scores as cs
from pair_counter import pair_counts_by_channel

ROOT = Path(__file__).resolve().parent.parent
SKY = ROOT / "aaron_dev_extension_2026-10-31/uniform_55289"

@pytest.fixture(scope="module")
def cfg():
    return cs.load_config()

@pytest.fixture(scope="module")
def ev():
    with np.load(SKY / "event_inventory.npz", allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}   # includes truth columns on purpose

def test_config_hash_guard(tmp_path, cfg):
    bad = dict(cfg); bad["energy"] = dict(cfg["energy"], min_ref_mean=3.0)
    p = tmp_path / "c.json"; p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        cs.load_config(p)

def test_truth_columns_ignored(ev, cfg):
    obs_only = {k: ev[k] for k in cs.OBSERVABLE_FIELDS}
    assert cs.score_both(ev, cfg) == cs.score_both(obs_only, cfg)
    poisoned = dict(ev, component=np.full(ev["reco_ra_deg"].size, "astro"),
                    source_index=np.zeros(ev["reco_ra_deg"].size, int))
    assert cs.score_both(poisoned, cfg) == cs.score_both(ev, cfg)

def test_event_order_invariance(ev, cfg):
    perm = np.random.default_rng(1).permutation(ev["reco_ra_deg"].size)
    shuffled = {k: ev[k][perm] for k in cs.OBSERVABLE_FIELDS}
    assert cs.score_both(shuffled, cfg) == cs.score_both(ev, cfg)

def test_original_and_randomized_inventory_share_reference(ev, cfg):
    """A uniform-RA draw changes only RA, so the reference (and scoring rule) is identical."""
    dec, ch = ev["reco_dec_deg"], ev["reco_energy_channel"].astype(np.int64)
    rnd = dict(ev, reco_ra_deg=np.random.default_rng(20316292).uniform(0, 360, dec.size))
    assert cs.inventory_key(dec, ch) == cs.inventory_key(rnd["reco_dec_deg"], ch)
    # score() on the randomized inventory == precomputed-count fast path with the same reference
    C = pair_counts_by_channel(rnd["reco_ra_deg"], dec, ch)
    for m in ("pooled", "energy"):
        assert cs.score(rnd, m, cfg) == cs.score_from_counts(C, dec, ch, m, cfg)

def test_reference_reproduces_frozen_numbers(ev, cfg):
    cs._REF_CACHE.clear()
    dec, ch = ev["reco_dec_deg"].astype(float), ev["reco_energy_channel"].astype(np.int64)
    ref = cs.get_reference(dec, ch, cfg)
    fz = np.load(ROOT / "frozen/reference_55289.npz")
    assert int(fz["inventory_key"]) == cs.inventory_key(dec, ch)
    assert np.array_equal(fz["dense"], ref["dense"])
    for k in ("pooled_mu", "pooled_sd", "T_mu", "T_sd"):          # floats: allow last-bit platform noise
        np.testing.assert_allclose(fz[k], ref[k], rtol=1e-10, err_msg=k)
    for t in range(5):
        np.testing.assert_allclose(fz[f"cell_mu_t{t}"], ref["cell_mu"][t], rtol=1e-10)
        np.testing.assert_allclose(fz[f"cell_den_t{t}"], ref["cell_den"][t], rtol=1e-10)

def test_reference_stream_differs_from_null_stream(ev, cfg):
    """Reference draw 0 must not equal Linh's null draw 0 for this sky."""
    dec, ch = ev["reco_dec_deg"].astype(float), ev["reco_energy_channel"].astype(np.int64)
    ss = np.random.SeedSequence(cfg["reference"]["entropy"], spawn_key=(cs.inventory_key(dec, ch),))
    ref0 = np.random.default_rng(ss).uniform(0, 360, dec.size)
    null0 = np.random.default_rng(20316292).uniform(0, 360, dec.size)
    assert not np.any(ref0 == null0)

def test_energy_label_shuffle_leaves_pooled_unchanged(ev, cfg):
    ch = ev["reco_energy_channel"].copy(); np.random.default_rng(2).shuffle(ch)
    C1 = pair_counts_by_channel(ev["reco_ra_deg"], ev["reco_dec_deg"], ev["reco_energy_channel"])
    C2 = pair_counts_by_channel(ev["reco_ra_deg"], ev["reco_dec_deg"], ch)
    assert np.array_equal(C1.sum((1, 2)), C2.sum((1, 2)))

def test_calibration_helper_matches_score(ev, cfg):
    draws = np.random.default_rng(5).uniform(0, 360, (3, ev["reco_ra_deg"].size))
    out = cs.score_sky_and_draws(ev, draws, cfg)
    for m in ("pooled", "energy"):
        assert out["observed"][m] == cs.score(ev, m, cfg)
        for i in range(3):
            assert out["null"][m][i] == cs.score(dict(ev, reco_ra_deg=draws[i]), m, cfg)
