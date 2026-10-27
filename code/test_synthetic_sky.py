"""Checks for the toy sky generator. Run: python test_synthetic_sky.py"""

import numpy as np

from synthetic_sky import DEC_MIN_DEG, ToySkyConfig, make_toy_sky
from pair_counter import angular_separation_deg, radec_to_xyz

OBS_FIELDS = {"period", "reco_log10_energy", "reco_energy_channel",
              "reco_ra_deg", "reco_dec_deg", "angular_uncertainty_deg"}


def test_reproducible():
    a, _ = make_toy_sky(ToySkyConfig(seed=3, f_multiplet=0.5))
    b, _ = make_toy_sky(ToySkyConfig(seed=3, f_multiplet=0.5))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_counts_and_format():
    obs, truth = make_toy_sky(ToySkyConfig(seed=1, n_total=3760, n_astro=80, f_multiplet=0.5))
    assert set(obs) == OBS_FIELDS  # no truth columns in the observable inventory
    assert all(v.size == 3760 for v in obs.values())
    assert (truth["component"] == "astro").sum() == 80
    assert np.all((obs["reco_ra_deg"] >= 0) & (obs["reco_ra_deg"] < 360))
    assert np.all((obs["reco_dec_deg"] >= DEC_MIN_DEG) & (obs["reco_dec_deg"] <= 90))
    assert obs["reco_energy_channel"].min() >= 0 and obs["reco_energy_channel"].max() <= 4


def test_multiplet_bookkeeping():
    _, t = make_toy_sky(ToySkyConfig(seed=2, n_astro=80, f_multiplet=0.0))
    assert t["source_is_multiplet"].sum() == 0
    _, t = make_toy_sky(ToySkyConfig(seed=2, n_astro=81, f_multiplet=1.0, multiplet_size=3))
    assert t["source_is_multiplet"].sum() == 27
    astro = t["source_index"][t["component"] == "astro"]
    assert np.all(np.bincount(astro) == 3)


def test_psf_scale():
    cfg = ToySkyConfig(seed=4, n_total=2000, n_astro=2000, f_multiplet=1.0, multiplet_size=2)
    obs, t = make_toy_sky(cfg)
    src = t["source_index"]
    xyz_e = radec_to_xyz(obs["reco_ra_deg"], obs["reco_dec_deg"])
    xyz_s = radec_to_xyz(t["source_ra_deg"][src], t["source_dec_deg"][src])
    err = angular_separation_deg(xyz_e, xyz_s)
    med = np.median(err)  # 2D Gaussian: median = sigma * sqrt(2 ln 2)
    assert abs(med - cfg.astro_psf_sigma_deg * np.sqrt(2 * np.log(2))) < 0.03, med


def test_astro_harder_than_atmo():
    obs, t = make_toy_sky(ToySkyConfig(seed=5, n_total=20000, n_astro=5000))
    ch, astro = obs["reco_energy_channel"], t["component"] == "astro"
    assert (ch[astro] > 0).mean() > 0.3 and (ch[~astro] > 0).mean() < 0.1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in tests:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
