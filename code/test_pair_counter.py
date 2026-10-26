"""
Tests for pair_counter.py. Run with:  python test_pair_counter.py
(or pytest test_pair_counter.py). Each check prints PASS or raises.
"""

import numpy as np

from pair_counter import (
    THRESHOLDS_DEG,
    brute_force_counts,
    channel_pair_labels,
    flatten_upper,
    pair_counts_by_channel,
    pooled_counts,
)


def random_sky(n, rng, dec_min=5.0, ra_lo=0.0, ra_hi=360.0, dec_hi=90.0):
    """Uniform on the sphere cap, optionally a small patch so pairs are common."""
    ra = rng.uniform(ra_lo, ra_hi, n) % 360.0
    s = rng.uniform(np.sin(np.radians(dec_min)), np.sin(np.radians(dec_hi)), n)
    dec = np.degrees(np.arcsin(s))
    ch = rng.integers(0, 5, n)
    return ra, dec, ch


def test_matches_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = int(rng.integers(2, 120))
        # dense patch spanning RA=0 so wraparound pairs are exercised
        ra, dec, ch = random_sky(n, rng, dec_min=10, dec_hi=25, ra_lo=-8, ra_hi=8)
        assert np.array_equal(pair_counts_by_channel(ra, dec, ch),
                              brute_force_counts(ra, dec, ch)), trial
    for trial in range(5):  # full sky, sparse
        ra, dec, ch = random_sky(300, rng)
        assert np.array_equal(pair_counts_by_channel(ra, dec, ch),
                              brute_force_counts(ra, dec, ch))


def test_no_self_pairs_or_double_counts():
    # one event: zero pairs
    assert pooled_counts(pair_counts_by_channel([10.0], [20.0], [0])).sum() == 0
    # two events at the identical position: exactly one pair at every threshold
    C = pair_counts_by_channel([10.0, 10.0], [20.0, 20.0], [1, 1])
    assert np.all(pooled_counts(C) == 1) and np.all(C[:, 1, 1] == 1)
    # k events at one spot: k(k-1)/2 pairs
    k = 7
    C = pair_counts_by_channel(np.full(k, 50.0), np.full(k, 30.0), np.zeros(k, int))
    assert np.all(pooled_counts(C) == k * (k - 1) // 2)


def test_units_are_degrees():
    # pair 0.7 deg apart in Dec: not within 0.5, within 1, 2, 3, 5
    C = pair_counts_by_channel([100.0, 100.0], [30.0, 30.7], [0, 0])
    assert pooled_counts(C).tolist() == [0, 1, 1, 1, 1]
    # pair 4 deg apart: only within 5
    C = pair_counts_by_channel([100.0, 100.0], [30.0, 34.0], [0, 0])
    assert pooled_counts(C).tolist() == [0, 0, 0, 0, 1]
    # RA difference shrinks by cos(dec): 1.0 deg in RA at dec=60 is ~0.5 deg on sky
    C = pair_counts_by_channel([100.0, 100.9], [60.0, 60.0], [0, 0])
    assert pooled_counts(C).tolist() == [1, 1, 1, 1, 1]


def test_ra_wraparound():
    # 359.8 and 0.1 at dec 0: 0.3 deg apart
    C = pair_counts_by_channel([359.8, 0.1], [0.0, 0.0], [0, 0])
    assert pooled_counts(C).tolist() == [1, 1, 1, 1, 1]
    # 359.1 and 0.8 at dec 0: 1.7 deg apart, counted at >= 2
    # (separations exactly on a threshold are not tested: floating point
    #  decides those, and they have probability zero for continuous RA)
    C = pair_counts_by_channel([359.1, 0.8], [0.0, 0.0], [0, 0])
    assert pooled_counts(C).tolist() == [0, 0, 1, 1, 1]


def test_pole():
    # opposite RAs right next to the pole are close together
    C = pair_counts_by_channel([0.0, 180.0], [89.9, 89.9], [0, 0])
    assert pooled_counts(C).tolist() == [1, 1, 1, 1, 1]  # 0.2 deg apart
    C = pair_counts_by_channel([0.0, 180.0], [88.0, 88.0], [0, 0])
    assert pooled_counts(C).tolist() == [0, 0, 0, 0, 1]  # 4 deg apart


def test_cross_channel_stored_upper_triangle():
    C = pair_counts_by_channel([10.0, 10.0], [20.0, 20.1], [3, 1])
    assert np.all(C[:, 1, 3] == 1) and np.all(C[:, 3, 1] == 0)
    assert np.all(np.tril(C[0], k=-1) == 0)


def test_channel_sum_equals_pooled():
    rng = np.random.default_rng(2)
    ra, dec, ch = random_sky(2000, rng)
    C = pair_counts_by_channel(ra, dec, ch)
    flat = flatten_upper(C)
    assert flat.shape == (len(THRESHOLDS_DEG), len(channel_pair_labels()))
    assert np.array_equal(flat.sum(axis=1), pooled_counts(C))


def test_event_order_invariance():
    rng = np.random.default_rng(3)
    ra, dec, ch = random_sky(1500, rng)
    p = rng.permutation(ra.size)
    assert np.array_equal(pair_counts_by_channel(ra, dec, ch),
                          pair_counts_by_channel(ra[p], dec[p], ch[p]))


def test_pooled_ignores_energy_labels():
    rng = np.random.default_rng(4)
    ra, dec, ch = random_sky(1500, rng)
    ch2 = rng.permutation(ch)
    assert np.array_equal(pooled_counts(pair_counts_by_channel(ra, dec, ch)),
                          pooled_counts(pair_counts_by_channel(ra, dec, ch2)))


def test_counts_nondecreasing_with_threshold():
    rng = np.random.default_rng(5)
    ra, dec, ch = random_sky(1500, rng)
    C = pair_counts_by_channel(ra, dec, ch)
    assert np.all(np.diff(C, axis=0) >= 0)


def test_rejects_bad_inputs():
    for args in [([0.0], [100.0], [0]),        # dec > 90
                 ([0.0], [10.0], [5]),         # channel out of range
                 ([0.0], [10.0], [0.5]),       # non-integer channel
                 ([np.nan], [10.0], [0])]:
        try:
            pair_counts_by_channel(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input {args}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
