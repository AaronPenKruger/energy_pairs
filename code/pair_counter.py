"""
Energy-resolved angular pair counter.

For each angular threshold theta, counts unique unordered event pairs (i < j)
with angular separation <= theta, split by the energy channels of the two
events. Result C[t, a, b] is stored for a <= b only (upper triangle);
entries with a > b are always zero.

    pooled[t] = sum over a <= b of C[t, a, b]

Inputs are reconstructed quantities only: RA, Dec (degrees) and the integer
energy channel. Truth columns (component, source_index, true_*, psf_deg) are
never read here.

Geometry: events are converted to 3D unit vectors, so RA wraparound at
0/360 deg and the poles need no special handling. Angular separation theta
corresponds to chord length 2*sin(theta/2) between unit vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

THRESHOLDS_DEG = (0.5, 1.0, 2.0, 3.0, 5.0)
N_CHANNELS = 5


def radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def angular_separation_deg(xyz_i: np.ndarray, xyz_j: np.ndarray) -> np.ndarray:
    """Numerically stable separation from unit vectors (atan2 form)."""
    cross = np.linalg.norm(np.cross(xyz_i, xyz_j), axis=-1)
    dot = np.einsum("ij,ij->i", xyz_i, xyz_j)
    return np.degrees(np.arctan2(cross, dot))


def _check_inputs(ra, dec, channel, n_channels):
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    channel = np.asarray(channel)
    if not (ra.shape == dec.shape == channel.shape and ra.ndim == 1):
        raise ValueError("ra, dec, channel must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValueError("non-finite RA/Dec")
    if np.any(np.abs(dec) > 90.0):
        raise ValueError("|Dec| > 90 deg: are these radians or the wrong column?")
    if not np.issubdtype(channel.dtype, np.integer):
        raise ValueError("energy channel must be integer")
    if channel.size and (channel.min() < 0 or channel.max() >= n_channels):
        raise ValueError(f"energy channel outside 0..{n_channels - 1}")
    return ra, dec, channel.astype(np.int64)


def candidate_pairs(ra_deg, dec_deg, max_theta_deg):
    """All unique pairs (i<j) within max_theta_deg and their separations."""
    xyz = radec_to_xyz(ra_deg, dec_deg)
    # small margin so floating point never drops a pair at the boundary;
    # the exact cut is applied on the separation afterwards
    r = 2.0 * np.sin(np.radians(max_theta_deg) / 2.0) * (1 + 1e-9) + 1e-12
    pairs = cKDTree(xyz).query_pairs(r, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    sep = angular_separation_deg(xyz[pairs[:, 0]], xyz[pairs[:, 1]])
    return pairs.astype(np.int64), sep


def pair_counts_by_channel(
    ra_deg,
    dec_deg,
    channel,
    thresholds_deg=THRESHOLDS_DEG,
    n_channels: int = N_CHANNELS,
) -> np.ndarray:
    """
    Returns C with shape (len(thresholds), n_channels, n_channels), int64.
    C[t, a, b] (a <= b) = number of unique pairs with separation <= thresholds[t]
    whose channels are {a, b}.
    """
    ra, dec, ch = _check_inputs(ra_deg, dec_deg, channel, n_channels)
    thr = np.asarray(thresholds_deg, dtype=float)
    if np.any(np.diff(thr) <= 0):
        raise ValueError("thresholds must be strictly increasing")

    C = np.zeros((thr.size, n_channels, n_channels), dtype=np.int64)
    pairs, sep = candidate_pairs(ra, dec, thr.max())
    if sep.size == 0:
        return C

    a = np.minimum(ch[pairs[:, 0]], ch[pairs[:, 1]])
    b = np.maximum(ch[pairs[:, 0]], ch[pairs[:, 1]])
    flat = a * n_channels + b
    for t, theta in enumerate(thr):
        m = sep <= theta
        C[t] = np.bincount(flat[m], minlength=n_channels * n_channels).reshape(
            n_channels, n_channels
        )
    return C


def pooled_counts(C: np.ndarray) -> np.ndarray:
    """Pooled pair count per threshold = sum of all channel-pair counts."""
    return C.sum(axis=(-2, -1))


def channel_pair_labels(n_channels: int = N_CHANNELS):
    """The 15 (a, b) combinations with a <= b, in a fixed order."""
    return [(a, b) for a in range(n_channels) for b in range(a, n_channels)]


def flatten_upper(C: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """(..., T, n, n) -> (..., T, n(n+1)/2) in channel_pair_labels order."""
    iu = np.triu_indices(n_channels)
    return C[..., iu[0], iu[1]]


def brute_force_counts(ra_deg, dec_deg, channel, thresholds_deg=THRESHOLDS_DEG,
                       n_channels: int = N_CHANNELS) -> np.ndarray:
    """Direct O(N^2) reference, for tests only. Uses spherical law of cosines
    via haversine on RA/Dec directly (independent of the 3D-vector path)."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    ch = np.asarray(channel)
    thr = np.asarray(thresholds_deg, float)
    C = np.zeros((thr.size, n_channels, n_channels), dtype=np.int64)
    n = ra.size
    for i in range(n):
        for j in range(i + 1, n):
            h = (np.sin((dec[j] - dec[i]) / 2) ** 2
                 + np.cos(dec[i]) * np.cos(dec[j]) * np.sin((ra[j] - ra[i]) / 2) ** 2)
            sep = np.degrees(2 * np.arcsin(min(1.0, np.sqrt(h))))
            a, b = sorted((int(ch[i]), int(ch[j])))
            for t, theta in enumerate(thr):
                if sep <= theta:
                    C[t, a, b] += 1
    return C
