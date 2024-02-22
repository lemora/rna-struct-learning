from rnasl.jax_setup import jfloat

import itertools

import jax.numpy as jnp
import numpy as np

import rnasl.gconst as gc
from rnasl.folding_primitives.pairing_energies import PairingEnergies
from rnasl.utils.formats import BASE_TO_INT

# 1st,99th percentiles for each base in each dataset
RDATA_STATS_PERCENTILE = {
    "2A3": {
        0: (-0.219, 2.774),  # A
        1: (-0.469, 2.203),  # C
        2: (-0.539, 2.096),  # G
        3: (-0.359, 3.092),  # U
    },
    "DMS": {
        0: (-0.178, 3.394),  # A
        1: (-0.313, 3.246),  # C
        2: (-0.499, 0.739),  # G
        3: (-0.403, 0.737),  # U
    },
}


def normalize_reactivities_per_base_np(seq, reactivities, dataset_type, stats=RDATA_STATS_PERCENTILE):
    normed = np.full_like(reactivities, np.nan)
    for i, base in enumerate(seq):
        if base not in BASE_TO_INT or not np.isfinite(reactivities[i]):
            continue
        base_int = BASE_TO_INT[base]
        p1, p99 = stats[dataset_type][base_int]
        denom = p99 - p1 if p99 != p1 else 1.0
        # normed[i] = np.clip((reactivities[i] - p1) / denom, 0.0, 1.0)
        normed[i] = (reactivities[i] - p1) / denom
    return normed


# ------ boltz

def boltz(energy: float, temperature=gc.TEMP) -> float:
    return np.exp(-energy / (gc.K_B * temperature))


def log_boltz(energy: float, temperature=gc.TEMP) -> float:
    return -energy / (gc.K_B * temperature)


def boltz_jax(energy: float, temperature=gc.TEMP) -> float:
    return jnp.exp(-energy / (gc.K_B * temperature))


def log_boltz_jax(energy, temperature=gc.TEMP):
    return -energy / (gc.K_B * temperature)


# ------ seq and struct validators

def ensure_rna_bases(seq: str):
    seq = seq.upper()
    if "T" in seq:
        print("seq is DNA, converting to RNA.")
        return seq.replace('T', 'U')
    return seq


def isXNA(seq):
    is_rna = all(c in "AUGC" for c in seq.upper())
    if is_rna:
        return True
    is_dna = all(c in "ATGC" for c in seq.upper())
    if is_dna:
        return True
    return False


def is_valid_non_crossing(pairs: list[tuple[int, int]]) -> bool:
    """Check that base pairs are non-crossing and non-overlapping."""
    seen = set()
    for i, j in pairs:
        if i in seen or j in seen:
            return False
        seen.add(i)
        seen.add(j)

    for (i, j), (k, l) in itertools.combinations(pairs, 2):
        if i < k < j < l or k < i < l < j:
            return False

    return True


# ------ seq/struct/bpp generators

def count_valid_structures(seq, min_loop_len=0):
    n = len(seq)

    def count(i, j):
        if i >= j:
            return 1  # empty structure
        total = count(i + 1, j)
        for k in range(i + min_loop_len + 1, j + 1):
            left = count(i + 1, k - 1)
            right = count(k + 1, j)
            total += left * right
        return total

    return count(0, n - 1)


def valid_pairs(n, h=3):
    return [(i, j) for i in range(n) for j in range(i + h + 1, n)]


def generate_structures(n, h=3):
    all_pairs = valid_pairs(n, h)
    for r in range(n // 2 + 1):
        for pairs in itertools.combinations(all_pairs, r):
            if is_valid_non_crossing(pairs):
                yield pairs


def structure_energy(seq: str, pairs: list[tuple[int, int]], energies: jnp.ndarray) -> float:
    energy = 0.0
    for i, j in pairs:
        energy += energies[seq[i], seq[j]]
    return energy


def brute_force_bpp(seq: str, energies: PairingEnergies, h=3) -> np.ndarray:
    n = len(seq)
    Z_total = 0.0
    bpp = np.zeros((n, n))
    structures = list(generate_structures(n, h=h))

    for struct in structures:
        E = jnp.sum(jnp.array([energies.paired(seq[i], seq[j]) for i, j in struct]))
        weight = boltz(E)
        Z_total += weight
        for i, j in struct:
            bpp[i, j] += weight
            bpp[j, i] += weight

    bpp /= Z_total
    return np.array(bpp)


def brute_force_bpp_jax(seq: jnp.ndarray, energies: jnp.ndarray, h: int = 3) -> jnp.ndarray:
    n = len(seq)
    bpp = jnp.zeros((n, n), dtype=jfloat)
    Z_total = jnp.array(0.0, dtype=jfloat)

    for struct in generate_structures(n, h=h):
        energy = jnp.sum(jnp.array([energies[seq[i], seq[j]] for i, j in struct]))
        weight = boltz_jax(energy).astype(jfloat)

        for i, j in struct:
            bpp = bpp.at[i, j].add(weight)
            bpp = bpp.at[j, i].add(weight)
        Z_total += weight

    return jnp.where(Z_total > 0, bpp / Z_total, bpp)
