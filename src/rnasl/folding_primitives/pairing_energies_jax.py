from typing import Set, Tuple

import jax
import jax.numpy as jnp

from rnasl.utils.formats import BASE_TO_INT, CANONICAL_PAIRS


def init_energy_mat(size: int = 4, value: float = 0.0) -> jnp.ndarray:
    return jnp.full((size, size), value)


def energy_mat_set_pair(mat: jnp.ndarray, i: int, j: int, energy: float) -> jnp.ndarray:
    mat = mat.at[i, j].set(energy)
    mat = mat.at[j, i].set(energy)
    return mat


def energy_mat_set_noncanonical(mat: jnp.ndarray, energy: float,
        canonical_pairs: Set[Tuple[int, int]] = None) -> jnp.ndarray:
    if not canonical_pairs:
        canonical_pairs = CANONICAL_PAIRS
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if (i, j) not in canonical_pairs and (j, i) not in canonical_pairs:
                mat = mat.at[i, j].set(energy)
    return mat


def init_random_energies(n: int = 4, seed=42) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    mat = jnp.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            key, subkey = jax.random.split(key)
            if (i, j) in CANONICAL_PAIRS or (j, i) in CANONICAL_PAIRS:
                energy = jax.random.uniform(subkey, shape=(), minval=-2.5, maxval=-0.5)
            else:
                energy = jax.random.uniform(subkey, shape=(), minval=1.0, maxval=3.0)
            mat = mat.at[i, j].set(energy)
            mat = mat.at[j, i].set(energy)
    return mat
