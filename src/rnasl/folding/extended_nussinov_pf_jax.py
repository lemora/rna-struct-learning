from rnasl.jax_setup import jfloat

from functools import partial

import jax
import jax.numpy as jnp
from jax import config, lax, vmap

from rnasl.folding_primitives.semiring import NumpyLogSumExpSemiring, Semiring, \
    make_logsumexp_semiring, JaxSemiringFrozen

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["semiring", "h"])
def calc_extended_pseudo_partition_function(seq: jnp.ndarray,
        energy_mat: jnp.ndarray,
        semiring: Semiring,
        h: int = 3,
        hairpin_pen: float = 1.0,
        internal_pen: float = 0.5,
        stem_initiation_pen: float = 3.0,
        stacking_bonus: float = -1.0
):
    """
    Pseudo-partition function with extended penalties and bonuses.

    Assumptions:
    - Any (i,j) >=h can pair
    - Loops are hairpins (<=30 nt) or internal (>30), penalized linearly
    - If space >=h exists between (i+1,j−1), stacking_bonus is applied; no check if (i+1,j−1) actually pairs

    Note:
    - Z overcounts configurations due to overlapping decompositions, so it is not a true partition function.
    - So Z is a *pseudo-partition function*: useful for optimization but not a true Boltzmann sum.
    """
    n = seq.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jfloat), jnp.zeros((0, 0), dtype=jfloat)
    return _impl_pseudo_pf(seq, energy_mat, semiring, h, hairpin_pen, internal_pen)


def _impl_pseudo_pf(seq, energy_mat, semiring, h,
        hairpin_pen=1.0, internal_pen=1.0,
        stem_initiation_pen=2.0, stacking_bonus=-1.0):
    n = seq.shape[0]
    Z = jnp.full((n, n), semiring.zero(), dtype=jfloat)
    Z_p = jnp.full((n, n), semiring.zero(), dtype=jfloat)

    i_idx, j_idx = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing='ij')
    mask = (i_idx >= j_idx) | (j_idx - i_idx == 1)
    Z = jnp.where(mask, semiring.one(), Z)

    def span_step(carry, span):
        Z, Z_p = carry

        def i_step(carry_inner, i):
            Z, Z_p = carry_inner
            j = i + span

            def do_compute(Z, Z_p):
                i_base, j_base = seq[i], seq[j]
                e_ij = energy_mat[i_base, j_base]
                can_pair = (j - i - 1) >= h

                loop_len = j - i - 1
                is_hairpin = (loop_len <= 30)

                # loop penalties: hairpin and internal
                loop_penalty = jnp.where(
                    is_hairpin,
                    loop_len * hairpin_pen,
                    loop_len * internal_pen
                )

                # stack bonus/penalties: longer stem or initiation
                stack_exists = (j - i - 2) >= h
                stack_penalty = jnp.where(
                    stack_exists,
                    stacking_bonus,
                    stem_initiation_pen
                )

                total_penalty = loop_penalty + stack_penalty
                penalty_enc = semiring.encode(total_penalty)

                Z_ij_paired = semiring.mul(
                    semiring.encode(e_ij),
                    semiring.mul(penalty_enc, Z[i + 1, j - 1])
                )

                Z_ij_paired = jnp.where(can_pair, Z_ij_paired, semiring.zero())
                Z_p = Z_p.at[i, j].set(Z_ij_paired)

                k_vals = jnp.arange(n)
                valid_k = (k_vals >= i) & (k_vals < (j - h + 1))
                lefts = jnp.where(k_vals > i, Z[i, k_vals - 1], semiring.one())
                rights = Z_p[k_vals, j]
                terms = semiring.mul(lefts, rights)
                terms = jnp.where(valid_k, terms, semiring.zero())
                sum_terms = semiring.add_n(terms)

                Z_ij = semiring.add(Z[i, j - 1], sum_terms)
                Z = Z.at[i, j].set(Z_ij)

                return Z, Z_p

            Z, Z_p = lax.cond(j < n, lambda _: do_compute(Z, Z_p), lambda _: (Z, Z_p), operand=None)
            return (Z, Z_p), None

        (Z, Z_p), _ = lax.scan(i_step, (Z, Z_p), jnp.arange(n))
        return (Z, Z_p), None

    (Z, Z_p), _ = lax.scan(span_step, (Z, Z_p), jnp.arange(1, n))
    return Z, Z_p
