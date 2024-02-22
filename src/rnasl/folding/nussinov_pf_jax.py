from rnasl.jax_setup import jfloat

from functools import partial

import jax
import jax.numpy as jnp
from jax import config, lax, vmap
import numpy as np

import rnasl.gconst as gc
from rnasl.folding_primitives.semiring import NumpyLogSumExpSemiring, Semiring, \
    make_logsumexp_semiring, JaxSemiringFrozen
from rnasl.utils.formats import encode_seq_jax, pairing_to_vienna
from rnasl.visualization.visualize import (bp_probability_circle_plot, bp_probability_dot_plot,
                                           plot_paired_unpaired_probs, print_npmat)

jax.config.update("jax_enable_x64", True)


def get_energy(energy_mat: jnp.ndarray, a: int, b: int) -> float:
    i = jnp.minimum(a, b)
    j = jnp.maximum(a, b)
    return energy_mat[i, j]

@partial(jax.jit, static_argnames=["semiring", "h"])
def calc_partition_function(seq: jnp.ndarray, energy_mat: jnp.ndarray, semiring: Semiring, h: int = 3):
    n = seq.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jfloat), jnp.zeros((0, 0), dtype=jfloat)
    return _partition_impl(seq, energy_mat, semiring, h)


def _partition_impl(seq, energy_mat, semiring, h):
    # init tables
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
                # Calc Z_p
                i_base, j_base = seq[i], seq[j]
                # e_ij = energy_mat[i_base, j_base]
                e_ij = get_energy(energy_mat, i_base, j_base)
                can_pair = (j - i - 1) >= h
                Z_ij_paired = semiring.mul(semiring.encode(e_ij), Z[i + 1, j - 1])
                Z_ij_paired = jnp.where(can_pair, Z_ij_paired, semiring.zero())
                Z_p = Z_p.at[i, j].set(Z_ij_paired)

                # Calc Z (fixed-size loop with masking)
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


# ------------ bpp algs ------------

def calc_base_pair_probs_n4(Z, Z_p, seq, semiring):
    """
    Correct but very slow (1.2s, diff 53s).
    O(n^4)
    """
    n = seq.shape[0]
    zero = semiring.zero()
    one = semiring.one()
    dtype = Z.dtype
    Z_total = Z[0, n - 1]

    # Generate all i < j index pairs
    i_idx, j_idx = jnp.triu_indices(n, k=1)
    pairs = jnp.stack([i_idx, j_idx], axis=-1)

    def compute_Pij(pair, P_cur):
        i, j = pair
        zp_ij = Z_p[i, j]

        def compute_valid():
            Z_left = lax.select(i > 0, Z[0, i - 1], one)
            Z_right = lax.select(j < n - 1, Z[j + 1, n - 1], one)
            Zt = jnp.where(jnp.isclose(Z_total, zero, atol=1e-6), one, Z_total)
            outer = semiring.div(semiring.mul(Z_left, semiring.mul(zp_ij, Z_right)), Zt)

            k = jnp.arange(n)
            l = jnp.arange(n)
            kk, ll = jnp.meshgrid(k, l, indexing='ij')
            kk = kk.ravel()
            ll = ll.ravel()

            def nested_term(k_, l_):
                valid = (k_ < i) & (l_ > j)
                Z_mid = lax.select(k_ + 1 <= l_ - 1, Z[k_ + 1, l_ - 1], one)
                denom_safe = jnp.where(jnp.isclose(Z_mid, zero, atol=1e-6), one, Z_mid)
                P_kl = P_cur[k_, l_]
                contrib = semiring.div(P_kl, denom_safe)
                return jnp.where(valid, contrib, zero)

            nested_contribs = jax.vmap(nested_term)(kk, ll)
            nested_sum = semiring.add_n(nested_contribs)
            nested = semiring.mul(zp_ij, nested_sum)

            return semiring.add(outer, nested)

        P_val = lax.cond(
            jnp.isclose(zp_ij, zero, atol=1e-6),
            lambda _: zero,
            lambda _: compute_valid(),
            operand=None
        )

        return P_cur.at[i, j].set(P_val)

    def update(P_cur, pair):
        return jax.checkpoint(compute_Pij)(pair, P_cur), None

    P0 = jnp.full((n, n), zero, dtype=dtype)
    P_final, _ = lax.scan(update, P0, pairs)

    P_real = semiring.to_real_matrix(P_final)
    P_real = jnp.nan_to_num(P_real)
    P_real = jnp.clip(P_real, 0.0, 1.0)
    P_real = jnp.maximum(P_real, P_real.T)
    P_real = P_real - jnp.diag(jnp.diag(P_real))
    return P_real


def calc_base_pair_probs_n4_fastad(Z, Z_p, seq, semiring):
    """
    Correct with better forward vs autodiff performance (1s, diff 2.9s).
    O(n^4)
    """
    n = seq.shape[0]
    zero = semiring.zero()
    one = semiring.one()
    dtype = Z.dtype
    Z_tot = jnp.where(semiring.is_zero(Z[0, n - 1]), one, Z[0, n - 1])

    # all i<j pairs, flattened index k
    i_idx, j_idx = jnp.triu_indices(n, k=1)
    m = i_idx.size

    # outer contribution for every pair (i,j)
    zp_vec = Z_p[i_idx, j_idx]
    Z_left = jnp.where(i_idx > 0, Z[0, i_idx - 1], one)
    Z_right = jnp.where(j_idx < n - 1, Z[j_idx + 1, n - 1], one)
    outer = semiring.div(semiring.mul(Z_left, semiring.mul(zp_vec, Z_right)), Z_tot)

    # Z_mid[k,l] = Z[k+1,l-1] (or 1)
    Z_mid = jnp.full((n, n), one, dtype=dtype)
    a, b = jnp.triu_indices(n, k=2)
    Z_mid = Z_mid.at[a, b].set(Z[a + 1, b - 1])
    Z_mid_flat = Z_mid[i_idx, j_idx]

    # flat P buffer in (i<j) order
    P_flat = jnp.full(m, zero, dtype=dtype)

    def body(idx, P_flat):
        i, j = i_idx[idx], j_idx[idx]
        zp, out = zp_vec[idx], outer[idx]

        # skip everything if Z_p[i,j] == 0
        def compute():
            # mask for k<i and l>j
            mask = (i_idx < i) & (j_idx > j)

            # compute nested sum with current P_flat
            div_terms = semiring.div(P_flat, Z_mid_flat)
            nested_sum = semiring.add_n(jnp.where(mask, div_terms, zero))
            nested = semiring.mul(zp, nested_sum)
            return semiring.add(out, nested)

        P_ij = lax.cond(jnp.isclose(zp, zero, atol=1e-6),
                        lambda _: zero,
                        lambda _: compute(),
                        operand=None)
        return P_flat.at[idx].set(P_ij)

    P_flat = lax.fori_loop(0, m, body, P_flat)

    # scatter into full matrix
    P = jnp.full((n, n), zero, dtype=dtype).at[i_idx, j_idx].set(P_flat)

    P_real = semiring.to_real_matrix(P)
    P_real = jnp.nan_to_num(P_real)
    P_real = jnp.clip(P_real, 0.0, 1.0)
    P_real = jnp.maximum(P_real, P_real.T)
    return P_real - jnp.diag(jnp.diag(P_real))


def calc_base_pair_probs_n3(Z, Z_p, seq, semiring):
    n = seq.shape[0]
    zero = semiring.zero()
    one = semiring.one()
    dtype = Z.dtype
    Z_tot = jnp.where(semiring.is_zero(Z[0, n - 1]), one, Z[0, n - 1])

    Z_mid = jnp.full((n, n), one, dtype=dtype)
    ii, jj = jnp.triu_indices(n, k=2)
    Z_mid = Z_mid.at[ii, jj].set(Z[ii + 1, jj - 1])

    # upper-triangular index order
    i_idx, j_idx = jnp.triu_indices(n, k=1)
    m = i_idx.size
    P = jnp.full((n, n), zero, dtype=dtype)

    # row-wise scan, iter pairs in asc span
    def body(idx, P_cur):
        i, j = i_idx[idx], j_idx[idx]
        zp_ij = Z_p[i, j]

        def compute():
            Z_left = lax.select(i > 0, Z[0, i - 1], one)
            Z_right = lax.select(j < n - 1, Z[j + 1, n - 1], one)

            outer = semiring.div(
                semiring.mul(Z_left, semiring.mul(zp_ij, Z_right)),
                Z_tot
            )

            mask_nested = (i_idx < i) & (j_idx > j)
            Z_mid_flat = Z_mid[i_idx, j_idx]
            div_terms = semiring.div(P_cur[i_idx, j_idx], Z_mid_flat)

            nested_sum = semiring.add_n(jnp.where(mask_nested, div_terms, zero))
            nested = semiring.mul(zp_ij, nested_sum)

            return semiring.add(outer, nested)

        P_ij = lax.cond(jnp.isclose(zp_ij, zero, atol=1e-6),
                        lambda _: zero,
                        lambda _: compute(),
                        operand=None)

        return P_cur.at[i, j].set(P_ij)

    P = lax.fori_loop(0, m, body, P)

    P_real = semiring.to_real_matrix(P)
    P_real = jnp.nan_to_num(P_real)
    P_real = jnp.clip(P_real, 0.0, 1.0)
    P_real = jnp.maximum(P_real, P_real.T)
    return P_real - jnp.diag(jnp.diag(P_real))


def calc_base_pair_probs_n3_fastad(Z, Z_p, seq, semiring):
    n = seq.shape[0]
    zero = semiring.zero()
    one = semiring.one()
    dtype = Z.dtype

    Z_tot = Z[0, n - 1]
    Z_tot = jnp.where(jnp.isneginf(Z_tot) | jnp.isnan(Z_tot), semiring.one(), Z_tot)

    Z_mid = jnp.full((n, n), one, dtype=dtype)
    ii, jj = jnp.triu_indices(n, k=2)
    mid_vals = Z[ii + 1, jj - 1]
    mid_vals_safe = jnp.where(jnp.isneginf(mid_vals) | jnp.isnan(mid_vals), one, mid_vals)
    Z_mid = Z_mid.at[ii, jj].set(mid_vals_safe)

    j_all = jnp.arange(n)

    def row_fn(i, carry):
        P, col_pref = carry
        Z_left = lax.select(i > 0, Z[0, i - 1], one)
        zp_row = Z_p[i]

        mask = j_all > i
        Z_right = jnp.where((j_all < n - 1) & mask, Z[j_all + 1, n - 1], one)

        outer_raw = semiring.mul(Z_left, semiring.mul(zp_row, Z_right))
        outer_raw = jnp.nan_to_num(outer_raw, nan=zero)
        outer = semiring.div(outer_raw, Z_tot)

        nested = semiring.mul(zp_row, jnp.nan_to_num(col_pref, nan=zero))
        P_row = semiring.add(outer, nested)
        P_row = jnp.where(mask, P_row, zero)
        P = P.at[i].set(P_row)

        denom = Z_mid[i]
        denom = jnp.where(jnp.isneginf(denom) | jnp.isnan(denom), one, denom)
        W_row = semiring.div(P_row, denom)
        W_row = jnp.where(mask, W_row, zero)
        W_row = jnp.nan_to_num(W_row, nan=zero)

        rev = W_row[::-1]
        rev_inclusive = lax.associative_scan(semiring.add, rev)
        rev_shift = jnp.concatenate([jnp.array([zero], dtype=dtype), rev_inclusive[:-1]])
        row_cum = rev_shift[::-1]

        col_pref = semiring.add(jnp.nan_to_num(col_pref, nan=zero), row_cum)
        return (P, col_pref)

    P0 = jnp.full((n, n), zero, dtype=dtype)
    col0 = jnp.full(n, zero, dtype=dtype)
    P_fin, _ = lax.fori_loop(0, n - 1, row_fn, (P0, col0))

    P_real = semiring.to_real_matrix(P_fin)
    P_real = jnp.nan_to_num(P_real)
    P_real = jnp.clip(P_real, 0.0, 1.0)
    P_real = jnp.maximum(P_real, P_real.T)
    return P_real - jnp.diag(jnp.diag(P_real))


def calc_base_pair_probs_intimg_refine(Z, Z_p, seq, semiring, *, refine_steps: int = 4, band: int = 50):
    n = seq.shape[0]
    band = min(band, n - 1)
    zero = semiring.zero()
    one = semiring.one()
    dtype = Z.dtype

    i_idx, j_idx = jnp.triu_indices(n, k=1)
    zp_ij = Z_p[i_idx, j_idx]
    Z_left = jnp.where(i_idx > 0, Z[0, i_idx - 1], one)
    Z_right = jnp.where(j_idx < n - 1, Z[j_idx + 1, n - 1], one)
    Z_tot_raw = Z[0, n - 1]
    Z_tot = jnp.where(semiring.is_zero(Z_tot_raw), one, Z_tot_raw)

    # initial outer contributions
    outer0 = semiring.div(semiring.mul(Z_left, semiring.mul(zp_ij, Z_right)), Z_tot)
    P = jnp.full((n, n), zero, dtype=dtype).at[i_idx, j_idx].set(outer0)

    # prepare for refinement
    kk, ll = jnp.triu_indices(n, k=1)
    valid_mask = kk + 1 < ll
    valid_idx = jnp.nonzero(valid_mask, size=kk.size, fill_value=0)[0]
    mid_kk = kk[valid_idx]
    mid_ll = ll[valid_idx]
    Z_mid_const = Z[mid_kk + 1, mid_ll - 1]

    @jax.checkpoint
    def refine_once(P_cur):
        A_vals = semiring.div(P_cur[mid_kk, mid_ll], Z_mid_const)
        A = jnp.full((n, n), zero, dtype=dtype).at[mid_kk, mid_ll].set(A_vals)

        # Stable nested sum via 2d flipped cumul sum
        S = jnp.flip(jnp.cumsum(jnp.cumsum(jnp.flip(A, 1), axis=0), axis=1), axis=1)

        i_safe = jnp.clip(i_idx - 1, 0)
        j_safe = jnp.clip(j_idx + 1, 0, n - 1)
        nested = semiring.mul(zp_ij, S[i_safe, j_safe])
        return P_cur.at[i_idx, j_idx].add(nested)

    for _ in range(refine_steps):
        P = refine_once(P)

    # Banded re-refinement
    if band > 0:
        band_mask = (j_idx - i_idx) <= band
        band_indices = jnp.nonzero(band_mask, size=band_mask.size, fill_value=0)[0]
        band_i = i_idx[band_indices]
        band_j = j_idx[band_indices]

        def update_entry(i, j):
            zp = Z_p[i, j]
            Zl = jnp.where(i > 0, Z[0, i - 1], one)
            Zr = jnp.where(j < n - 1, Z[j + 1, n - 1], one)
            outer = semiring.div(semiring.mul(Zl, semiring.mul(zp, Zr)), Z_tot)

            # vectorized nested sum over all (k, l)
            k_idx, l_idx = jnp.triu_indices(n, k=1)
            inside = (k_idx < i) & (l_idx > j) & (k_idx + 1 < l_idx)
            Zmid = jnp.where(inside, Z[k_idx + 1, l_idx - 1], one)
            Pkl = P[k_idx, l_idx]
            contrib = jnp.where(inside, semiring.div(Pkl, Zmid), zero)
            nested = semiring.mul(zp, semiring.add_n(contrib))
            return semiring.add(outer, nested)

        new_vals = jax.vmap(update_entry)(band_i, band_j)
        P = P.at[band_i, band_j].set(new_vals)

    P_sym = jnp.maximum(P, P.T)
    P_real = semiring.to_real_matrix(P_sym)
    P_real = jnp.nan_to_num(P_real)
    return jnp.clip(P_real - jnp.diag(jnp.diag(P_real)), 0.0, 1.0)


def calc_base_pair_probs_windowed(Z: jnp.ndarray, Z_p: jnp.ndarray, seq: jnp.ndarray, semiring) -> jnp.ndarray:
    """
    Approximates bpp using local nested pairs near (i,j) (limited by kl_window)
    """
    n = seq.shape[0]
    Z_total = Z[0, n - 1]
    kl_window = 4  # local window for enclosed pair corrections
    offsets = jnp.arange(-kl_window, kl_window + 1)

    # precompute all valid (i, j) pairs with i < j
    i_all, j_all = jnp.triu_indices(n, k=1)

    def compute_pair(P, idx):
        i = i_all[idx]
        j = j_all[idx]
        zp_ij = Z_p[i, j]
        should_compute = ~jnp.isclose(zp_ij, semiring.zero(), atol=1e-6)

        def update_pair_with_local_nested_support():
            Z_left = jnp.where(i > 0, Z[0, i - 1], semiring.one())
            Z_right = jnp.where(j < n - 1, Z[j + 1, n - 1], semiring.one())
            outer = semiring.div(semiring.mul(Z_left, semiring.mul(zp_ij, Z_right)), Z_total)

            # nearby (k, l) indices around (i, j)
            k_range = i + offsets
            l_range = j + offsets
            kl_k, kl_l = jnp.meshgrid(k_range, l_range, indexing="ij")
            k_flat = kl_k.ravel()
            l_flat = kl_l.ravel()

            # mask for valid and structurally nested (k < i, l > j) pairs
            in_bounds = (k_flat >= 0) & (k_flat < n) & (l_flat >= 0) & (l_flat < n)
            valid_structure = (k_flat < i) & (l_flat > j)
            mask = in_bounds & valid_structure

            P_kl = P[k_flat, l_flat]
            Z_inside = jnp.where(k_flat + 1 <= l_flat - 1, Z[k_flat + 1, l_flat - 1], semiring.one())
            non_zero = (~jnp.isclose(P_kl, semiring.zero(), atol=1e-6)) & (
                ~jnp.isclose(Z_inside, semiring.zero(), atol=1e-6))
            final_mask = mask & non_zero

            contrib = semiring.div(P_kl, Z_inside)
            contrib_masked = jnp.where(final_mask, contrib, semiring.zero())
            nested = semiring.mul(zp_ij, semiring.add_n(contrib_masked))

            return semiring.add(outer, nested)

        val = jax.lax.cond(should_compute, update_pair_with_local_nested_support, lambda: semiring.zero())
        return P.at[i, j].set(val), None

    # Initialize and run scan over all (i, j) pairs
    P_init = jnp.full((n, n), semiring.zero(), dtype=Z.dtype)
    indices = jnp.arange(i_all.shape[0])
    P_final, _ = lax.scan(compute_pair, P_init, indices)

    P_real = semiring.to_real_matrix(P_final)
    P_real = jnp.nan_to_num(P_real)
    P_real = jnp.clip(P_real, 0.0, 1.0)
    P_real = jnp.maximum(P_real, P_real.T)
    return P_real - jnp.diag(jnp.diag(P_real))


# ---------- derive other values from ensemble

def compute_marginal_probs(bpp: jnp.ndarray, paired=True) -> jnp.ndarray:
    """
    Compute the marginal probability that each base is paired (alternatively unpaired).
    """
    paired_probs = jnp.sum(bpp, axis=1)
    paired_probs = jnp.clip(paired_probs, 0.0, 1.0)
    return lax.cond(
        paired,
        lambda x: x,
        lambda x: 1.0 - x,
        paired_probs
    )


# ---------- get structure(s) from ensemble

def compute_mea_structure_plainpy(P: np.ndarray, h: int = 3, gamma: float = 1.0, threshold=1e-3) -> list[
    tuple[int, int]]:
    n = len(P)
    dp = np.zeros((n, n))
    trace = np.full((n, n), -1)  # -1: unpaired, otherwise store k

    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            if j - i < 1:
                continue

            # initialize with case j unpaired
            best = dp[i, j - 1] if j - 1 >= i else 0
            bt = -1

            # case j pairs with k
            for k in range(i, j - h + 1):
                if j - k >= h:
                    left = dp[i, k - 1] if k > i else 0
                    right = dp[k + 1, j - 1] if k + 1 <= j - 1 else 0

                    if P[k, j] < threshold:
                        continue  # skip numerically meaningless pairs

                    score = left + gamma * P[k, j] + right
                    if score > best:
                        best = score
                        bt = k

            # case split at s (i < s < j)
            for s in range(i + 1, j):
                if dp[i, s] + dp[s + 1, j] > best:
                    best = dp[i, s] + dp[s + 1, j]
                    bt = -2  # special split val

            dp[i, j] = best
            trace[i, j] = bt

    pairs = []

    def traceback(i, j):
        if i >= j:
            return
        k = trace[i, j]
        if k == -1:
            traceback(i, j - 1)
        elif k == -2:
            # find split point
            for s in range(i + 1, j):
                if np.isclose(dp[i, j], dp[i, s] + dp[s + 1, j]):
                    traceback(i, s)
                    traceback(s + 1, j)
                    return
        else:
            pairs.append((k, j))
            traceback(i, k - 1)
            traceback(k + 1, j - 1)

    traceback(0, n - 1)
    return pairs


def compute_mea_trace(P, h: int = 3, gamma: float = 1.0, threshold=1e-3):
    n = P.shape[0]
    dp = jnp.zeros((n, n), dtype=jfloat)
    trace = jnp.full((n, n), -1, dtype=jnp.int32)
    split = jnp.full((n, n), -1, dtype=jnp.int32)

    def span_loop(span, carry):
        dp, trace, split = carry

        def i_loop(i, carry2):
            dp, trace, split = carry2
            j = i + span

            best_score = jnp.where(j > i, dp[i, j - 1], 0.0)
            best_k = jnp.int32(-1)

            def k_body(k, state):
                best_score, best_k = state

                ok = (j - k >= h) & (P[k, j] > threshold)
                left = jnp.where(k > i, dp[i, k - 1], 0.0)
                right = jnp.where(k + 1 <= j - 1, dp[k + 1, j - 1], 0.0)
                score = left + gamma * P[k, j] + right
                score = jnp.where(ok, score, -jnp.inf)

                better = score > best_score
                best_score = jnp.where(better, score, best_score)
                best_k = jnp.where(better, k, best_k)
                return best_score, best_k

            best_score, best_k = lax.fori_loop(i, j - h + 1, k_body, (best_score, best_k))
            best_split = jnp.int32(-1)

            def s_body(s, state):
                best_score, best_k, best_split = state
                score = dp[i, s] + dp[s + 1, j]
                better = score > best_score
                best_score = jnp.where(better, score, best_score)
                best_k = jnp.where(better, jnp.int32(-2), best_k)
                best_split = jnp.where(better, s, best_split)
                return best_score, best_k, best_split

            best_score, best_k, best_split = lax.fori_loop(
                i + 1, j, s_body, (best_score, best_k, best_split)
            )

            dp = dp.at[i, j].set(best_score)
            trace = trace.at[i, j].set(best_k)
            split = split.at[i, j].set(best_split)
            return dp, trace, split

        return lax.fori_loop(0, n - span, i_loop, (dp, trace, split))

    _, trace, split = lax.fori_loop(1, n, span_loop, (dp, trace, split))
    return trace, split


def traceback_from_mea_trace(trace, split):
    n = trace.shape[0]
    max_pairs = 2 * n
    pairs = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
    pair_idx = jnp.int32(0)

    stack_i = jnp.zeros((max_pairs,), dtype=jnp.int32)
    stack_j = jnp.zeros((max_pairs,), dtype=jnp.int32)
    stack_i = stack_i.at[0].set(0)
    stack_j = stack_j.at[0].set(n - 1)
    stack_top = jnp.int32(1)

    def cond_fn(state):
        _, _, _, top = state
        return top > 0

    def body_fn(state):
        pairs, pair_idx, (stack_i, stack_j), stack_top = state
        stack_top -= 1
        i = stack_i[stack_top]
        j = stack_j[stack_top]

        def do_trace(s):
            pairs, pair_idx, (stack_i, stack_j), stack_top = s
            k = trace[i, j]

            def case_valid(ss):
                p, idx, (si, sj), top = ss
                p = p.at[idx].set(jnp.array([k, j]))
                si = si.at[top].set(i)
                sj = sj.at[top].set(k - 1)
                si = si.at[top + 1].set(k + 1)
                sj = sj.at[top + 1].set(j - 1)
                return p, idx + 1, (si, sj), top + 2

            def case_unpaired(ss):
                p, idx, (si, sj), top = ss
                si = si.at[top].set(i)
                sj = sj.at[top].set(j - 1)
                return p, idx, (si, sj), top + 1

            def case_split(ss):
                p, idx, (si, sj), top = ss
                s_val = split[i, j]
                si = si.at[top].set(i)
                sj = sj.at[top].set(s_val)
                si = si.at[top + 1].set(s_val + 1)
                sj = sj.at[top + 1].set(j)
                return p, idx, (si, sj), top + 2

            return lax.switch(k + 2, [case_split, case_unpaired, case_valid],
                              (pairs, pair_idx, (stack_i, stack_j), stack_top))

        return lax.cond(i >= j, lambda s: s, do_trace, (pairs, pair_idx, (stack_i, stack_j), stack_top))

    pairs, pair_idx, _, _ = lax.while_loop(cond_fn, body_fn, (pairs, pair_idx, (stack_i, stack_j), stack_top))
    return pairs, pair_idx


# ---------- entry point(s)

def compute_mea_structure(P: jnp.ndarray, gamma: float = 1.0, h: int = 3):
    trace, split = compute_mea_trace(P, h, gamma)
    pairs, pair_idx = traceback_from_mea_trace(trace, split)
    mask = jnp.arange(pairs.shape[0]) < pair_idx
    return pairs * mask[:, None]


def calc_base_pair_probs(Z, Z_p, seq, semiring):
    # return calc_base_pair_probs_n4(Z, Z_p, seq, semiring)
    # return calc_base_pair_probs_n4_fastad(Z, Z_p, seq, semiring)
    # return calc_base_pair_probs_n3(Z, Z_p, seq, semiring)
    return calc_base_pair_probs_n3_fastad(Z, Z_p, seq, semiring)
    # return calc_base_pair_probs_intimg_refine(Z, Z_p, seq, semiring, refine_steps=3, band=20)
    # return calc_base_pair_probs_windowed(Z, Z_p, seq, semiring)


def nussinov_pf(seq: jnp.ndarray, energy_mat: jnp.ndarray, semiring: Semiring, min_loop_length: int = 3):
    Z, Z_p = calc_partition_function(seq, energy_mat, semiring, min_loop_length)
    P = calc_base_pair_probs(Z, Z_p, seq, semiring)
    return Z, Z_p, P


SEMIRING = make_logsumexp_semiring()


def predict_structure(rna_seq: str, np_energy_mat: np.array, h=3, outdir: str = None):
    # print("x64 enabled:", config.read("jax_enable_x64"))
    # print(f"jfloat: {jfloat}")
    seq = encode_seq_jax(rna_seq)
    energy_mat = jnp.array(np_energy_mat)
    Z, Z_p, P = nussinov_pf(seq, energy_mat, SEMIRING, h)

    mea_pairs_untrimmed = compute_mea_structure(P, h)
    mea_pairs = [tuple(map(int, pair)) for pair in np.array(mea_pairs_untrimmed) if not (pair == 0).all()]

    if gc.VERBOSE:
        print("Z:")
        print_npmat(Z)
        print("Z_p:")
        print_npmat(Z_p)
        print("bpp:")
        print_npmat(P)
        unpaired = compute_marginal_probs(P, paired=False)
        print(f"Unpaired marginals: {unpaired}")

    if gc.DISPLAY:
        # bp_probability_dot_plot(P, seq, mea_pairs)
        # bp_probability_circle_plot(P, rna_seq)
        paired_probs = compute_marginal_probs(P, paired=True)
        plot_paired_unpaired_probs(paired_probs, rna_seq, outdir=outdir)

    vienna = pairing_to_vienna(rna_seq, mea_pairs)
    return vienna
