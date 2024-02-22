import numpy as np

import rnasl.gconst as gc
from rnasl.folding.nussinov_mfe import nussinov_mfe
from rnasl.folding_primitives.pairing_energies import PairingEnergies
from rnasl.folding_primitives.semiring import (NumpyLogSumExpSemiring, NumpyMinPlusSemiring, NumpySumProductSemiring,
                                               Semiring)
from rnasl.utils.formats import pairing_to_vienna
import rnasl.gconst as gc
from rnasl.visualization.visualize import (bp_probability_circle_plot, bp_probability_dot_plot,
                                           plot_paired_unpaired_probs, print_npmat)


def init_partition_tables(seq: str, semiring: Semiring):
    n = len(seq)
    Z = np.full((n, n), semiring.zero())
    Z_p = np.full((n, n), semiring.zero())
    for i in range(n):
        for j in range(n):
            if i > j:
                Z[i, j] = semiring.one()  # empty span
            elif i == j:
                Z[i, j] = semiring.one()  # single base (unpaired)
            elif j - i == 1:
                Z[i, j] = semiring.one()  # two unpaired bases
    return Z, Z_p


def calc_Z_paired_entry(i: int, j: int, seq: str, Z: np.ndarray, energies: PairingEnergies, semiring: Semiring,
        h: int) -> float:
    """Partition function over [i, j] where (i, j) is a base pair."""
    if j - i - 1 < h:
        return semiring.zero()
    interior = Z[i + 1, j - 1]
    pair_energy = semiring.encode(energies.paired(seq[i], seq[j]))
    return semiring.mul(pair_energy, interior)


def calc_Z_entry(i: int, j: int, seq: str, Z: np.ndarray, Z_p: np.ndarray, energies: PairingEnergies,
        semiring: Semiring, h: int) -> float:
    """Partition function over all structures in span [i, j]."""
    if j - i - 1 < h:
        return semiring.one()
    terms = []
    # j unpaired
    if i <= j - 1:
        terms.append(Z[i, j - 1])
    # j paired with k
    for k in range(i, j - h + 1):
        left = Z[i, k - 1] if k > i else semiring.one()
        terms.append(semiring.mul(left, Z_p[k, j]))
    return semiring.add_n(np.array(terms)) if terms else semiring.zero()


def calc_partition_function(seq: str, energies: PairingEnergies, semiring: Semiring, h: int = 3):
    n = len(seq)
    Z, Z_p = init_partition_tables(seq, semiring)

    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            Z_p[i, j] = calc_Z_paired_entry(i, j, seq, Z, energies, semiring, h)
            Z[i, j] = calc_Z_entry(i, j, seq, Z, Z_p, energies, semiring, h)
    return Z, Z_p


def calc_base_pair_probs(Z: np.ndarray, Z_p: np.ndarray, seq: str, semiring: Semiring) -> np.ndarray:
    """
    Compute base-pair probabilities using optimized n^3 inside-outside algorithm that uses helper table P_E.
    """
    n = len(seq)
    Z_total = Z[0, n - 1]
    if semiring.is_zero(Z_total):
        print("WARN: Z_total is zero as there are no valid structures.")
        return np.zeros((n, n))

    P = np.full((n, n), semiring.zero())

    # compute in decreasing span order so outer pairs (k,l) are filled before inner (i,j)
    for span in range(n, 0, -1):
        for i in range(n - span):
            j = i + span
            if semiring.is_zero(Z_p[i, j]):
                continue

            # outer contribs
            Z_left = Z[0, i - 1] if i > 0 else semiring.one()
            Z_right = Z[j + 1, n - 1] if j < n - 1 else semiring.one()
            outer = semiring.div(semiring.mul(Z_left, semiring.mul(Z_p[i, j], Z_right)), Z_total)

            # Eeclosed contribs using already computed P(k,l)
            enclosed_sum = semiring.zero()
            for k in range(i):
                for l in range(j + 1, n):
                    if semiring.is_zero(P[k, l]):
                        continue
                    Z_inside = Z[k + 1, l - 1]
                    if semiring.is_zero(Z_inside):
                        continue
                    enclosed_sum = semiring.add(enclosed_sum, semiring.div(P[k, l], Z_inside))

            nested = semiring.mul(Z_p[i, j], enclosed_sum)
            P[i, j] = semiring.add(outer, nested)

    P = semiring.to_real_matrix(P)
    P = np.nan_to_num(P)
    P = np.clip(P, 0.0, 1.0)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, P.T)
    return P


def calc_base_pair_probs_n3(Z, Z_p, seq, semiring):
    """ Faster BPP using prefix-sum trick.
    Avoids explicit summation over all enclosing (k, l) by reusing accumulated W = P / Z_inside
    values across iterations.
    O(n^3)
    """
    n = len(seq)
    one = semiring.one()
    zero = semiring.zero()
    Z_total = Z[0, n - 1]

    P = np.full((n, n), zero, dtype=Z.dtype)
    col_prefix = np.full(n, zero, dtype=Z.dtype)

    # Precompute Z_inside: Z[i+1, j-1] or 1
    Z_mid = np.full((n, n), one, dtype=Z.dtype)
    for k in range(n - 1):
        Z_mid[k, k + 2:] = Z[k + 1, k + 1:n - 1]

    for i in range(n - 1):
        row_cum = np.full(n, zero, dtype=Z.dtype)
        for j in range(n - 1, i, -1):
            zp_ij = Z_p[i, j]

            # Outer term
            Z_left = Z[0, i - 1] if i > 0 else one
            Z_right = Z[j + 1, n - 1] if j < n - 1 else one
            outer = semiring.div(semiring.mul(Z_left, semiring.mul(zp_ij, Z_right)), Z_total)

            # Nested term: reuse prefix sum of W[k,l] for k<i, l>j
            nested = semiring.mul(zp_ij, col_prefix[j])
            P_ij = semiring.add(outer, nested)
            P[i, j] = P_ij

            # Update W[i,j-1] = P[i,j] / Z[i+1,j-1] and accumul into row_cum
            if j - 1 > i:
                W_ij = semiring.div(P_ij, Z_mid[i, j])
                row_cum[j - 1] = semiring.add(W_ij, row_cum[j])

        # After proc. row i, add row_cum to col_prefix
        for l in range(n):
            col_prefix[l] = semiring.add(col_prefix[l], row_cum[l])

    P_real = semiring.to_real_matrix(P)
    P_real = np.nan_to_num(P_real)
    P_real = np.clip(P_real, 0.0, 1.0)
    P_real = np.maximum(P_real, P_real.T)
    np.fill_diagonal(P_real, 0.0)
    return P_real


# ---------- derive other values from ensemble

def compute_marginal_probs(bpp: np.ndarray, paired=True) -> np.ndarray:
    """
    Compute the marginal probability that each base is paired (/unpaired).
    """
    paired_probs = np.sum(bpp, axis=1)
    paired_probs = np.clip(paired_probs, 0.0, 1.0)
    if paired:
        return paired_probs
    return 1.0 - paired_probs


# ---------- get structure(s) from ensemble

def compute_mea_structure(P: np.ndarray, h: int = 3, gamma: float = 1.0, threshold=1e-3) -> list[tuple[int, int]]:
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


def nussinov_pf(seq: str, energies: PairingEnergies, semiring: Semiring, h: int = 3):
    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    P = calc_base_pair_probs_n3(Z, Z_p, seq, semiring)
    return Z, Z_p, P


# ---------- entry point

# SEMIRING = NumpySumProductSemiring()
SEMIRING = NumpyLogSumExpSemiring()


def predict_structure(seq: str, energies: PairingEnergies, h=3):
    Z, Z_p, P = nussinov_pf(seq, energies, SEMIRING, h)
    mea_pairs = compute_mea_structure(P, h, gamma=1.0)

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
        mfe_pairs, _ = nussinov_mfe(seq, energies, NumpyMinPlusSemiring(), h)
        bp_probability_dot_plot(P, seq, mfe_pairs, mea_pairs)
        # bp_probability_circle_plot(P, rna_seq)
        paired_probs = compute_marginal_probs(P, paired=True)
        plot_paired_unpaired_probs(paired_probs, seq)

    return pairing_to_vienna(seq, mea_pairs)
