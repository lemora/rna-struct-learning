import numpy as np

import rnasl.gconst as gc
from rnasl.folding_primitives.pairing_energies import PairingEnergies
from rnasl.folding_primitives.semiring import NumpyMaxPlusSemiring, NumpyMinPlusSemiring, Semiring
from rnasl.utils.formats import pairing_to_vienna
from rnasl.visualization.visualize import print_npmat

# ---------- Nussinov things

def calc_dp_entry(i: int, j: int, dp: np.ndarray, seq: str, energies: PairingEnergies, semiring: Semiring, h: int = 3):
    if i >= j:
        return semiring.one()
    terms = []

    # j unpaired
    if i <= j - 1:
        terms.append(dp[i, j - 1])

    # j paired with k
    for k in range(i, j - h):
        pair_energy = energies.paired(seq[k], seq[j])
        left = dp[i, k - 1] if k > i else semiring.one()
        middle = dp[k + 1, j - 1] if k + 1 <= j - 1 else semiring.one()
        combined = semiring.mul(left, semiring.mul(middle, pair_energy))
        terms.append(combined)

    return semiring.add_n(np.array(terms)) if terms else semiring.zero()


def calc_dp(seq: str, energies: PairingEnergies, semiring: Semiring, h: int = 3):
    n = len(seq)
    dp = np.zeros((n, n), dtype=np.float32)
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            val = 0.0
            if abs(j - i) > h:
                val = calc_dp_entry(i, j, dp, seq, energies, semiring, h)
            dp[i, j] = val  # dp[j, i] = val
    return dp


def traceback(i: int, j: int, dp: np.ndarray, seq: str, energies: PairingEnergies, semiring: Semiring, h: int,
        pairs: list):
    if i > j:
        return

    # j unpaired
    if i <= j - 1 and semiring.equal(dp[i, j], dp[i, j - 1]):
        traceback(i, j - 1, dp, seq, energies, semiring, h, pairs)
        return

    # j is paired with k
    for k in range(i, j - h):
        pair_energy = energies.paired(seq[k], seq[j])

        left = dp[i, k - 1] if k > i else semiring.one()
        middle = dp[k + 1, j - 1] if k + 1 <= j - 1 else semiring.one()
        combined = semiring.mul(left, semiring.mul(middle, pair_energy))

        if k < j and semiring.equal(dp[i, j], combined):
            pairs.append((k, j))
            traceback(i, k - 1, dp, seq, energies, semiring, h, pairs)
            traceback(k + 1, j - 1, dp, seq, energies, semiring, h, pairs)
            return


def nussinov_mfe(seq: str, energies: PairingEnergies, semiring: Semiring, min_loop_length=3):
    n = len(seq)
    dp = calc_dp(seq, energies, semiring, min_loop_length)
    if gc.VERBOSE:
        print("dp:")
        print_npmat(dp)
        print("DP[0,n-1] = {}", dp[0, n - 1])
    pairs = []
    traceback(0, len(seq) - 1, dp, seq, energies, semiring, min_loop_length, pairs)
    return pairs, dp


# ---------- entry point

def predict_structure(rna_seq: str, pairing_energies: PairingEnergies, min_loop_len=3):
    semiring = NumpyMinPlusSemiring()
    # semiring = NumpyMaxPlusSemiring()
    pairs, dp = nussinov_mfe(rna_seq, pairing_energies, semiring, min_loop_len)
    return pairing_to_vienna(rna_seq, pairs)
