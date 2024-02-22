import pytest
import numpy as np

from rnasl.folding.nussinov_mfe import nussinov_mfe
from rnasl.folding_primitives.semiring import Semiring, NumpyMinPlusSemiring
from rnasl.folding_primitives.pairing_energies import PairingEnergies


# ---- test helper functions

def get_321_pairing_energies():
    alphabet_list = ["A", "C", "G", "U"]
    energies = PairingEnergies(alphabet_list)
    energies.set("C", "G", -3)
    energies.set("A", "U", -2)
    energies.set("G", "U", -1)
    energies.set_noncanonical(10)
    return energies


def get_all0_pairing_energies():
    alphabet_list = ["A", "C", "G", "U"]
    energies = PairingEnergies(alphabet_list)
    return energies


def structure_energy(seq: str, pairs: list[tuple[int, int]], energies: PairingEnergies) -> float:
    energy = 0.0
    for i, j in pairs:
        energy += energies.paired(seq[i], seq[j])
    return energy


# ---- constants/global objects:

ENERGIES_321 = get_321_pairing_energies()
ENERGIES_ALL0 = get_all0_pairing_energies()

SEMIRING_MINPLUS = NumpyMinPlusSemiring()


# ---- Nussinov MFE tests

@pytest.mark.parametrize("h, expected_pairs", [(0, [(0, 3), (1, 2)]), (1, [(0, 3)]), (2, [(0, 3)]), (3, [])])
def test_mfe_pairing_minlooplen(h: int, expected_pairs: tuple[int, int]):
    seq = "GCGC"
    energies = ENERGIES_321
    semiring = SEMIRING_MINPLUS

    pairs, _ = nussinov_mfe(seq, energies, semiring, h)
    pairs = sorted(pairs)
    assert pairs == expected_pairs, f"Expected {expected_pairs}, got {pairs}"


def test_mfe_energy():
    seq = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
    energies = ENERGIES_321
    semiring = SEMIRING_MINPLUS
    h = 3

    pairs, dp = nussinov_mfe(seq, energies, semiring, h)
    pairs = sorted(pairs)
    mfe = dp[0, len(seq) - 1]
    mfe_expected = structure_energy(seq, pairs, energies)
    assert mfe == mfe_expected, f"Expected {mfe_expected}, got {mfe}"


def test_mfe_reverse_seq_energy_symmetry():
    seq = "GCGGAUUUAGCUCAGU"
    energies = ENERGIES_321
    semiring = SEMIRING_MINPLUS
    h = 3

    _, dp = nussinov_mfe(seq, energies, semiring, h)
    mfe = dp[0, len(seq) - 1]
    req_reverse = seq[::-1]
    _, dp_rev = nussinov_mfe(req_reverse, energies, semiring, h)
    mfe_rev = dp_rev[0, len(seq) - 1]
    assert np.isclose(mfe, mfe_rev), f"MFE energy from forward RNA seq {mfe}, MFE from backward seq {mfe_rev}"
