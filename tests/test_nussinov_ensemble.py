import pytest
import itertools
import numpy as np

from rnasl.folding.nussinov_pf import calc_partition_function, calc_base_pair_probs, compute_marginal_probs, \
    compute_mea_structure
from rnasl.folding_primitives.semiring import Semiring, NumpySumProductSemiring, NumpyMinPlusSemiring, \
    NumpyLogSumExpSemiring, SumProductSemiring
from rnasl.folding_primitives.pairing_energies import PairingEnergies
from rnasl.utils.helper import count_valid_structures, brute_force_bpp


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


# ---- constants/global objects:

ENERGIES_321 = get_321_pairing_energies()
ENERGIES_ALL0 = get_all0_pairing_energies()

SEMIRING_SUMPRODUCT = NumpySumProductSemiring()
SEMIRING_LOGSUMEXP = NumpyLogSumExpSemiring()


# ------ partition functions Z and Z_p

@pytest.mark.parametrize("h", [0, 1, 2, 3])
def test_partition_function_minlooplen(h):
    seq = "GCUA"
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, _ = calc_partition_function(seq, energies, semiring, h)
    expected = count_valid_structures(seq, h)
    assert np.isclose(Z[0, len(seq) - 1], expected), f"Z={Z[0, len(seq) - 1]}, expected={expected}"


def test_partition_function_empty_seq():
    seq = ""
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, _ = calc_partition_function(seq, energies, semiring, h=0)
    assert Z.shape == (0, 0)


def test_partition_function_min_loop_length():
    seq = "GCGA"
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z1, _ = calc_partition_function(seq, energies, semiring, h=0)
    Z2, _ = calc_partition_function(seq, energies, semiring, h=3)
    assert Z1[0, len(seq) - 1] > Z2[0, len(seq) - 1], "Loop length constraint not enforced"


@pytest.mark.parametrize("energies", [ENERGIES_ALL0, ENERGIES_321])
def test_partition_function_logspace_consistency(energies):
    seq = "GCUA"
    h = 0

    Z1, _ = calc_partition_function(seq, energies, SEMIRING_SUMPRODUCT, h)
    Z2_log, _ = calc_partition_function(seq, energies, SEMIRING_LOGSUMEXP, h)
    Z2 = np.exp(Z2_log[0, len(seq) - 1])

    assert np.isclose(Z1[0, len(seq) - 1], Z2, atol=1e-6)


def test_partition_tables_gcua():
    seq = "GCUA"
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT
    h = 0

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    assert Z[0, len(seq) - 1] == 9.0, f"Z = Z[0,3] = {Z[0, 3]}, expected 9.0"

    expected_zp_03 = semiring.encode(energies.paired("G", "A")) * Z[1, 2]
    actual_zp_03 = Z_p[0, 3]
    assert np.isclose(actual_zp_03, expected_zp_03), f"Z_p[0,3] = {actual_zp_03}, expected {expected_zp_03}"


@pytest.mark.parametrize("h, expected_pairs", [(0, [(0, 3), (1, 2)]), (1, [(0, 3)]), (2, [(0, 3)]), (3, [])])
def test_partition_function_base_pair_probs_seqlen4(h: int, expected_pairs: tuple[int, int]):
    seq = "GCUA"
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i, j in expected_pairs:
        assert bpp[i, j] > 0.01, f"Missing expected pair ({i},{j})"
    assert np.all((bpp >= 0.0) & (bpp <= 1.0)), "bpp values out of bounds"


@pytest.mark.parametrize("h, expected_pairs", [(3, [(0, 7), (1, 8), (2, 9), (3, 11), (13, 18), (14, 19)])])
def test_partition_function_base_pair_probs_seqlen20(h: int, expected_pairs: tuple[int, int]):
    seq = "GGGGCGUCCCGCUAAAUUUU"
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    cutoff = 0.005 * bpp[0, len(seq) - 1]
    print(bpp)
    print(cutoff)

    for i, j in expected_pairs:
        assert bpp[i, j] > cutoff, f"Missing expected pair ({i},{j})"
    assert np.all((bpp >= 0.0) & (bpp <= 1.0)), "bpp values out of bounds"


@pytest.mark.parametrize("h, expected_pairs", [(2, [(0, 8), (1, 7), (2, 5), (9, 16), (10, 15), (11, 14)])])
def test_mea_structure_predicts_expected_pairs(h: int, expected_pairs: list[tuple[int, int]]):
    seq = "GGGUUUUCCCCCCGGGG"
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    mea_pairs = compute_mea_structure(bpp, h=h, gamma=1.0)

    for i, j in expected_pairs:
        assert (i, j) in mea_pairs or (j, i) in mea_pairs, f"Missing expected pair ({i},{j})"

    assert np.all((bpp >= 0.0) & (bpp <= 1.0)), "bpp values out of bounds"


@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_partition_functions_consistency(semiring):
    """
    Test that Z_p[i, j] <= Z[i, j] for all i, j,
    and that Z_p[i, j] is zero for invalid pairs.
    """
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h=h)
    n = len(seq)

    for i in range(n):
        for j in range(i + h + 1, n):
            assert Z_p[i, j] <= Z[i, j], f"Z_p[{i},{j}] > Z[{i},{j}]"
            pairing_energy = energies.paired(seq[i], seq[j])
            if np.isinf(pairing_energy) or pairing_energy == semiring.zero():
                assert semiring.is_zero(Z_p[i, j]), f"Z_p[{i},{j}] > 0 for invalid pair"

        for j in range(i + 1, min(i + h + 1, n)):
            assert semiring.is_zero(Z_p[i, j]), f"Z_p[{i},{j}] should be zero due to minimum loop length"

    total_partition = Z[0, n - 1]
    max_Zp = np.max(Z_p)
    assert total_partition >= max_Zp, "Total partition function smaller than some Z_p entry"


def test_no_pairs_below_min_loop_length_in_Z():
    seq = "GCGCGC"
    energies = ENERGIES_321
    semiring = NumpySumProductSemiring()
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, min(i + h + 1, n)):
            assert Z[i, j] == semiring.one(), f"Z[{i},{j}] should be zero due to min loop length"


def test_no_pairs_below_min_loop_length_in_Z_p():
    seq = "GCGCGC"
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, min(i + h + 1, n)):
            assert Z_p[i, j] == semiring.zero(), f"Z_p[{i},{j}] should be zero due to min loop length"


@pytest.mark.parametrize("h", [1, 2, 3])
def test_loop_length_respected_in_Z_p(h):
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    for i in range(len(seq)):
        for j in range(i + 1, min(i + h + 1, len(seq))):
            assert Z_p[i, j] == 0.0, f"Z_p[{i},{j}] should be 0 due to loop length constraint"


# ------ base pair probabilities P

@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_partition_function_bpp_in_range(semiring):
    seq = "ACGU"
    energies = ENERGIES_321
    h = 1

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    pair_probs = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i in range(len(seq)):
        paired_prob = np.sum(pair_probs[i, :]) - pair_probs[i, i]
        assert 0.0 <= paired_prob <= 1.0


def test_bpp_sum_to_one():
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i in range(len(seq)):
        prob_sum = np.sum(bpp[i, :])
        assert prob_sum <= 1.0 + 1e-6, f"Base pair probabilities at position {i} sum to > 1"


def test_empty_structure_included_when_no_pairs_allowed():
    seq = "ACGU"
    h = 3
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    Z_total = Z[0, len(seq) - 1]

    assert Z_total == pytest.approx(1.0), (
        f"Partition function should be 1.0 when only the empty structure is allowed, got {Z_total}")

    assert np.all(Z_p == 0), "Z_p should be all zero when no base pairs are allowed"


def test_all_structures_disallowed_if_loop_too_strict():
    seq = "GGGA"
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)

    assert np.allclose(Z_p, 0.0), "No base pairs should be allowed with strict loop length"
    assert np.isclose(Z[0, len(seq) - 1], 1.0), "Only empty structure should be allowed"
    assert np.allclose(bpp, 0.0), "All base pair probabilities should be 0"


@pytest.mark.parametrize("seq", ["ACGU", "GGGAAAUUUCCC"])
@pytest.mark.parametrize("h", [0, 1, 3])
def test_partition_function_bpp_consistency_between_semirings(seq, h):
    energies = ENERGIES_321

    Z_sp, Z_p_sp = calc_partition_function(seq, energies, SEMIRING_SUMPRODUCT, h)
    bpp_sp = calc_base_pair_probs(Z_sp, Z_p_sp, seq, SEMIRING_SUMPRODUCT)

    Z_log, Z_p_log = calc_partition_function(seq, energies, SEMIRING_LOGSUMEXP, h)
    bpp_log = calc_base_pair_probs(Z_log, Z_p_log, seq, SEMIRING_LOGSUMEXP)

    assert np.allclose(bpp_sp, bpp_log, rtol=1e-4, atol=1e-5), "bpp matrices differ between semirings."


def test_bpp_paircount_not_over_max():
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h=3)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    total_bpp_mass = 0.5 * np.sum(bpp)
    max_paircount = len(seq) / 2.0
    assert total_bpp_mass <= max_paircount + 1e+6, "Total bpp mass should be ≤ 1.0"
    marginals = np.sum(bpp, axis=1)
    assert np.all(marginals <= 1.0 + 1e-6), f"Some bases have >1 pairing probability mass ({marginals})"


def test_long_sequence_has_many_pairs():
    seq = "GGGGGAAACCCCC"
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 2

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    likely_pairs = np.sum(np.triu(bpp, k=1))
    assert likely_pairs > 4, f"Expected >4 pairs in long sequence, got {likely_pairs:.2f}"


@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_compare_bpp_with_brute_forced(semiring):
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    brute_forced_bpp = brute_force_bpp(seq, energies, h)
    print("bpp:")
    print(bpp)
    print("brute forced:")
    print(brute_forced_bpp)
    assert np.allclose(bpp, brute_forced_bpp, rtol=1e-4, atol=1e-5), "DP and brute force bpp matrices differ."


def test_compare_paired_marginals_with_brute_forced():
    seq = "GGGAAAUUUCCC"
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    bpp_marginals = compute_marginal_probs(bpp)
    bf_bpp = brute_force_bpp(seq, energies, h)
    bpp_bf_marginals = compute_marginal_probs(bf_bpp)

    assert np.allclose(bpp_marginals, bpp_bf_marginals, rtol=1e-4, atol=1e-5), "Marginals differ from brute forced."
