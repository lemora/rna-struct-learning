import pytest
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from rnasl.folding.nussinov_pf_jax import calc_partition_function, calc_base_pair_probs, compute_marginal_probs, \
    compute_mea_structure
from rnasl.folding_primitives.semiring import Semiring, SumProductSemiring, make_logsumexp_semiring
from rnasl.folding_primitives.pairing_energies_jax import init_energy_mat, energy_mat_set_pair, \
    energy_mat_set_noncanonical
from rnasl.utils.formats import encode_seq_jax, BASE_TO_INT
from rnasl.utils.helper import count_valid_structures, brute_force_bpp_jax


# ---- test helper functions

def get_321_energy_matrix(alphabet=("A", "C", "G", "U")) -> jnp.ndarray:
    mat = init_energy_mat(4, 10.0)
    mat = energy_mat_set_pair(mat, BASE_TO_INT['C'], BASE_TO_INT['G'], -3)
    mat = energy_mat_set_pair(mat, BASE_TO_INT['A'], BASE_TO_INT['U'], -2)
    mat = energy_mat_set_pair(mat, BASE_TO_INT['G'], BASE_TO_INT['U'], -1)
    mat = energy_mat_set_noncanonical(mat, 10.0)
    return mat


def get_all0_energy_matrix(alphabet=("A", "C", "G", "U")) -> jnp.ndarray:
    mat = init_energy_mat(4, 0.0)
    return mat


# ---- constants/global objects:

ENERGIES_321 = get_321_energy_matrix()
ENERGIES_ALL0 = get_all0_energy_matrix()

SEMIRING_SUMPRODUCT = SumProductSemiring()
SEMIRING_LOGSUMEXP = make_logsumexp_semiring()


# ------ partition functions Z and Z_p

@pytest.mark.parametrize("h", [0, 1, 2, 3])
def test_partition_function_minlooplen(h):
    seq = encode_seq_jax("GCUA")
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, _ = calc_partition_function(seq, energies, semiring, h)
    expected = count_valid_structures(seq, h)
    assert np.isclose(Z[0, len(seq) - 1], expected), f"Z={Z[0, len(seq) - 1]}, expected={expected}"


def test_partition_function_empty_seq():
    seq = encode_seq_jax("")
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, _ = calc_partition_function(seq, energies, semiring, h=0)
    assert Z.shape == (0, 0)


def test_partition_function_min_loop_length():
    seq = encode_seq_jax("GCGA")
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z1, _ = calc_partition_function(seq, energies, semiring, h=0)
    Z2, _ = calc_partition_function(seq, energies, semiring, h=3)
    assert Z1[0, len(seq) - 1] > Z2[0, len(seq) - 1], "Loop length constraint not enforced"


@pytest.mark.parametrize("energies", [ENERGIES_ALL0, ENERGIES_321])
def test_partition_function_logspace_consistency(energies):
    seq = encode_seq_jax("GCUA")
    h = 0

    Z1, _ = calc_partition_function(seq, energies, SEMIRING_SUMPRODUCT, h)
    Z2_log, _ = calc_partition_function(seq, energies, SEMIRING_LOGSUMEXP, h)
    Z2 = np.exp(Z2_log[0, len(seq) - 1])

    assert np.isclose(Z1[0, len(seq) - 1], Z2, atol=1e-6)


def test_partition_tables_gcua():
    seq = encode_seq_jax("GCUA")
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT
    h = 0

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    assert Z[0, len(seq) - 1] == 9.0, f"Z = Z[0,3] = {Z[0, 3]}, expected 9.0"

    expected_zp_03 = semiring.encode(energies[BASE_TO_INT['G'], BASE_TO_INT['A']]) * Z[1, 2]
    actual_zp_03 = Z_p[0, 3]
    assert np.isclose(actual_zp_03, expected_zp_03), f"Z_p[0,3] = {actual_zp_03}, expected {expected_zp_03}"


@pytest.mark.parametrize("h, expected_pairs", [(0, [(0, 3), (1, 2)]), (1, [(0, 3)]), (2, [(0, 3)]), (3, [])])
def test_partition_function_base_pair_probs_seqlen4(h: int, expected_pairs: tuple[int, int]):
    seq = encode_seq_jax("GCUA")
    energies = ENERGIES_ALL0
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i, j in expected_pairs:
        assert bpp[i, j] > 0.01, f"Missing expected pair ({i},{j})"
    assert np.all((bpp >= 0.0) & (bpp <= 1.0)), "bpp values out of bounds"


@pytest.mark.parametrize("h, expected_pairs", [(3, [(0, 7), (1, 8), (2, 9), (3, 11), (13, 18), (14, 19)])])
def test_partition_function_base_pair_probs_seqlen20(h: int, expected_pairs: tuple[int, int]):
    seq = encode_seq_jax("GGGGCGUCCCGCUAAAUUUU")
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


@pytest.mark.parametrize("h, expected_pairs", [
    (2, [(0, 8), (1, 7), (2, 5), (9, 16), (10, 15), (11, 14)])
])
def test_mea_structure_predicts_expected_pairs(h: int, expected_pairs: list[tuple[int, int]]):
    seq = encode_seq_jax("GGGUUUUCCCCCCGGGG")
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    mea_pairs_untrimmed = compute_mea_structure(bpp, h=h, gamma=1.0)

    mea_pairs = [tuple(map(int, pair)) for pair in np.array(mea_pairs_untrimmed) if not (pair == 0).all()]
    for i, j in expected_pairs:
        assert (i, j) in mea_pairs or (j, i) in mea_pairs, f"Missing expected pair ({i},{j})"

    assert np.all((bpp >= 0.0) & (bpp <= 1.0)), "bpp values out of bounds"


@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_partition_functions_consistency(semiring):
    """
    Test that Z_p[i, j] <= Z[i, j] for all i, j,
    and that Z_p[i, j] is zero for invalid pairs.
    """
    seq = encode_seq_jax("GGGAAAUUUCCC")
    energies = ENERGIES_321
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h=h)
    n = len(seq)

    for i in range(n):
        for j in range(i + h + 1, n):
            assert Z_p[i, j] <= Z[i, j], f"Z_p[{i},{j}] > Z[{i},{j}]"
            pairing_energy = energies[seq[i], seq[j]]
            if np.isinf(pairing_energy) or pairing_energy == semiring.zero():
                assert semiring.is_zero(Z_p[i, j]), f"Z_p[{i},{j}] > 0 for invalid pair"

        for j in range(i + 1, min(i + h + 1, n)):
            assert semiring.is_zero(Z_p[i, j]), f"Z_p[{i},{j}] should be zero due to minimum loop length"

    total_partition = Z[0, n - 1]
    max_Zp = np.max(Z_p)
    assert total_partition >= max_Zp, "Total partition function smaller than some Z_p entry"


def test_no_pairs_below_min_loop_length_in_Z():
    seq = encode_seq_jax("GCGCGC")
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, min(i + h + 1, n)):
            assert Z[i, j] == semiring.one(), f"Z[{i},{j}] should be zero due to min loop length"


def test_no_pairs_below_min_loop_length_in_Z_p():
    seq = encode_seq_jax("GCGCGC")
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
    seq = encode_seq_jax("GGGAAAUUUCCC")
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    for i in range(len(seq)):
        for j in range(i + 1, min(i + h + 1, len(seq))):
            assert Z_p[i, j] == 0.0, f"Z_p[{i},{j}] should be 0 due to loop length constraint"


def test_partition_function_is_differentiable():
    seq = encode_seq_jax("GCUA")
    n = 4
    energy_mat = jnp.zeros((4, 4), dtype=jnp.float32)
    semiring = SEMIRING_LOGSUMEXP
    h = 0

    def wrapper(energy_mat):
        Z, _ = calc_partition_function(seq, energy_mat, semiring, h)
        return Z[0, len(seq) - 1]

    grad_fn = jax.grad(wrapper)
    grad_val = grad_fn(energy_mat)
    # print(f" grad val: {grad_val}")

    assert grad_val.shape == (4, 4), "Gradient shape mismatch"
    assert not jnp.any(jnp.isnan(grad_val)), "Gradient contains NaNs"
    assert not jnp.all(grad_val == 0), "Gradient is all zero (check implementation)"


# ------ base pair probabilities P

@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_partition_function_bpp_in_range(semiring):
    seq = encode_seq_jax("ACGU")
    energies = ENERGIES_321
    h = 1

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    pair_probs = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i in range(len(seq)):
        paired_prob = np.sum(pair_probs[i, :]) - pair_probs[i, i]
        assert 0.0 <= paired_prob <= 1.0


def test_bpp_sum_to_one():
    seq = encode_seq_jax("GGGAAAUUUCCC")
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)

    for i in range(len(seq)):
        prob_sum = np.sum(bpp[i, :])
        assert prob_sum <= 1.0 + 1e-6, f"Base pair probabilities at position {i} sum to > 1"


def test_empty_structure_included_when_no_pairs_allowed():
    seq = encode_seq_jax("ACGU")
    h = 3
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    Z_total = Z[0, len(seq) - 1]

    assert Z_total == pytest.approx(1.0), (
        f"Partition function should be 1.0 when only the empty structure is allowed, got {Z_total}")

    assert np.all(Z_p == 0), "Z_p should be all zero when no base pairs are allowed"


def test_all_structures_disallowed_if_loop_too_strict():
    seq = encode_seq_jax("GGGA")
    energies = ENERGIES_321
    semiring = SEMIRING_SUMPRODUCT
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, SEMIRING_SUMPRODUCT)

    assert np.allclose(Z_p, 0.0), "No base pairs should be allowed with strict loop length"
    assert np.isclose(Z[0, len(seq) - 1], 1.0), "Only empty structure should be allowed"
    assert np.allclose(bpp, 0.0), "All base pair probabilities should be 0"


@pytest.mark.parametrize("seq", ["ACGU", "GGGAAAUUUCCC"])
@pytest.mark.parametrize("h", [0, 1, 3])
def test_partition_function_bpp_consistency_between_semirings(seq, h):
    energies = ENERGIES_321
    seq = encode_seq_jax(seq)

    Z_sp, Z_p_sp = calc_partition_function(seq, energies, SEMIRING_SUMPRODUCT, h)
    bpp_sp = calc_base_pair_probs(Z_sp, Z_p_sp, seq, SEMIRING_SUMPRODUCT)

    Z_log, Z_p_log = calc_partition_function(seq, energies, SEMIRING_LOGSUMEXP, h)
    bpp_log = calc_base_pair_probs(Z_log, Z_p_log, seq, SEMIRING_LOGSUMEXP)

    assert np.allclose(bpp_sp, bpp_log, rtol=1e-4, atol=1e-5), "bpp matrices differ between semirings."


def test_bpp_paircount_not_over_max():
    seq = encode_seq_jax("GGGAAAUUUCCC")
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
    seq = encode_seq_jax("GGGGGAAACCCCC")
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 2

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    likely_pairs = np.sum(np.triu(bpp, k=1))
    assert likely_pairs > 4, f"Expected >4 pairs in long sequence, got {likely_pairs:.2f}"


@pytest.mark.parametrize("semiring", [SEMIRING_SUMPRODUCT, SEMIRING_LOGSUMEXP])
def test_compare_bpp_with_brute_forced(semiring):
    seq = encode_seq_jax("GGGAAAUUUCCC")
    energies = ENERGIES_321
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    brute_forced_bpp = brute_force_bpp_jax(seq, energies, h)
    print("bpp:")
    print(bpp)
    print("brute forced:")
    print(brute_forced_bpp)
    assert np.allclose(bpp, brute_forced_bpp, rtol=1e-4, atol=1e-5), "DP and brute force bpp matrices differ."


def test_compare_paired_marginals_with_brute_forced():
    seq = encode_seq_jax("GGGAAAUUUCCC")
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3

    Z, Z_p = calc_partition_function(seq, energies, semiring, h)
    bpp = calc_base_pair_probs(Z, Z_p, seq, semiring)
    bpp_marginals = compute_marginal_probs(bpp)
    bf_bpp = brute_force_bpp_jax(seq, energies, h)
    bpp_bf_marginals = compute_marginal_probs(bf_bpp)

    bpp_bf_marginals = bpp_bf_marginals.astype(bpp_marginals.dtype)

    assert np.allclose(bpp_marginals, bpp_bf_marginals, rtol=1e-4, atol=1e-4), "Marginals differ from brute forced."


@pytest.mark.parametrize("seq", ["ACGUACGU"])
def test_gradient_through_marginals(seq):
    seq = encode_seq_jax(seq)
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3
    vocab_size = 4

    def loss_fn(energy_mat):
        Z, Z_p = calc_partition_function(seq, energy_mat, semiring, h)
        P = calc_base_pair_probs(Z, Z_p, seq, semiring)
        unpaired_probs = compute_marginal_probs(P, paired=False)
        # Arbitrary loss: sum of unpaired probs (want fewer unpaired bases)
        return jnp.sum(unpaired_probs)

    grad_fn = jax.grad(loss_fn)
    grad_val = grad_fn(energies)
    print(f"gradient value: {grad_val}")

    assert grad_val.shape == (vocab_size, vocab_size)
    assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains NaNs or infs"
    assert jnp.any(grad_val != 0), "Gradient is all zeros — check connectivity"
    # assert False


def _loss_unpaired(seq, energy_mat, semiring, h):
    Z, Z_p = calc_partition_function(seq, energy_mat, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, seq, semiring)
    unpaired_probs = compute_marginal_probs(P, paired=False)
    return jnp.sum(unpaired_probs)


@pytest.mark.parametrize("seq_str", ["ACGUACGU"])
def test_gradients_marginals_finite(seq_str):
    seq = encode_seq_jax(seq_str)
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3

    loss_fn = lambda e: _loss_unpaired(seq, e, semiring, h)

    g = jax.grad(loss_fn)(energies)
    assert jnp.all(jnp.isfinite(g)), "NaN/inf in analytic grad"

    check_grads(loss_fn, (energies,), order=1, modes=["rev"], atol=1e-3, rtol=5e-2)


@pytest.mark.parametrize("seq_str", ["ACGUACGU"])
def test_gradients_pf_against_manual(seq_str):
    seq = encode_seq_jax(seq_str)
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3

    def _loss_pf(seq, energy_mat, semiring, h):
        Z, _ = calc_partition_function(seq, energy_mat, semiring, h)
        return jnp.sum(Z)

    loss_fn = lambda e: _loss_pf(seq, e, semiring, h)
    g_an = jax.grad(loss_fn)(energies)
    assert jnp.all(jnp.isfinite(g_an)), "Analytic grad has NaNs or infs"

    def fd_grad(e_mat):
        # finite difference with adaptive step (float64)
        fd = jnp.zeros_like(e_mat)
        for i in range(e_mat.shape[0]):
            for j in range(e_mat.shape[1]):
                step = 1e-2 * max(1.0, abs(float(e_mat[i, j])))
                bump = jnp.zeros_like(e_mat).at[i, j].set(step)
                fd_val = (loss_fn(e_mat + bump) - loss_fn(e_mat - bump)) / (2.0 * step)
                fd = fd.at[i, j].set(fd_val)
        return fd

    g_fd = fd_grad(energies)
    assert jnp.all(jnp.isfinite(g_fd)), "FD grad has NaNs or infs"

    assert jnp.allclose(g_an, g_fd, rtol=3e-3, atol=1e-4), (
        f"Analytic vs FD mismatch\nanalytic:\n{g_an}\nfinite-diff:\n{g_fd}"
    )

    check_grads(loss_fn, (energies,), modes=["rev"], order=1, rtol=3e-3, atol=1e-4)


@pytest.mark.parametrize("seq_str", ["ACGUACGU"])
def test_gradients_marginals_against_manual(seq_str):
    seq = encode_seq_jax(seq_str)
    energies = ENERGIES_321
    semiring = SEMIRING_LOGSUMEXP
    h = 3

    loss_fn = lambda e: _loss_unpaired(seq, e, semiring, h)
    grad_an = jax.grad(loss_fn)(energies)
    assert jnp.all(jnp.isfinite(grad_an))

    eps = 1e-2
    fd = jnp.zeros_like(energies)

    for i in range(energies.shape[0]):
        for j in range(energies.shape[1]):
            dE = jnp.zeros_like(energies).at[i, j].set(eps)
            fd = fd.at[i, j].set(
                (loss_fn(energies + dE) - loss_fn(energies - dE)) / (2 * eps)
            )

    assert jnp.all(jnp.isfinite(fd)), "NaN/inf in FD grad"

    diff = jnp.abs(grad_an - fd)
    mask_big = diff > (1e-3 + 0.05 * jnp.abs(fd))
    print("Strong mismatches:")
    print(jnp.where(mask_big, diff, 0.0))
    print("Analytic grad:")
    print(grad_an)
    print("FD grad:")
    print(fd)

    assert jnp.allclose(grad_an, fd, rtol=5e-2, atol=1e-3), (
        f"analytic:\n{grad_an}\nfd:\n{fd}"
    )
