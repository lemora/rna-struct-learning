from rnasl.jax_setup import jfloat

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from rnasl.folding.nussinov_pf_jax import calc_base_pair_probs, calc_partition_function, compute_marginal_probs
from rnasl.folding_primitives.semiring import make_logsumexp_semiring
from rnasl.io.experiment_io import energy_mat_from_file

from rnasl.utils.formats import CANONICAL_PAIRS, encode_seq_jax
import rnasl.gconst as gc


# ------ visualisation

def plot_loss_surface_slice(losses: jnp.ndarray, alphas: jnp.ndarray, slice_direction: str, outdir: str,
        outname: str = "loss_slice"):
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, losses)
    # plt.title(f"Loss Slice: {slice_direction}")
    plt.xlabel("Perturbation α")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    outname = f"{outname}_{slice_direction.replace(' ', '')}.png"
    if outdir:
        filepath = os.path.join(outdir, outname)
        plt.savefig(filepath)
        print(f"Saved loss slice to {filepath}")

    plt.show()
    plt.close()


# ------ test set eval metrics + helpers

def masked_mse(y_pred, y_true):
    # clip to match training + model output range
    y_true = jnp.clip(y_true, 0.0, 1.0)

    mask = jnp.isfinite(y_true)
    diff = jnp.where(mask, y_pred - y_true, 0.0)
    return jnp.sum(diff ** 2) / jnp.maximum(jnp.sum(mask), 1)


def masked_pearson_corr(y_pred, y_true):
    # clip to match training + model output range
    y_true = jnp.clip(y_true, 0.0, 1.0)

    mask = jnp.isfinite(y_true)
    if jnp.sum(mask) < 2:
        return jnp.nan

    y_pred = jnp.where(mask, y_pred, 0.0)
    y_true = jnp.where(mask, y_true, 0.0)

    pred_mean = jnp.sum(y_pred) / jnp.sum(mask)
    true_mean = jnp.sum(y_true) / jnp.sum(mask)

    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean

    numerator = jnp.sum(pred_centered * true_centered)
    denominator = jnp.sqrt(jnp.sum(pred_centered ** 2) * jnp.sum(true_centered ** 2))

    return jnp.where(denominator > 0, numerator / denominator, 0.0)


# ------ loss landscape eval

def make_loss_fn(seq, target, semiring, h):
    def loss_fn(energy_mat):
        Z, Z_p = calc_partition_function(seq, energy_mat, semiring, h)
        P = calc_base_pair_probs(Z, Z_p, seq, semiring)
        unpaired_probs = compute_marginal_probs(P, paired=False)
        return masked_mse(unpaired_probs, target)

    return loss_fn


def get_default_directions(shape):
    directions = {}
    gc_dir = jnp.zeros(shape).at[2, 1].set(1.0).at[1, 2].set(1.0)
    directions["GC"] = gc_dir
    # canonical vs noncanon
    canon_vs_noncanon = jnp.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            canon_vs_noncanon = canon_vs_noncanon.at[i, j].set(
                1.0 if (i, j) in CANONICAL_PAIRS else -1.0
            )
    directions["Canonical vs Noncanonical"] = canon_vs_noncanon
    return directions


def generate_all_pair_directions(shape=(4, 4)):
    directions = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            name = f"{'ACGU'[i]}-{'ACGU'[j]}"
            dir_mat = jnp.zeros(shape).at[i, j].set(1.0)
            if i != j:
                dir_mat = dir_mat.at[j, i].set(1.0)
            directions[name] = dir_mat
    return directions


def loss_surface_slice(loss_fn, energies, direction, alpha_range=jnp.linspace(-2, 2, 100)):
    losses = []
    for alpha in alpha_range:
        perturbed = energies + alpha * direction
        loss = loss_fn(perturbed)
        losses.append(loss)
    return jnp.array(losses), alpha_range


def analyse_loss_landscape(energy_mat_path: str, h: int = 3, directions=None, alpha_range=jnp.linspace(-5.0, 5.0, 100)):
    energy_mat = energy_mat_from_file(energy_mat_path)
    print(f"Energy mat:")
    print(energy_mat)

    semiring = make_logsumexp_semiring()
    example_seq = "AUCGGCAUAGCUAGCUUAGCGACGUAGCUACACGUU"
    encoded_seq = encode_seq_jax(example_seq)
    rval = 0.12  # mean: generic/neutral loss surface
    target = jnp.full((len(encoded_seq),), rval, dtype=jfloat)
    loss_fn = make_loss_fn(encoded_seq, target, semiring, h)

    if gc.VERBOSE:
        loss_val = loss_fn(energy_mat)
        print("Loss:", loss_val)
        grads = jax.grad(loss_fn)(energy_mat)
        print("Grad:", grads)

    if directions is None:
        directions = get_default_directions(energy_mat.shape)

    outdir = os.path.dirname(energy_mat_path)
    for name, direction in directions.items():
        print(f"Scanning loss slice: {name}")
        losses, alphas = loss_surface_slice(loss_fn, energy_mat, direction, alpha_range)
        plot_loss_surface_slice(losses, alphas, name, outdir)
