from rnasl.jax_setup import jfloat, get_preferred_device

import csv
import ctypes
import functools
import os
import random
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import lax

from rnasl.folding.nussinov_pf_jax import calc_base_pair_probs, calc_partition_function, compute_marginal_probs
from rnasl.folding.extended_nussinov_pf_jax import calc_extended_pseudo_partition_function
from rnasl.folding_primitives.pairing_energies_jax import init_energy_mat, init_random_energies
from rnasl.folding_primitives.semiring import Semiring, SumProductSemiring, make_logsumexp_semiring
from rnasl.io.experiment_io import save_config_file, save_energies, save_training_outputs
from rnasl.training.loss_train_configs import LossConfig, TrainConfig
import rnasl.gconst as gc

from rnasl.utils.formats import CANONICAL_PAIRS, CANONICAL_MASK, NONCANONICAL_MASK, BASE_TO_INT, encode_seq_jax


# ---- loss functions galore

def compute_helix_loss(P: jnp.ndarray, min_len: int = 3) -> jnp.ndarray:
    n = P.shape[0]
    i = jnp.arange(n - min_len)
    j = jnp.arange(min_len + 1, n)
    ii, jj = jnp.meshgrid(i, j, indexing="ij")

    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    def stem_score(i, j):
        ks = jnp.arange(min_len)
        rows = i + ks
        cols = j - ks
        valid = rows < cols
        vals = P[rows, cols]
        return jnp.where(jnp.all(valid), jnp.mean(vals), 0.0)

    stem_score_vmap = jax.vmap(stem_score)
    scores = stem_score_vmap(ii_flat, jj_flat)

    total = jnp.sum(scores)
    count = jnp.sum(scores > 0)
    return -total / jnp.maximum(count, 1.0)


def pairing_usage_prior(P, seq):
    canon_pair_mask = CANONICAL_MASK[seq[:, None], seq[None, :]] & jnp.triu(jnp.ones_like(P, dtype=bool), k=0)
    vals = P * canon_pair_mask
    usage_score = jnp.sum(vals) / jnp.maximum(jnp.sum(canon_pair_mask), 1)
    amplify = seq.shape[0]
    return -amplify * usage_score


def calibrate(pred: jnp.ndarray, tgt: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Linear rescaling of predicted to better match scale of target (least-squares fit).
    Finds a, b such that a*pred + b ~= tgt over masked entries
    """
    w = mask.astype(pred.dtype)
    x = pred * w
    y = tgt * w
    n = jnp.maximum(jnp.sum(w), 1.0)

    x_mean = jnp.sum(x) / n
    y_mean = jnp.sum(y) / n

    x_dev = (x - x_mean * w)
    y_dev = (y - y_mean * w)

    # compute covariance between pred & target, and variance of pred
    num = jnp.sum(x_dev * y_dev)
    denom = jnp.sum(x_dev * x_dev)
    # slope a = covariance / variance, intercept b = y_mean - a*x_mean
    a = jnp.where(denom > 0, num / denom, 1.0)
    b = y_mean - a * x_mean

    return a * pred + b


def loss_with_structure_prior(
        predicted: jnp.ndarray,
        target_reactivities: jnp.ndarray,
        energies: jnp.ndarray,
        lossconf: LossConfig,
        P: jnp.ndarray,
        seq: jnp.ndarray,
) -> jnp.ndarray:
    """ Calculates MSE + L2 + noncanon + entropy + mag (depending on config).
    Masks NaN in target, clips target to [0,1], downweights clipped. """

    # --- Preprocess reactivity target and weights
    if lossconf.clip_target:
        target = jnp.clip(target_reactivities, lossconf.eps, 1.0 - lossconf.eps)
        was_clipped = target_reactivities != target
    else:
        target = target_reactivities
        was_clipped = jnp.zeros_like(target, dtype=bool)

    predicted = jnp.clip(predicted, lossconf.eps, 1.0 - lossconf.eps)

    mask = jnp.isfinite(target)
    weights = jnp.where(mask, 1.0, 0.0)
    weights = jnp.where(was_clipped & mask, lossconf.downweight_clipped, weights)

    # --- Calibration
    pred_calibrated = predicted
    # if lossconf.rescale:
    #     pred_calibrated = calibrate(predicted, target, mask)

    # --- Central MSE loss
    diff = jnp.where(mask, pred_calibrated - target, 0.0)
    mse_loss = jnp.sum(weights * diff ** 2) / jnp.maximum(jnp.sum(weights), 1.0)

    # upper right energy triangle (all unique base pairs)
    upper = jnp.triu(jnp.ones_like(energies, dtype=bool), k=0)

    # ---- L2 reg
    l2 = 0.0
    if lossconf.l2_reg != 0.0:
        l2 = jnp.sum(jnp.where(upper, energies ** 2, 0.0))
        l2 = lossconf.l2_reg * jnp.nan_to_num(l2)

    # ---- Entropy penalty
    entropy_penalty = 0.0
    if lossconf.entropy_weight != 0.0:
        # 0.5 = highest entropy -> higher loss. close to 0 ot 1 encouraged
        pred_safe = jnp.clip(predicted, 1e-4, 1.0)
        entropy_penalty = -jnp.sum(pred_safe * jnp.log(pred_safe)) / jnp.maximum(jnp.sum(mask), 1.0)
        entropy_penalty = lossconf.entropy_weight * entropy_penalty

    # --- Magnitude alignment penalty
    magnitude_penalty = 0.0
    if lossconf.mag_weight != 0.0:
        avg_reactivity = jnp.nanmean(target)
        avg_pred = jnp.nanmean(predicted)
        magnitude_penalty = (avg_pred - avg_reactivity) ** 2
        magnitude_penalty = lossconf.mag_weight * magnitude_penalty

    # ---- Non-canonical penalty
    noncanon_penalty = 0.0
    if lossconf.noncanon_weight != 0.0:
        mask = NONCANONICAL_MASK & upper
        vals = jnp.where(mask, -energies, 0.0)
        relu_vals = 10 * jax.nn.relu(vals)
        penalty = jnp.sum(relu_vals + relu_vals ** 2) / jnp.maximum(jnp.sum(mask), 1)
        noncanon_penalty = lossconf.noncanon_weight * penalty

    # ---- Canonical penalty
    canon_penalty = 0.0
    if lossconf.canon_weight != 0.0:
        mask = CANONICAL_MASK & upper
        vals = jnp.where(mask, energies, 0.0)
        relu_vals = 10 * jax.nn.relu(vals)
        penalty = jnp.sum(relu_vals + relu_vals ** 2) / jnp.maximum(jnp.sum(mask), 1)
        canon_penalty = lossconf.canon_weight * penalty

    # ---- canonical usage reward
    canon_usage_reward = 0.0
    if lossconf.canon_usage_weight != 0.0:
        canon_usage_reward = pairing_usage_prior(P, seq)
        canon_usage_reward = lossconf.canon_usage_weight * canon_usage_reward

    # ---- helix penalty
    helix_reward = 0.0
    if lossconf.helix_weight != 0.0:
        helix_reward = compute_helix_loss(P)
        helix_reward = lossconf.helix_weight * helix_reward

    loss_terms = {
        "mse": mse_loss,
        "l2": l2,
        "entropy": entropy_penalty,
        "mag": magnitude_penalty,
        "canon": canon_penalty,
        "noncanon": noncanon_penalty,
        "canon_usage": canon_usage_reward,
        "helix": helix_reward,
    }

    total_loss = (
            loss_terms["mse"] +
            loss_terms["l2"] +
            loss_terms["entropy"] +
            loss_terms["mag"] +
            loss_terms["canon"] +
            loss_terms["noncanon"] +
            loss_terms["canon_usage"] +
            loss_terms["helix"]
    )

    return total_loss, loss_terms


def loss_mse_mask_clip_downweight(
        predicted: jnp.ndarray,
        target: jnp.ndarray,
        energies: jnp.ndarray,
        lossconf: LossConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """ Calculates MSE + L2, masks NaN in target, clips target to [0,1], downweights clipped. """

    if lossconf.clip_target:
        tgt_clipped = jnp.clip(target, lossconf.eps, 1.0 - lossconf.eps)
        was_clipped = target != tgt_clipped
        target = tgt_clipped
    else:
        was_clipped = jnp.zeros_like(target, dtype=bool)

    mask = jnp.isfinite(target)
    weights = jnp.where(mask, 1.0, 0.0)
    weights = jnp.where(was_clipped & mask, lossconf.downweight_clipped, weights)

    predicted = jnp.clip(predicted, lossconf.eps, 1.0 - lossconf.eps)
    diff = jnp.where(mask, predicted - target, 0.0)
    denom = jnp.sum(weights)
    mse_loss = jnp.where(denom > 0, jnp.sum(weights * diff ** 2) / denom, 0.0)

    l2 = lossconf.l2_reg * jnp.sum(jnp.square(energies))
    total_loss = mse_loss + l2

    loss_terms = {
        "mse": mse_loss,
        "l2": l2,
        "noncanon": 0.0,
        "entropy": 0.0,
        "mag": 0.0,
    }

    return total_loss, loss_terms


def loss_mse_masked(
        unpaired_probs: jnp.ndarray,
        target: jnp.ndarray,
        energies: jnp.ndarray,
        lossconf: LossConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """ Calculates MSE + L2, masks NaN in target. """

    mask = jnp.isfinite(target)
    diff = jnp.where(mask, unpaired_probs - target, 0.0)

    count = jnp.sum(mask)
    masked_mse = jnp.where(count > 0, jnp.sum(diff ** 2) / count, 0.0)

    l2 = lossconf.l2_reg * jnp.sum(energies ** 2)
    total_loss = masked_mse + l2

    loss_terms = {
        "mse": masked_mse,
        "l2": l2,
        "noncanon": 0.0,
        "entropy": 0.0,
        "mag": 0.0,
    }

    return total_loss, loss_terms


def symmetrise_maxabs(mat):
    i, j = jnp.triu_indices(mat.shape[0], k=1)
    c = jnp.where(jnp.abs(mat[i, j]) >= jnp.abs(mat[j, i]), mat[i, j], mat[j, i])
    return mat.at[i, j].set(c).at[j, i].set(c)


# ------ training code

@partial(jax.jit, static_argnames=("semiring", "trainconf", "lossconf"))
def compute_loss(energies, seq, target, semiring, trainconf: TrainConfig, lossconf: LossConfig):
    Z, Z_p = lax.cond(
        trainconf.loop_pen,
        lambda _: calc_extended_pseudo_partition_function(seq, energies, semiring, trainconf.h),
        lambda _: calc_partition_function(seq, energies, semiring, trainconf.h),
        operand=None
    )
    P = calc_base_pair_probs(Z, Z_p, seq, semiring)

    unpaired_probs = compute_marginal_probs(P, paired=False)
    total_loss, loss_terms = loss_with_structure_prior(unpaired_probs, target, energies, lossconf, P, seq)
    return total_loss, loss_terms


@functools.lru_cache(maxsize=64)
def get_update_step(seq_len: int):
    print(f"Compiling update_step for sequence length {seq_len}")

    def bind(semiring, optimizer, lossconf: LossConfig, trainconf: TrainConfig):
        @jax.jit
        def update_step(energies, opt_state, seq, target):
            def loss_fn(e):
                loss, terms = compute_loss(e, seq, target, semiring, trainconf, lossconf)
                return loss, terms

            if gc.VERBOSE:
                jax.debug.print("energies:\n{}", energies)

            (total_loss, loss_terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(energies)

            grads = jnp.nan_to_num(grads, nan=0.0, posinf=1e6, neginf=-1e6)

            if gc.VERBOSE:
                jax.debug.print("grads:\n{}", grads)

            if trainconf.freeze_nc:
                grads = grads * CANONICAL_MASK

            updates, opt_state = optimizer.update(grads, opt_state)
            new_energies = optax.apply_updates(energies, updates)

            if gc.VERBOSE:
                jax.debug.print("new energies:\n{}", new_energies)

            # symmetrise energies
            # new_energies = (new_energies + new_energies.T) / 2.0

            # symm energies by copying upper tri to lower (as lower is never touched anyway)
            new_energies = jnp.triu(new_energies) + jnp.triu(new_energies, k=1).T

            if gc.VERBOSE:
                jax.debug.print("symm energies:\n{}", new_energies)

            return new_energies, opt_state, total_loss, loss_terms

        return update_step

    return bind


# ---- Trainer Class

class RNATrainer:
    def __init__(self, training_data_stream, semiring: Semiring, init_energies_mat: jnp.ndarray, trainconf: TrainConfig,
            lossconf: LossConfig, validation_data_stream=None, outdir="out"):
        self.training_data_stream = training_data_stream
        self.validation_data_stream = validation_data_stream
        self.outdir = outdir
        self.semiring = semiring
        self.energies = init_energies_mat

        if gc.VERBOSE:
            jax.debug.print("initial energies:\n{}", init_energies_mat)

        # initial energy mat symm
        self.energies = (self.energies + self.energies.T) / 2.0

        if gc.VERBOSE:
            jax.debug.print("symm energies:\n{}", self.energies)

        self.lossconf = lossconf
        self.trainconf = trainconf

        # self.optimizer = optax.rmsprop(lr)
        self.optimizer = optax.adam(trainconf.lr)
        self.opt_state = self.optimizer.init(self.energies)

    def train(self):
        loss_keys = ["total", "mse", "l2", "entropy", "mag", "canon", "noncanon", "canon_usage", "helix"]
        train_logs = {k: [] for k in loss_keys}
        val_logs = {k: [] for k in loss_keys}
        energy_history = []

        log_filename = "training_log.csv"
        self._init_log_file(loss_keys, filename=log_filename)

        best_val_loss = float("inf")
        best_energies = self.energies
        patience_ctr = 0

        for step in range(self.trainconf.steps):
            step_loss = {k: 0.0 for k in loss_keys}
            example_count = 0

            # ---------- Training
            for seq, target in self.training_data_stream:
                update_fn = get_update_step(seq.shape[0])(
                    self.semiring, self.optimizer, self.lossconf, self.trainconf
                )
                self.energies, self.opt_state, tot_loss, loss_terms = update_fn(
                    self.energies, self.opt_state, seq, target
                )

                step_loss["total"] += float(tot_loss)
                for k in loss_keys[1:]:
                    step_loss[k] += float(loss_terms.get(k, 0.0))

                example_count += 1
                if example_count >= self.trainconf.examples_per_step:
                    break

            if example_count == 0:
                break

            for k in loss_keys:
                step_loss[k] /= example_count
                train_logs[k].append(step_loss[k])

            if gc.VERBOSE:
                print(f"train loss terms:\n{step_loss}")

            self._append_log_row(step, "train", step_loss, filename=log_filename)

            # ---------- Validation
            if self.validation_data_stream is not None:
                val_record = self.eval_validation_loss(step, loss_keys, filename=log_filename)
                for k in loss_keys:
                    val_logs[k].append(val_record[k])

                print(f"[step {step + 1}] train loss {step_loss['total']:.4f} | "
                      f"val loss {val_record['total']:.4f}")

                if val_record["total"] < best_val_loss:
                    best_val_loss = val_record["total"]
                    best_energies = self.energies
                    patience_ctr = 0
                elif step >= self.trainconf.patience_warmup:
                    # else:
                    patience_ctr += 1
                    print(f"No improvement. Patience {patience_ctr}/{self.trainconf.patience}")
                    if patience_ctr >= self.trainconf.patience:
                        print("Early stopping triggered.")
                        break
                if step < self.trainconf.patience_warmup:
                    print(f"Warmup: {step + 1}/{self.trainconf.patience_warmup}")

            energy_history.append(jax.device_get(jnp.array(self.energies)))

        return best_energies, train_logs, val_logs, energy_history

    def eval_validation_loss(self, step: int, loss_keys: list[str], filename: str) -> dict[str, float]:
        """Run one validation pass and log it. Returns dict of averaged losses."""
        val_loss_total = 0.0
        val_loss_terms = {k: 0.0 for k in loss_keys}
        val_count = 0

        for seq, target in self.validation_data_stream:
            loss, loss_terms = compute_loss(
                self.energies, seq, target,
                self.semiring, self.trainconf, self.lossconf
            )
            val_loss_total += float(loss)
            for k in loss_keys[1:]:
                val_loss_terms[k] += float(loss_terms.get(k, 0.0))
            val_count += 1
            if val_count >= self.trainconf.val_examples_per_step:
                break

        val_record = {"total": val_loss_total / val_count}
        for k in loss_keys[1:]:
            val_record[k] = val_loss_terms[k] / val_count

        # print(f"val loss: {val_record}")

        self._append_log_row(step, "val", val_record, filename=filename)
        return val_record

    # ------ writing to file

    def _init_log_file(self, loss_keys: list[str], filename="training_log.csv") -> None:
        outpath = os.path.join(self.outdir, filename)
        os.makedirs(self.outdir, exist_ok=True)
        with open(outpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "split"] + loss_keys)
            writer.writeheader()

    def _append_log_row(self, step: int, split: str, log_dict: dict[str, float], filename="training_log.csv") -> None:
        outpath = os.path.join(self.outdir, filename)
        row = {"step": step, "split": split, **log_dict}
        with open(outpath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)


# ------ Data Loading

def load_dataset(filepath: str, val_count=None) -> list:
    dataset = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        reactivity_cols = [col for col in reader.fieldnames if col.startswith('reactivity_0')]

        for i, row in enumerate(reader):
            try:

                if val_count and i > val_count:
                    break
                seq = row['sequence'].strip()
                if not seq:
                    continue

                SN_filter = float(row.get('SN_filter', '1.0').strip())
                if SN_filter < 1:
                    continue

                encoded_seq = encode_seq_jax(seq)
                reactivities_raw = [row[col].strip() for col in reactivity_cols[:len(seq)]]
                reactivities = jnp.array([
                    float(v) if v and v.lower() != "null" else -jnp.nan
                    for v in reactivities_raw
                ], dtype=jfloat)

                if len(encoded_seq) != len(reactivities):
                    continue

                # print(f"seq: {seq}")
                # print(f"encoded seq: {encoded_seq}")
                # print(f"react: {reactivities}")
                dataset.append((encoded_seq, reactivities))
            except Exception:
                continue
    return dataset


def data_stream(dataset):
    while True:
        random.shuffle(dataset)
        for seq, target in dataset:
            yield seq, target


# ------ training

def check_cuda():
    try:
        ctypes.CDLL("libcuda.so")
        print("libcuda.so found: NVIDIA driver is installed.")
    except OSError as e:
        print("libcuda.so not found. CUDA likely unavailable.")
        print("Error:", e)

    print("JAX backend platform:", jax.default_backend())
    print("JAX devices:", jax.devices())
    device = get_preferred_device()
    print()


def train(datafile_train: str,
        datafile_validate: str,
        outdir: str,
        trainconf: TrainConfig,
        lossconf: LossConfig):
    check_cuda()
    # jax.config.update("jax_disable_jit", True)

    print("================================")
    print("---- Training Configuration ----")
    print(f"Data file: {datafile_train}")
    print(f"Validation file: {datafile_validate}")
    print(trainconf)
    print(lossconf)
    print("================================")

    # dataset_type = "DMS" if "/DMS/" in datafile_train else "2A3"

    rand_seed = int(time.time() * 1e6) % (2 ** 32 - 1)
    init_energies = init_random_energies(4, rand_seed)

    #init_energies = init_energy_mat(4, 0.0)
    if trainconf.freeze_nc:
        init_energies = init_energies * CANONICAL_MASK + 1.0 * NONCANONICAL_MASK

    print("Initial 4x4 energies:")
    print(init_energies)

    limit_loaded_data = None
    # norm_by = dataset_type if trainconf.normalise_data else None
    training_data = load_dataset(datafile_train, limit_loaded_data)
    validation_data = load_dataset(datafile_validate, limit_loaded_data) if datafile_validate else None

    training_data_stream = data_stream(training_data)
    validation_data_stream = data_stream(validation_data) if validation_data else None

    print("\nData loaded and shuffled.")
    start = time.time()

    semiring = make_logsumexp_semiring()
    trainer = RNATrainer(
        training_data_stream=training_data_stream,
        semiring=semiring,
        init_energies_mat=init_energies,
        trainconf=trainconf,
        lossconf=lossconf,
        validation_data_stream=validation_data_stream,
        outdir=outdir
    )

    learned_energies, train_losses, val_losses, energy_history = trainer.train()
    learned_energies = jax.lax.stop_gradient(learned_energies)
    trained_params_host = jax.device_get(learned_energies)
    print("Learned 4x4 energies:")
    print(trained_params_host)

    print(f"Training took {time.time() - start:.2f} seconds")

    save_energies(energy_history, learned_energies, outdir)
    save_config_file(lossconf, trainconf, datafile_train, datafile_validate, outdir)
    return learned_energies
