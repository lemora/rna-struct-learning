import csv
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
import subprocess
import tempfile
import RNA
from functools import partial

from rnasl.folding.nussinov_pf_jax import calc_base_pair_probs, calc_partition_function, compute_marginal_probs
from rnasl.folding_primitives.semiring import make_logsumexp_semiring
from rnasl.io.experiment_io import energy_mat_from_file, save_eval_outputs_json
from rnasl.utils.formats import encode_seq_jax
import rnasl.gconst as gc


# ------ test set eval metrics + helpers

def masked_mse(y_pred, y_true):
    # clip to match training and model output range
    y_true = jnp.clip(y_true, 0.0, 1.0)
    mask = jnp.isfinite(y_true)
    diff = jnp.where(mask, y_pred - y_true, 0.0)
    denom = jnp.maximum(mask.sum(), 1)
    return jnp.sum(diff ** 2) / denom


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


def run_viennarna_pred(sequence: str) -> list[tuple[int, int]]:
    temp_celsius = gc.TEMP - 273.15
    RNA.cvar.temperature = temp_celsius
    fc = RNA.fold_compound(sequence)
    pf = fc.pf()

    bpp_matrix_full = jnp.array(fc.bpp())  # shape (n+1, n+1), as viennarna is 1-indexed...
    bpp_matrix = bpp_matrix_full[1:, 1:]  # remove padding

    unpaired_probs = 1.0 - jnp.sum(bpp_matrix, axis=1)
    # print(f"unpaired margs: {unpaired_probs}")
    return unpaired_probs


def run_eternafold_bpp_pred(sequence: str, eternafold_dir: str) -> np.ndarray:
    relative_efold_path = "src/contrafold"
    relative_param_path = "parameters/EternaFoldParams.v1"

    eternafold_path = os.path.join(eternafold_dir, relative_efold_path)
    eternafold_params = os.path.join(eternafold_dir, relative_param_path)

    with tempfile.NamedTemporaryFile(mode='w+', suffix=".fasta", delete=False) as temp_seq_file:
        temp_seq_file.write(sequence)
        temp_seq_file_path = temp_seq_file.name

    bpp_file_path = temp_seq_file_path + ".bps.txt"

    try:
        # run eternafold to compute posterior base-pair probabilities
        subprocess.run(
            [eternafold_path, "predict", temp_seq_file_path, "--params", eternafold_params,
             "--posteriors", "0.00001", bpp_file_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        seq_len = len(sequence)
        bpp_matrix = np.zeros((seq_len, seq_len))

        # parse bps.txt file with bpp data
        with open(bpp_file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                i = int(parts[0]) - 1  # 1-based to 0-based
                for pair_info in parts[2:]:
                    if ':' not in pair_info:
                        continue
                    j_str, prob_str = pair_info.split(":")
                    j = int(j_str) - 1
                    prob = float(prob_str)
                    bpp_matrix[i, j] = prob
                    bpp_matrix[j, i] = prob

        unpaired_probs = 1.0 - np.sum(bpp_matrix, axis=1)
        return unpaired_probs

    finally:
        # remove temporary files: seq and bpp
        os.remove(temp_seq_file_path)
        if os.path.exists(bpp_file_path):
            os.remove(bpp_file_path)


@partial(jax.jit, static_argnames=["semiring", "h"])
def compute_unpaired_probs_jit(enc_seq, energy_mat, semiring, h):
    Z, Z_p = calc_partition_function(enc_seq, energy_mat, semiring, h)
    P = calc_base_pair_probs(Z, Z_p, enc_seq, semiring)
    unpaired_probs = compute_marginal_probs(P, paired=False)
    return unpaired_probs


# ------ reactivity eval

def evaluate_reactivity_predictions(
        test_data_path: str,
        energy_mat_path: str | None,
        h: int = 3,
        predictor: str = "probnuss",
        predictor_path: str | None = None):
    print(f"Evaluating predictor {predictor} on the reactivity test set.")

    energy_mat = None
    if energy_mat_path is not None:
        energy_mat = energy_mat_from_file(energy_mat_path)
        energy_mat = jnp.array(energy_mat)

    semiring = make_logsumexp_semiring()

    mse_vals = []
    corr_vals = []
    n = 0

    with open(test_data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        react_cols = [c for c in reader.fieldnames if c.startswith('reactivity_0')]

        for row in reader:
            seq = row['sequence'].strip()
            if not seq:
                continue
            if float(row.get('SN_filter', '1').strip()) < 1:
                continue

            enc_seq = encode_seq_jax(seq)
            reacts = np.array([
                float(v) if (v and v.lower() != "null") else np.nan
                for v in [row[c].strip() for c in react_cols[:len(seq)]]
            ], dtype=np.float32)

            if len(enc_seq) != len(reacts):
                continue

            reacts = np.clip(reacts, 0.0, 2.0)

            def react_to_unpaired(react, midpoint=0.5, slope=10):
                return 1.0 / (1.0 + jnp.exp(-slope * (react - midpoint)))

            reacts = react_to_unpaired(reacts)

            if predictor == "probnuss":
                unpaired_probs = compute_unpaired_probs_jit(enc_seq, energy_mat, semiring, h)
            elif predictor == "viennarna":
                unpaired_probs = run_viennarna_pred(seq)
            elif predictor == "eternafold":
                unpaired_probs = run_eternafold_bpp_pred(seq, predictor_path)
            else:
                raise Exception(f"Unknown predictor {predictor}")

            mse = float(masked_mse(unpaired_probs, reacts))
            corr = float(masked_pearson_corr(unpaired_probs, reacts))

            mse_vals.append(mse)
            corr_vals.append(corr)

            if n % 1000 == 0:
                print(f"[{n:5d}] MSE: {mse:.4f} | Corr: {corr:.4f}")
            n += 1

    avg_mse = float(np.mean(mse_vals))
    avg_corr = float(np.mean(corr_vals))

    summary = {
        "test_data_path": test_data_path,
        "files_evaluated": n,
        "predictor": predictor,
        "avg_mse": round(avg_mse, 4),
        "avg_corr": round(avg_corr, 4)
    }

    print("\n========== Reactivity Evaluation Summary ==========")
    print(json.dumps(summary, indent=2))
    print("===================================================\n")

    results_dir = os.path.dirname(energy_mat_path) if energy_mat_path else os.getcwd()
    save_eval_outputs_json(summary, results_dir,
                           f"reactivity_test_eval_{predictor}.json")

    return avg_mse, avg_corr
