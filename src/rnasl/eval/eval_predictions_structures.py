import functools
from functools import partial
import json
import os
import RNA
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import rnasl.gconst as gc
from rnasl.folding.nussinov_pf_jax import calc_base_pair_probs, calc_partition_function, compute_mea_structure
from rnasl.folding_primitives.semiring import Semiring, make_logsumexp_semiring
from rnasl.io.dataset_loader import load_structure_dataset_archiveii, load_structure_dataset_eterna
from rnasl.io.experiment_io import energy_mat_from_file, save_eval_outputs_json
from rnasl.utils.formats import decode_seq_jax, pairing_to_vienna, vienna_to_pairing


# ------ test set eval metrics + helpers

def struct_base_sens_spec(pred_pairs, true_pairs, seq_len):
    pred_paired = jnp.zeros(seq_len, dtype=bool)
    true_paired = jnp.zeros(seq_len, dtype=bool)

    for i, j in pred_pairs:
        pred_paired = pred_paired.at[i].set(True)
        pred_paired = pred_paired.at[j].set(True)

    for i, j in true_pairs:
        true_paired = true_paired.at[i].set(True)
        true_paired = true_paired.at[j].set(True)

    TP = jnp.sum(jnp.logical_and(pred_paired, true_paired))
    TN = jnp.sum(jnp.logical_and(~pred_paired, ~true_paired))
    FP = jnp.sum(jnp.logical_and(pred_paired, ~true_paired))
    FN = jnp.sum(jnp.logical_and(~pred_paired, true_paired))

    sensitivity = TP / jnp.maximum(TP + FN, 1)
    specificity = TN / jnp.maximum(TN + FP, 1)
    accuracy = (TP + TN) / jnp.maximum(TP + TN + FP + FN, 1)

    return float(sensitivity), float(specificity), float(accuracy)


def struct_pair_f1(pred_pairs, true_pairs, seq_len):
    pred_mat = jnp.zeros((seq_len, seq_len), dtype=bool)
    true_mat = jnp.zeros((seq_len, seq_len), dtype=bool)

    for i, j in pred_pairs:
        if i < j:
            pred_mat = pred_mat.at[i, j].set(True)
        else:
            pred_mat = pred_mat.at[j, i].set(True)

    for i, j in true_pairs:
        if i < j:
            true_mat = true_mat.at[i, j].set(True)
        else:
            true_mat = true_mat.at[j, i].set(True)

    tp = jnp.sum(jnp.logical_and(pred_mat, true_mat))
    fp = jnp.sum(jnp.logical_and(pred_mat, jnp.logical_not(true_mat)))
    fn = jnp.sum(jnp.logical_and(jnp.logical_not(pred_mat), true_mat))

    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / jnp.maximum(precision + recall, 1e-8)

    return float(precision), float(recall), float(f1)


@functools.lru_cache(maxsize=64)
def get_mea_fn(seq_len: jnp.int32):
    print(f"Compiling update_step for sequence length {seq_len}")

    @partial(jax.jit, static_argnames=["semiring", "h"])
    def _fn(seq_encoded: jnp.ndarray, energy_mat: jnp.ndarray, semiring: Semiring, h: jnp.int32):
        Z, Z_p = calc_partition_function(seq_encoded, energy_mat, semiring, h)
        P = calc_base_pair_probs(Z, Z_p, seq_encoded, semiring)
        return compute_mea_structure(P, h=h)

    return _fn


def get_mea_struct_jax(seq: jnp.ndarray, energy_mat: jnp.ndarray, semiring: Semiring, h: jnp.int32):
    mea_fn = get_mea_fn(seq.shape[0])
    return mea_fn(seq, energy_mat, semiring, h)


# ------ baselines


def run_viennarna_pred(sequence: str) -> list[tuple[int, int]]:
    temp_celsius = gc.TEMP - 273.15
    RNA.cvar.temperature = temp_celsius
    fc = RNA.fold_compound(sequence)
    pf = fc.pf()
    # centroid_struct, e = fc.centroid()
    # mfe_strcut, e = fc.mfe()
    mea_struct, e = fc.MEA()
    pairing = vienna_to_pairing(mea_struct)
    return pairing


def run_eternafold_pred(sequence: str, eternafold_dir: str) -> list[tuple[int, int]]:
    """ Calls eternafold (contrafold) with its default parameters and returns the predicted structure. """
    relative_efold_path = "src/contrafold"
    relative_param_path = "parameters/EternaFoldParams.v1"

    eternafold_path = os.path.join(eternafold_dir, relative_efold_path)
    eternafold_params = os.path.join(eternafold_dir, relative_param_path)

    with tempfile.NamedTemporaryFile(mode='w+', suffix=".fasta", delete=False) as temp_seq_file:
        temp_seq_file.write(sequence)
        temp_seq_file_path = temp_seq_file.name

    try:
        result = subprocess.run(
            [eternafold_path, "predict", temp_seq_file_path, "--params", eternafold_params],
            capture_output=True,
            text=True,
            check=False
        )
        vienna_struct = result.stdout.strip().splitlines()[-1]
        pairing = vienna_to_pairing(vienna_struct)
        return pairing
    finally:
        os.remove(temp_seq_file_path)


def baseline_empty():
    return []


def baseline_full_stem(seq: str, h: int = 3) -> list[tuple[int, int]]:
    n = len(seq)
    pairs = []
    i = 0
    j = n - 1
    while i < j:
        if j - i - 1 >= h:
            pairs.append((i, j))
        i += 1
        j -= 1
    return pairs


# ------ reactivity and structure eval


def evaluate_structure_predictions(energy_mat_path: str, h: int = 4, predictor="probnuss", predictor_path: str = None):
    print(f"Evaluating predictor {predictor} on the structure test set.")
    energy_mat = None if energy_mat_path == None else energy_mat_from_file(energy_mat_path)
    semiring = make_logsumexp_semiring()

    dataset = load_structure_dataset_archiveii()
    if not dataset:
        print("No valid structures found.")
        return 0.0, 0.0, 0.0
    print("Dataset loaded.\n")

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_sens = 0.0
    sum_spec = 0.0
    sum_acc = 0.0

    n = 0
    for (seq, true_pairs) in dataset:
        # if len(seq) == 2968:
        #     print("Processing seq of len 2968")
        #     print(decode_seq_jax(seq))
        #     print()

        if predictor == "probnuss":
            mea_pairs_untrimmed = get_mea_struct_jax(seq, energy_mat, semiring, h)
            pred_pairs = [tuple(map(int, pair)) for pair in np.array(mea_pairs_untrimmed) if not (pair == 0).all()]
        elif predictor == "baseline_nopairs":
            pred_pairs = baseline_empty()
        elif predictor == "baseline_fullstem":
            pred_pairs = baseline_full_stem(seq, h)
        elif predictor == "eternafold":
            pred_pairs = run_eternafold_pred(decode_seq_jax(seq), predictor_path)
        elif predictor == "viennarna":
            pred_pairs = run_viennarna_pred(decode_seq_jax(seq))
        else:
            raise Exception(f"Unknown predictor {predictor}")

        precision, recall, f1 = struct_pair_f1(pred_pairs, true_pairs, len(seq))
        sens, spec, acc = struct_base_sens_spec(pred_pairs, true_pairs, len(seq))

        if gc.VERBOSE and (f1 > 0.9 or f1 < 0.2):
            print(f"[{n}] comparing structures:")
            print(f"pred:  {pairing_to_vienna(seq, pred_pairs)}")
            print(f"truth: {pairing_to_vienna(seq, true_pairs)}")

        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1
        sum_sens += sens
        sum_spec += spec
        sum_acc += acc
        n += 1

        if n % 100 == 0:
            print(f"[{n:3d}] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
                  f"Sens: {sens:.4f} | Spec: {spec:.4f} | Acc: {acc:.4f}")

    avg_precision = sum_precision / n
    avg_recall = sum_recall / n
    avg_f1 = sum_f1 / n
    avg_sens = sum_sens / n
    avg_spec = sum_spec / n
    avg_acc = sum_acc / n
    avg_balanced_acc = (avg_sens + avg_spec) / 2.0

    summary = {
        "files_evaluated": len(dataset),
        "predictor": predictor,
        "avg_pair_precision": round(avg_precision, 3),
        "avg_pair_recall": round(avg_recall, 3),
        "avg_pair_f1": round(avg_f1, 3),
        "avg_base_sensitivity": round(avg_sens, 3),
        "avg_base_specificity": round(avg_spec, 3),
        "avg_base_accuracy": round(avg_acc, 3),
        "avg_base_balanced_accuracy": round(avg_balanced_acc, 3)
    }

    print("\n========== Structure Evaluation Summary ==========")
    print(json.dumps(summary, indent=2))
    print("===================================================\n")

    results_dir = os.path.dirname(energy_mat_path) if energy_mat_path else os.getcwd()
    save_eval_outputs_json(summary, results_dir, f"structure_test_eval_{predictor}.json")

    return avg_f1, avg_sens, avg_spec
