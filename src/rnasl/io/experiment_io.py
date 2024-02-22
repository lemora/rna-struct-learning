import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from rnasl.training.loss_train_configs import LossConfig, TrainConfig
from rnasl.utils.formats import BASE_TO_INT, INT_TO_BASE


# ------ seq to/from file

def write_rna_struct_to_file(rna: str, struct: str, filename: str = "struct.dbn", overwrite=True):
    outname = filename
    if not outname.lower().endswith('.dbn'):
        outname += '.dbn'

    if os.path.isfile(outname):
        print(f"WARN: File {outname} exixts.")
        print("Overwriting." if overwrite else "Cancelling.")
    with open(outname, "w") as f:
        f.write(f"{rna.upper()}\n{struct}")
    print(f"RNA structure saved to file '{outname}'.")


def read_fasta_seq(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines[0].startswith('>'):
            return ''.join(lines[1:])
        else:
            return ''.join(lines)


def read_dbn_seq(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= 2:
            return lines[0]
        else:
            print("ERR: .dbn file format not recognized or incomplete.")
            quit()


def read_in_seq(seq_arg):
    if os.path.isfile(seq_arg) and seq_arg.lower().endswith(('.seq', '.dbn', '.fasta', '.fa')):
        print(f"Reading in sequence from file: {seq_arg}")
        ext = os.path.splitext(seq_arg)[1].lower()
        if ext == '.dbn':
            return read_dbn_seq(seq_arg)
        else:
            return read_fasta_seq(seq_arg)
    else:
        return seq_arg


# ------ energies to/from file

def energy_mat_from_file(filename: str) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=float)

    with open(filename, 'r') as file:
        for line in file:
            words = line.strip().split()
            if len(words) != 2:
                continue
            name, value = words
            if name.startswith("base_pair"):
                b1, b2 = name[-2], name[-1]
                i, j = BASE_TO_INT[b1], BASE_TO_INT[b2]
                energy = float(value)
                mat[i, j] = energy
                mat[j, i] = energy
    return mat


def write_energy_mat_to_file(mat: np.ndarray, filename: str):
    with open(filename, 'w') as file:
        for i in range(4):
            for j in range(i, 4):
                b1, b2 = INT_TO_BASE[i], INT_TO_BASE[j]
                energy = mat[i, j]
                file.write(f"base_pair_{b1}{b2} {energy:.6g}\n")
    print(f"Wrote energies to file {filename}.")


# ------ training history/eval

def write_loss_history_to_file(history: list, filename: str):
    """
    history: list of dicts: [{'epoch': 1, 'train_loss': 0.123, 'val_loss': 0.456}, ...]
    """
    if not history:
        raise ValueError("History is empty")

    fieldnames = history[0].keys()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"Wrote training loss history to file {filename}.")


def save_training_outputs(train_losses: np.ndarray, val_losses: np.ndarray, energy_history: np.ndarray,
        learned_energies: np.ndarray, outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)

    # loss history
    history = [{'epoch': i + 1, 'train_loss': tr, 'val_loss': val_losses[i] if i < len(val_losses) else None}
               for i, tr in enumerate(train_losses)]
    loss_file = os.path.join(outdir, "loss_history.csv")
    write_loss_history_to_file(history, loss_file)

    # energy history
    energy_history_file = os.path.join(outdir, "energy_history.npy")
    np.save(energy_history_file, np.array(energy_history))

    # best learned energies
    energy_file = os.path.join(outdir, "TrainedEnergies.txt")
    write_energy_mat_to_file(learned_energies, energy_file)


def save_energies(energy_history: np.ndarray, learned_energies: np.ndarray, outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)
    # energy history
    energy_history_file = os.path.join(outdir, "energy_history.npy")
    np.save(energy_history_file, np.array(energy_history))

    # best learned energies
    energy_file = os.path.join(outdir, "TrainedEnergies.txt")
    write_energy_mat_to_file(learned_energies, energy_file)


def save_eval_outputs(eval_results: str, outdir: str, filename: str):
    out_file = os.path.join(outdir, filename)
    with open(out_file, 'w') as file:
        file.write(eval_results)
    print(f"Wrote test eval results to file {out_file}.")


def save_eval_outputs_json(eval_results: str, outdir: str, filename: str):
    out_file = os.path.join(outdir, filename)
    with open(out_file, "w") as jf:
        json.dump(eval_results, jf, indent=4)
    print(f"Wrote json test eval results to file {out_file}.")


# ------ store/load training configs


def save_config_file(lossconf: LossConfig, trainconf: TrainConfig,
        train_file: str, val_file: str, outdir: str):
    config = {
        "train_file": train_file,
        "val_file": val_file,
        "loss_config": asdict(lossconf),
        "train_config": asdict(trainconf)
    }

    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
