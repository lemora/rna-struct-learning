import csv
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from rnasl.utils.formats import CANONICAL_PAIRS, INT_TO_BASE
import rnasl.gconst as gc

# ---------- results

reactivity_baselines = {
    "2A3": {"Constant": 0.121, "Random": 0.231},
    "DMS": {"Constant": 0.123, "Random": 0.250},
    "2A3_norm": {"Constant": 0.032, "Random": 0.171},
    "DMS_norm": {"Constant": 0.032, "Random": 0.176},
}

reactivity_results = {
    "2A3": {
        "Random": {"mse": 0.231, "pcc": 0.0},
        "Static": {"mse": 0.121, "pcc": 0.0},
        "ViennaRNA": {"mse": 0.366, "pcc": 0.609},
        "EternaFold": {"mse": 0.198, "pcc": 0.664},
        # "ProbNuss1.1": {"mse": 0.119, "pcc": 0.501},
        "ProbNuss1.2": {"mse": 0.091, "pcc": 0.484},
        "ProbNuss1.3": {"mse": 0.180, "pcc": 0.539},
        "ProbNuss2": {"mse": 0.601, "pcc": 0.554},
        "ProbNuss3": {"mse": 0.589, "pcc": 0.546},
    },
    "DMS": {
        "Random": {"mse": 0.249, "pcc": 0.0},
        "Static": {"mse": 0.123, "pcc": 0.0},
        "ViennaRNA": {"mse": 0.384, "pcc": 0.589},
        "EternaFold": {"mse": 0.208, "pcc": 0.641},
        # "ProbNuss1.1": {"mse": 0.184, "pcc": 0.474},
        "ProbNuss1.2": {"mse": 0.095, "pcc": 0.442},
        "ProbNuss1.3": {"mse": 0.189, "pcc": 0.472},
        "ProbNuss2": {"mse": 0.648, "pcc": 0.511},
        "ProbNuss3": {"mse": 0.625, "pcc": 0.516},
    },
    "2A3_norm": {
        "Random": {"mse": 0.171, "pcc": 0.0},
        "Static": {"mse": 0.032, "pcc": 0.0},
        "ViennaRNA": {"mse": 0.4865, "pcc": 0.4915},
        "EternaFold": {"mse": 0.252, "pcc": 0.573},
        # "ProbNuss1.1": {"mse": 0.119, "pcc": 0.501},
        "ProbNuss1.2": {"mse": 0.0913, "pcc": 0.484},
        "ProbNuss1.3": {"mse": 0.127, "pcc": 0.500},
        "ProbNuss2": {"mse": 0.719, "pcc": 0.515},
        "ProbNuss3": {"mse": 0.697, "pcc": 0.513},
    },
    "DMS_norm": {
        "Random": {"mse": 0.176, "pcc": 0.0},
        "Static": {"mse": 0.032, "pcc": 0.0},
        "ViennaRNA": {"mse": 0.486, "pcc": 0.495},
        "EternaFold": {"mse": 0.252, "pcc": 0.568},
        # "ProbNuss1.1": {"mse": 0.127, "pcc": 0.452},
        "ProbNuss1.2": {"mse": 0.095, "pcc": 0.442},
        "ProbNuss1.3": {"mse": 0.134, "pcc": 0.459},
        "ProbNuss2": {"mse": 0.736, "pcc": 0.494},
        "ProbNuss3": {"mse": 0.711, "pcc": 0.500},
    }
}

structure_results = {
    "Baseline No Pairs": {
        "prec": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "sens": 0.0,
        "spec": 1.0,
        "bal_acc": 0.5,
    },
    "Baseline Full Stem": {
        "prec": 0.018,
        "recall": 0.03,
        "f1": 0.022,
        "sens": 0.964,
        "spec": 0.028,
        "bal_acc": 0.495,
    },
    "EternaFold": {
        "prec": 0.535,
        "recall": 0.594,
        "f1": 0.561,
        "sens": 0.848,
        "spec": 0.677,
        "bal_acc": 0.763,
    },
    "ViennaRNA": {
        "prec": 0.536,
        "recall": 0.574,
        "f1": 0.552,
        "sens": 0.826,
        "spec": 0.697,
        "bal_acc": 0.762,
    },
    # "ProbNuss1.1": {
    #     "prec": 0.001,
    #     "recall": 0.002,
    #     "f1": 0.002,
    #     "sens": 0.871,
    #     "spec": 0.133,
    #     "bal_acc": 0.502,
    # },
    "ProbNuss1.2": {
        "prec": 0.002,
        "recall": 0.003,
        "f1": 0.003,
        "sens": 0.873,
        "spec": 0.127,
        "bal_acc": 0.500,
    },
    "ProbNuss1.3": {
        "prec": 0.002,
        "recall": 0.003,
        "f1": 0.002,
        "sens": 0.872,
        "spec": 0.125,
        "bal_acc": 0.498,
    },
    "ProbNuss2": {
        "prec": 0.126,
        "recall": 0.103,
        "f1": 0.106,
        "sens": 0.416,
        "spec": 0.599,
        "bal_acc": 0.508,
    },
    "ProbNuss3": {
        "prec": 0.057,
        "recall": 0.040,
        "f1": 0.044,
        "sens": 0.357,
        "spec": 0.620,
        "bal_acc": 0.489,
    },
}


# ---------- static results plots


def plot_reactivity_comparison(metric="mse", outname=None):
    """
    Compare reactivity prediction across datasets.
    metric: "mse" or "pcc"
    """
    if metric not in ["mse", "pcc"]:
        raise ValueError("Metric must be 'mse' or 'pcc'")

    plt.figure(figsize=(10, 4))
    datasets = list(reactivity_results.keys())
    y_ticks, y_labels = [], []
    colours = plt.cm.tab10.colors

    for i, dataset in enumerate(datasets):
        y_base = len(datasets) - i
        offset = 0

        for j, (tool, res) in enumerate(reactivity_results[dataset].items()):
            val = res[metric]
            y = y_base - offset * 0.1

            if tool.lower() == "random":
                colour, linestyle = "grey", ":"
            elif tool.lower() == "static":
                colour, linestyle = "grey", "--"
            else:
                colour = colours[j - 2 % len(colours)]
                linestyle = "-"

            plt.hlines(y, xmin=0, xmax=val, color=colour, linestyle=linestyle,
                       linewidth=2, label=tool if i == 0 else "")
            plt.plot(val, y, 'o', color=colour)
            offset += 1

        y_ticks.append(y_base)
        y_labels.append(dataset)

    plt.yticks(y_ticks, y_labels)
    plt.xlabel(metric.upper())
    plt.grid(axis='x')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if outname is None:
        outname = f"reactivity_prediction_comparison_{metric}.png"
    plt.savefig(outname, dpi=300)
    print(f"Saved plot to {outname}")
    plt.show()
    plt.close()


def plot_structure_comparison(metric="f1", outname=None):
    """
    Compare structure prediction across predictors.
    metric: "f1" or "bal_acc"
    """
    if metric not in ["f1", "bal_acc"]:
        raise ValueError("Metric must be 'f1' or 'bal_acc'")

    metric_name = {"f1": "Pair F1 Score", "bal_acc": "Base Balanced Accuracy"}[metric]

    plt.figure(figsize=(10, 3))
    predictors = list(structure_results.keys())
    y_ticks, y_labels = [], []
    colours = plt.cm.tab10.colors

    for i, predictor in enumerate(predictors):
        val = structure_results[predictor][metric]
        y = len(predictors) - i

        if "baseline" in predictor.lower():
            colour, linestyle = "grey", "--" if "no pairs" in predictor.lower() else ":"
        else:
            colour = colours[i - 2 % len(colours)]
            linestyle = "-"

        plt.hlines(y, xmin=0, xmax=val, color=colour, linestyle=linestyle,
                   linewidth=2, label=predictor if i == 0 else "")
        plt.plot(val, y, 'o', color=colour)

        y_ticks.append(y)
        y_labels.append(predictor)

    plt.yticks(y_ticks, y_labels)
    plt.xlabel(metric_name)
    plt.grid(axis='x')
    plt.tight_layout()

    if outname is None:
        outname = f"structure_prediction_comparison_{metric}.png"
    plt.savefig(outname, dpi=300)
    print(f"Saved plot to {outname}")
    plt.show()
    plt.close()


# ---------- dynamic learningeval things

def get_baselines(dataset_key):
    bl = reactivity_baselines.get(dataset_key)
    if bl is None:
        raise ValueError(f"Unknown dataset type: {dataset_key}")
    return bl["Constant"], bl["Random"]


def plot_training_curve_with_baselines(epochs: list, train: list, val: list, outdir=None,
        outname="loss_history", show_baselines=True, loss_type="", dataset_type="2A3"):
    name_suf = f" ({loss_type})" if loss_type else ""
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train, label=f"Train Loss{name_suf}", marker='o')
    if any(v is not None for v in val):
        plt.plot(epochs, val, label=f"Validation Loss{name_suf}", marker='x')

    if show_baselines and dataset_type:
        constant_bl, random_bl = get_baselines(dataset_type)
        plt.axhline(y=random_bl, color='black', linestyle='-.',
                    label=f"{dataset_type} rand. baseline ({random_bl:.3f})")
        plt.axhline(y=constant_bl, color='black', linestyle='--',
                    label=f"{dataset_type} const. baseline ({constant_bl:.3f})")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    # plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if outdir:
        outname_suf = f"_{loss_type.replace(' ', '')}" if loss_type else ""
        outname = f"{outname}{outname_suf}.png"
        filepath = os.path.join(outdir, outname)
        plt.savefig(filepath)
        print(f"Saved loss history curve to {filepath}")

    plt.show()
    plt.close()


def plot_energy_evolution(energy_history, outdir=None, outname="energies_history.png"):
    num_epochs = energy_history.shape[0]
    cmap = mpl.colormaps['tab10']
    colors = list(cmap(np.linspace(0, 1, 10)))
    colors = colors[6:] + colors[:6]

    plt.figure(figsize=(10, 4))
    curves = []

    color_idx = 0
    for i in range(4):
        for j in range(i, 4):  # plot only the 10 unique entries
            base_i = INT_TO_BASE[i]
            base_j = INT_TO_BASE[j]
            label = f"e[{base_i},{base_j}]"
            color = colors[color_idx]
            linewidth = 1.5

            if base_i == base_j:
                linestyle = "dotted"
                category = "same"
            elif (i, j) in CANONICAL_PAIRS or (j, i) in CANONICAL_PAIRS:
                linestyle = "solid"
                category = "canonical"
            else:
                linestyle = "dashed"
                category = "other"

            line, = plt.plot(range(num_epochs),
                             energy_history[:, i, j],
                             label=label,
                             color=color,
                             linestyle=linestyle,
                             linewidth=linewidth)

            curves.append((category, label, line))
            color_idx += 1

    curves.sort(key=lambda x: ({"canonical": 0, "same": 1, "other": 2}[x[0]], x[1]))

    sorted_lines = [line for _, _, line in curves]
    sorted_labels = [label for _, label, _ in curves]
    plt.legend(sorted_lines, sorted_labels, ncol=5, fontsize="small", loc="upper right")

    plt.xlabel("Steps")
    plt.ylabel("Energy Value")
    plt.grid(True)
    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, outname)
        plt.savefig(filepath)
        print(f"Saved energy history curve to {filepath}")

    plt.show()
    plt.close()


def plot_loss_history_full(train_logs: list[dict[str, float]], val_logs: list[dict[str, float]], outdir: str,
        exclude: list[str] = ()):
    if not train_logs:
        print("No training log data to plot.")
        return

    keys = train_logs[0].keys()

    for key in keys:
        if key in exclude:
            continue
        train_vals = [d.get(key, float("nan")) for d in train_logs]
        val_vals = [d.get(key, float("nan")) for d in val_logs]

        capkey = key.capitalize()
        plt.figure()
        plt.plot(train_vals, label=f"Train {capkey}", linewidth=2)
        plt.plot(val_vals, label=f"Val {capkey}", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel(f"{capkey} Loss")
        # plt.title(f"{key} Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(outdir, f"loss_history_{key}.png")
        plt.savefig(path)
        if gc.DISPLAY:
            plt.show()
        plt.close()
        print(f"Saved full loss plot: {path}")


def plot_loss_components_stacked_area(train_logs: list[dict[str, float]], outdir=None,
        outname="loss_components_stacked_area.png", exclude=("total",)):
    """
    Plot a stacked filled line plot showing contribs of loss components over time.

    train_logs: list of dicts with per-step loss components
    exclude: keys to exclude (e.g., 'total', or others)
    """
    if not train_logs:
        print("No training log data to plot.")
        return

    keys = list(train_logs[0].keys())
    keys = [k for k in keys if k not in exclude]
    steps = np.arange(len(train_logs))

    # data matrix (steps times components)
    data = np.array([[log.get(k, 0.0) for k in keys] for log in train_logs])

    if 'total' in train_logs[0]:
        total = np.array([log.get('total', 0.0) for log in train_logs])
    else:
        total = np.sum(data, axis=1)  # fallback

    itemcount = len(keys)
    cmap = mpl.colormaps['tab10']
    colors = list(cmap(np.linspace(0, 1, itemcount)))
    colors = colors[0:] + colors[:0]

    plt.figure(figsize=(10, 4))

    for i, key in enumerate(keys):
        y = data[:, i]
        plt.fill_between(
            steps, 0, y,
            where=(y >= 0), interpolate=True,
            color=colors[i], alpha=0.6
        )
        plt.fill_between(
            steps, 0, y,
            where=(y < 0), interpolate=True,
            color=colors[i], alpha=0.3
        )
        plt.plot(steps, y, color=colors[i], linewidth=1, label=key)

    plt.plot(
        steps, total,
        color='black', linewidth=2.5, linestyle='--',
        label='Total'
    )

    plt.axhline(0, color='grey', linewidth=1)
    plt.xlabel("Steps")
    plt.ylabel("Loss Contribution")
    # plt.title("Loss Components Over Training")
    plt.legend(loc='upper right', fontsize='small', ncol=3)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, outname)
        plt.savefig(path)
        print(f"Saved stacked area plot: {path}")

    plt.show()
    plt.close()


# ------ data loading

def load_loss_history(path: str):
    epochs, train_losses, val_losses = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_loss = row['val_loss']
            val_losses.append(float(val_loss) if val_loss != '' else None)
    return epochs, train_losses, val_losses


def load_loss_log(filepath: str) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Load CSV loss log and return full per-step train and val records."""
    full_train, full_val = [], []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            record = {k: float(v) for k, v in row.items() if k not in ("step", "split")}
            if split == "train":
                full_train.append(record)
            elif split == "val":
                full_val.append(record)

    return full_train, full_val


# ------ reactivity and structure eval

def evaluate_training_old(loss_history_file: str, energy_history_file: str):
    epochs, train_losses, val_losses = load_loss_history(loss_history_file)
    plot_training_curve_with_baselines(epochs, train_losses, val_losses, outdir=os.path.dirname(loss_history_file))

    energy_history = np.load(energy_history_file)
    plot_energy_evolution(energy_history, os.path.dirname(energy_history_file))
    # plot_energy_evolution_subplots(energy_history)


def get_dataset_type_from_outdir(outdir):
    """
    Given an output directory, read the contained `conf.yaml`,
    and extract the dataset type from the `train_file` entry.
    """
    conf_path = os.path.join(outdir, "config.yaml")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Configuration file not found: {conf_path}")

    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)

    train_file = conf.get("train_file")
    if not train_file:
        raise ValueError(f"No 'train_file' found in {conf_path}")

    # Extract dataset type from train_file path
    parts = train_file.split(os.sep)
    for part in parts:
        if part.lower().startswith("2a3") or part.lower().startswith("dms"):
            return part
    print(f"Could not determine dataset type from train_file path: {train_file}")
    return None


def eval_global_results():
    plot_reactivity_comparison("mse")
    plot_reactivity_comparison("pcc")
    plot_structure_comparison("f1")
    plot_structure_comparison("bal_acc")
    quit()


def evaluate_training(loss_history_file: str, energy_history_file: str, dataset_type: str = None):
    # eval_global_results()

    outdir = os.path.dirname(loss_history_file)
    ds_type = dataset_type if dataset_type else get_dataset_type_from_outdir(outdir)

    print(f"Using the baselines for dataset type '{ds_type}'.")

    train_logs, val_logs = load_loss_log(loss_history_file)
    train_total = [d.get("total", float("nan")) for d in train_logs]
    val_total = [d.get("total", float("nan")) for d in val_logs]

    train_mse = [d.get("mse", float("nan")) for d in train_logs]
    val_mse = [d.get("mse", float("nan")) for d in val_logs]
    steps = list(range(len(train_total)))

    plot_loss_components_stacked_area(train_logs, outdir)
    plot_training_curve_with_baselines(steps, train_total, val_total, outdir, loss_type="total", dataset_type=ds_type)
    plot_training_curve_with_baselines(steps, train_mse, val_mse, outdir, loss_type="MSE", dataset_type=ds_type)
    plot_loss_history_full(train_logs, val_logs, outdir=outdir, exclude=["total", "mse"])

    energy_history = np.load(energy_history_file)
    plot_energy_evolution(energy_history, os.path.dirname(energy_history_file))
