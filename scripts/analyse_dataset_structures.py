import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rnasl.utils.formats import decode_seq_jax

from rnasl.io.dataset_loader import load_structure_dataset_eterna, load_structure_dataset_archiveii


def analyse_structure_dataset(dataset, show_plots=True, top_n_pair_types=10):
    lengths = []
    num_pairs = []
    densities = []
    pair_type_counter = Counter()

    for seq_encoded, base_pairs in dataset:
        seq = decode_seq_jax(seq_encoded)
        length = len(seq_encoded)
        n_pairs = len(base_pairs)
        density = n_pairs / length if length > 0 else 0

        lengths.append(length)
        num_pairs.append(n_pairs)
        densities.append(density)

        # Count pair types
        for i, j in base_pairs:
            if i >= length or j >= length:
                continue
            nt1 = seq[i]
            nt2 = seq[j]
            pair = tuple(sorted((nt1, nt2)))
            pair_type_counter[pair] += 1

    lengths = np.array(lengths)
    num_pairs = np.array(num_pairs)
    densities = np.array(densities)

    print(f"Total sequences: {len(dataset)}")
    print(f"Mean length: {lengths.mean():.2f} | Median: {np.median(lengths):.2f}")
    print(f"Mean base pairs: {num_pairs.mean():.2f} | Median: {np.median(num_pairs):.2f}")
    print(f"Mean pairing density: {densities.mean():.3f}")
    print(f"Average fraction of paired positions: {(2 * num_pairs.sum()) / lengths.sum():.3f}")

    total_pairs = sum(pair_type_counter.values())
    print(f"\nIncluded base pair types:")
    for pair, count in pair_type_counter.most_common():
        freq = count / total_pairs
        print(f"  {pair[0]}-{pair[1]}: {count} ({freq:.2%})")

    if show_plots:
        _plot_histogram(lengths, "Sequence Length", "Sequence Length", "Count", log_y=True, color="steelblue")
        _plot_histogram(num_pairs, "# Base Pairs", "Number of Base Pairs", "Count", log_y=True, color="salmon")
        _plot_histogram(densities, "Pairing Density", "Density (pairs / seq_len)", "Count", color="lightgreen")


def _plot_histogram(data, title, xlabel, ylabel, bins=30, log_y=False, color="steelblue"):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, color=color, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Analysing structures....")

    loadEterna = False
    if loadEterna:
        struct_dir = "data/structEterna"
        dataset = load_structure_dataset_eterna(struct_dir)
    else:
        dataset = load_structure_dataset_archiveii()

    analyse_structure_dataset(dataset)
