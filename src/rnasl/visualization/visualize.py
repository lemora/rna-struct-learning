import matplotlib.pyplot as plt
import numpy as np
import os


# ---------- rna


def bp_probability_dot_plot(prob_matrix: np.ndarray, seq: str, mfe_pairs: list[tuple[int, int]] = (),
        mea_pairs: list[tuple[int, int]] = (), title="Base Pair Probabilities (Discrete Log Scale)"):
    hierarchies = 8
    min_prob = 1e-8
    sizes = np.linspace(2 * (hierarchies + 1) - 1, 0, hierarchies + 1)
    colors = [(0.1 + i * 0.07, 0.2 + i * 0.07, 1.0) for i in range(hierarchies + 1)]
    n = prob_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(n):
        for j in range(i, n):
            p = prob_matrix[i, j]
            if p < min_prob:
                index = len(sizes) - 1
            elif p >= 0.5:
                index = 0
            else:
                index = int(np.floor(-np.log10(p)))
                index = np.clip(index, 0, hierarchies - 1) + 1
            size = sizes[index]
            color = colors[index]
            ax.scatter(j, i, s=size, marker='s', color=color)

    if len(mea_pairs) > 0:
        for (i, j) in mea_pairs:
            ax.scatter(i, j, s=sizes[0], marker='s', color="gray")
    if len(mfe_pairs) > 0:
        for (i, j) in mfe_pairs:
            ax.scatter(i, j, s=sizes[1], marker='x', color="black")

    ax.plot([0, n - 1], [0, n - 1], color='black', linewidth=1)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(title)
    # ax.legend()
    plt.tight_layout()
    plt.show()


# ---------- circle plots

def get_percentile_threshold(bpp_matrix, percentile=90):
    values = bpp_matrix[np.triu_indices_from(bpp_matrix, k=1)]  # upper triangle, excluding diagonal
    values = values[values > 0]  # exclude exact zeros
    if len(values) == 0:
        return 0.0
    return np.percentile(values, percentile)


def bp_probability_circle_plot(prob_matrix, seq: str, percentile=95, cmap_name='plasma'):
    threshold = get_percentile_threshold(prob_matrix, percentile)
    n = prob_matrix.shape[0]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    num_categories = 10
    sizes = np.linspace(4, 0.5, num_categories)
    linewidth = 1
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(1.0 - i / (num_categories - 1)) for i in range(num_categories)]
    n = prob_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2))

    # plot positions and labels
    font_size = max(9, 18 - n // 2)
    points_to_display = 10 if n > 50 else n
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.plot(xi, yi, 'o', color='black', markersize=3)
        # ax.text(xi * 1.1, yi * 1.1, str(i), ha='center', va='center', fontsize=font_size, color='black')
        ax.text(xi * 1.1, yi * 1.1, str(seq[i]), ha='center', va='center', fontsize=font_size, color='black')
        if i % max(1, n // points_to_display) == 0:
            ax.text(xi * 1.18, yi * 1.18, str(i), ha='center', va='center', fontsize=font_size,
                    color='black')  # ax.plot(xi, yi, 'o', color='blue')

    # draw connections
    for i in range(n):
        for j in range(i + 1, n):
            p = prob_matrix[i, j]
            if p > threshold:
                index = int(np.floor(-np.log10(p)))
                index = np.clip(index, 0, len(sizes) - 1)
                color = colors[index]
                ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=linewidth)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=threshold, vmax=1.0))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, shrink=0.7, label='normalized P(i,j)')
    cbar.set_ticks([])
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    plt.title('Base-Pair Probabilities (Circle Plot)', fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------- unpaired probabilities/reactivities

def plot_paired_unpaired_probs(paired_probs: np.ndarray, seq: str, title="Base (Un)Paired Probabilities",
        label_seq=True, outdir=None, outname=""):
    print(f"outdir: {outdir}")
    unpaired_probs = 1.0 - paired_probs
    # print(f"paired: {paired_probs}")
    # print(f"unpaired: {unpaired_probs}")
    n = len(seq)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 3.5))

    # stacked bars: unpaired (bottom), paired (top)
    # bars_paired = ax.bar(x, paired_probs, color='salmon', edgecolor='black', hatch="/", label="Paired")
    # bars_unpaired = ax.bar(x, unpaired_probs, bottom=paired_probs, color='skyblue', edgecolor='black',
    # label="Unpaired")
    bars_unpaired = ax.bar(x, unpaired_probs, color='skyblue', edgecolor='black', hatch="/", label="Unpaired")
    # bars_paired = ax.bar(x, paired_probs, bottom=unpaired_probs, color='white', edgecolor='black', label="Paired")

    # ax.set_yscale('log')
    if label_seq:
        ax.set_xticks(x)
        ax.set_xticklabels(list(seq), fontsize=8)
    else:
        ax.set_xticks(x[::10])
        ax.set_xticklabels([str(i) for i in x[::10]], ha='right', fontsize=8)

    ax.set_xlabel("Position (base)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    # ax.set_title(title)
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.legend()
    plt.tight_layout()

    if outdir:
        outname_suf = f"_{outname.replace(' ', '')}" if outname else ""
        outname = f"unpaired_marginals{outname_suf}.png"
        filepath = os.path.join(outdir, outname)
        plt.savefig(filepath)
        print(f"Saved unpaired marginals to {filepath}")

    plt.show()
    plt.close()


# ---------- other

def npmat_to_str(mat):
    rows, cols = mat.shape
    formatted_rows = []
    for i in range(rows):
        formatted_row = [f"{x:.2f}" if isinstance(x, (float, np.float32, np.float32)) else str(x) for x in mat[i]]
        formatted_rows.append(" ".join(formatted_row))
    return "\n".join(formatted_rows)


def print_npmat(mat):
    mat_str = npmat_to_str(mat)
    print(mat)
