import collections
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# ---- Plotting Functions

def plot_per_position_counts(title, counts):
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.title(title)
    plt.xlabel("Position")
    ylabel = "Fraction" if np.max(counts) <= 1 else "Count"
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_signal_to_noise_distribution_hist(sig_to_noise):
    valid_sn = sig_to_noise[~np.isnan(sig_to_noise)]
    plt.figure(figsize=(8, 4))

    bins = np.linspace(0, np.nanmax(valid_sn), 500)
    # bins = np.linspace(0, min(20, np.nanmax(valid_sn)), 100)
    plt.hist(valid_sn, bins=bins, edgecolor='black', alpha=0.75)
    plt.axvline(1.0, color='red', linestyle='--', label='S/N = 1 threshold')

    plt.title("Signal-to-Noise Distribution")
    plt.xlabel("Signal-to-Noise")
    plt.ylabel("Count")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_signal_to_noise_distribution(sig_to_noise):
    valid_sn = sig_to_noise[~np.isnan(sig_to_noise)]

    plt.figure(figsize=(10, 2.5))
    y_vals = np.ones_like(valid_sn)

    plt.scatter(valid_sn, y_vals, alpha=0.5, s=10)
    plt.xscale('log')
    plt.axvline(1.0, color='red', linestyle='--', label='S/N = 1')

    plt.title("Signal-to-Noise Distribution")
    plt.xlabel("Signal-to-Noise Ratio")
    plt.yticks([])
    plt.legend()
    plt.grid(True, axis='x', which='both')
    plt.tight_layout()
    plt.show()


def plot_sn_strip_two_datasets(sn1, sn2, label1="2A3 (SHAPE)", label2="DMS"):
    sn1 = np.array(sn1)
    sn2 = np.array(sn2)
    valid_sn1 = sn1[~np.isnan(sn1)]
    valid_sn2 = sn2[~np.isnan(sn2)]

    plt.figure(figsize=(10, 3))
    plt.scatter(valid_sn1, np.ones_like(valid_sn1), alpha=0.4, s=10, label=label1)
    plt.scatter(valid_sn2, np.ones_like(valid_sn2) * 2, alpha=0.4, s=10, label=label2)

    plt.xscale('log')
    plt.axvline(1.0, color='red', linestyle='--', label='S/N = 1')

    plt.title("Signal-to-Noise Distribution")
    plt.yticks([1, 2], [label1, label2])
    plt.xlabel("Signal-to-Noise Ratio")
    plt.legend()
    plt.grid(True, axis='x', which='both')
    plt.tight_layout()
    plt.show()


# ------ analysis

def analyse_reactivities(filepath: str, max_count=None):
    sig_to_noise_all = []
    sig_to_noise_ge_1_count = 0

    all_reactivities = []
    per_base_stats = collections.defaultdict(list)
    per_base_issues = {base: {'nan': 0, 'lt_zero': 0, 'gt_one': 0, 'total': 0} for base in "ACGU"}

    global_issue_counts = {'nan': 0, 'lt_zero': 0, 'gt_one': 0}

    per_base_quarter_fractions_allseq = {
        base: [[] for _ in range(4)]
        for base in "ACGU"
    }

    base_counter = collections.Counter()
    base_total = 0

    missing_frac_list = []
    below_zero_frac_list = []
    above_one_frac_list = []
    gc_frac_list = []
    gc_balance_frac_list = []

    total_entries_all = 0
    total_entries_filtered = 0

    with open(filepath, newline='') as infile:
        reader = csv.DictReader(infile)
        reactivity_cols = [col for col in reader.fieldnames if col.startswith('reactivity_0')]

        for i, row in enumerate(reader):
            if max_count is not None and total_entries_all >= max_count:
                break

            try:
                seq = row['sequence'].strip()
                if not seq:
                    continue

                SN_filter = float(row.get('SN_filter', '1.0').strip())
                sn = row.get('signal_to_noise', '')
                try:
                    sn_val = float(sn)
                    sig_to_noise_all.append(sn_val)
                    if sn_val >= 1.0:
                        sig_to_noise_ge_1_count += 1
                except (ValueError, TypeError):
                    sig_to_noise_all.append(np.nan)

                total_entries_all += 1

                if SN_filter < 1.0:
                    continue
                total_entries_filtered += 1

                reactivities = np.array([
                    float(row[col]) if row[col].strip() not in ('', 'nan') else np.nan
                    for col in reactivity_cols[:len(seq)]
                ], dtype=float)

                non_nan_mask = ~np.isnan(reactivities)
                valid_len = np.sum(non_nan_mask)
                if valid_len == 0:
                    continue

                missing_frac_list.append(np.mean(np.isnan(reactivities)))
                below_zero_frac_list.append(np.mean((reactivities < 0) & non_nan_mask))
                above_one_frac_list.append(np.mean((reactivities > 1) & non_nan_mask))

                base_positions_in_quarters = {base: [0, 0, 0, 0] for base in "ACGU"}
                base_counts_in_seq = {base: 0 for base in "ACGU"}

                g_count = c_count = valid_bases = 0
                for idx, (base, val) in enumerate(zip(seq, reactivities)):
                    seq_len = len(seq)

                    if base in "ACGU":
                        valid_bases += 1
                        per_base_issues[base]['total'] += 1
                        base_counter[base] += 1
                        base_total += 1

                        base_counts_in_seq[base] += 1
                        pos_frac = idx / seq_len
                        if pos_frac < 0.25:
                            quarter = 0
                        elif pos_frac < 0.50:
                            quarter = 1
                        elif pos_frac < 0.75:
                            quarter = 2
                        else:
                            quarter = 3
                        base_positions_in_quarters[base][quarter] += 1

                        if base == 'G':
                            g_count += 1
                        elif base == 'C':
                            c_count += 1

                        if np.isnan(val):
                            per_base_issues[base]['nan'] += 1
                            global_issue_counts['nan'] += 1
                        elif val < 0:
                            per_base_issues[base]['lt_zero'] += 1
                            global_issue_counts['lt_zero'] += 1
                            per_base_stats[base].append(val)
                        elif val > 1:
                            per_base_issues[base]['gt_one'] += 1
                            global_issue_counts['gt_one'] += 1
                            per_base_stats[base].append(val)
                        else:
                            per_base_stats[base].append(val)

                for base in "ACGU":
                    total_in_seq = base_counts_in_seq[base]
                    if total_in_seq > 0:
                        for q in range(4):
                            frac = base_positions_in_quarters[base][q] / total_in_seq
                            per_base_quarter_fractions_allseq[base][q].append(frac)

                if valid_bases > 0:
                    gc_frac_list.append((g_count + c_count) / valid_bases)
                    gc_balance_frac_list.append(min(g_count, c_count) / valid_bases)

                clipped = [
                    np.clip(float(row[col]), 0.0, 1.0)
                    for col in reactivity_cols[:len(seq)]
                    if row[col].strip()
                ]
                all_reactivities.extend(clipped)

            except Exception as e:
                print(f"Row {i} skipped due to error: {e}")
                continue

    sig_to_noise_all = np.array(sig_to_noise_all, dtype=float)

    if all_reactivities:
        all_reactivities = np.array(all_reactivities)
        mean = np.mean(all_reactivities)
        var = np.var(all_reactivities)
    else:
        mean, var = float('nan'), float('nan')

    base_prevalence = {
        base: base_counter[base] / base_total if base_total else 0.0
        for base in "ACGU"
    }

    per_base_quarter_fractions_avg = {
        base: [
            float(np.mean(per_base_quarter_fractions_allseq[base][q]))
            if per_base_quarter_fractions_allseq[base][q] else float('nan')
            for q in range(4)
        ]
        for base in "ACGU"
    }

    return {
        'total_entries': total_entries_all,
        'sn_filtered_entries': total_entries_filtered,
        'sig_to_noise': sig_to_noise_all.tolist(),
        'sig_to_noise_ge_1_count': sig_to_noise_ge_1_count,
        'missing_value_frac_per_seq': missing_frac_list,
        'reactivity_lt_zero_frac_per_seq': below_zero_frac_list,
        'reactivity_gt_one_frac_per_seq': above_one_frac_list,
        'gc_content_per_seq': gc_frac_list,
        'gc_balance_per_seq': gc_balance_frac_list,
        'filtered_mean': float(mean),
        'filtered_var': float(var),
        'base_prevalence': base_prevalence,
        'global_issue_counts': global_issue_counts,
        'per_base_quarter_fractions': per_base_quarter_fractions_avg,
        'per_base_summary': {
            base: {
                'n_total': per_base_issues[base]['total'],
                'fraction_of_bases': base_prevalence.get(base, 0.0),
                'n_nan': per_base_issues[base]['nan'],
                'n_lt_zero': per_base_issues[base]['lt_zero'],
                'n_gt_one': per_base_issues[base]['gt_one'],
            }
            for base in "ACGU"
        },
        'per_base_distribution_stats': {
            base: {
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'std': float(np.std(vals)) if vals else float('nan'),
                'p1': float(np.percentile(vals, 1)) if vals else float('nan'),
                'p5': float(np.percentile(vals, 5)) if vals else float('nan'),
                'p95': float(np.percentile(vals, 95)) if vals else float('nan'),
                'p99': float(np.percentile(vals, 99)) if vals else float('nan'),
            }
            for base, vals in per_base_stats.items()
        }
    }


def summarize_results(name, result):
    def r(val):
        try:
            return round(float(val), 3)
        except (TypeError, ValueError):
            return None

    def safe_frac(n, d):
        return round(n / d, 4) if d else None

    per_base_summary = {}
    dist_stats_all = result.get("per_base_distribution_stats", {})

    for base in "ACGU":
        stats = result.get("per_base_summary", {}).get(base, {})
        dist = dist_stats_all.get(base, {})
        quarters = result.get('per_base_quarter_fractions').get(base, {})

        n_total = stats.get("n_total", 0)
        n_nan = stats.get("n_nan", 0)
        n_lt_zero = stats.get("n_lt_zero", 0)
        n_gt_one = stats.get("n_gt_one", 0)
        n_valid_in_range = n_total - n_nan - n_lt_zero - n_gt_one

        per_base_summary[base] = {
            "n_total": n_total,
            "fraction_of_bases": r(stats.get("fraction_of_bases")),
            "position_quarters": [r(q) for q in quarters] if quarters else None,
            "n_nan": n_nan,
            "n_lt_zero": n_lt_zero,
            "n_gt_one": n_gt_one,
            "n_valid_in_range": n_valid_in_range,
            "fraction_nan": safe_frac(n_nan, n_total),
            "fraction_lt_zero": safe_frac(n_lt_zero, n_total),
            "fraction_gt_one": safe_frac(n_gt_one, n_total),
            "fraction_valid_in_range": safe_frac(n_valid_in_range, n_total),
            "mean_reactivity": r(dist.get("mean")),
            "std_reactivity": r(dist.get("std")),
            "reactivity_p1": r(dist.get("p1")),
            "reactivity_p5": r(dist.get("p5")),
            "reactivity_p95": r(dist.get("p95")),
            "reactivity_p99": r(dist.get("p99")),
        }

    return {
        "dataset": name,
        "total_entries": result.get("total_entries"),
        "sn_filtered_entries": result.get("sn_filtered_entries"),
        "sig_to_noise_mean": r(np.nanmean(result.get("sig_to_noise", []))),
        "sig_to_noise_std": r(np.nanstd(result.get("sig_to_noise", []))),
        "sig_to_noise_ge_1_count": result.get("sig_to_noise_ge_1_count"),

        "avg_missing_value_frac_per_seq": r(np.mean(result.get("missing_value_frac_per_seq", []))),
        "avg_reactivity_lt_zero_frac_per_seq": r(np.mean(result.get("reactivity_lt_zero_frac_per_seq", []))),
        "avg_reactivity_gt_one_frac_per_seq": r(np.mean(result.get("reactivity_gt_one_frac_per_seq", []))),

        "avg_gc_content": r(np.mean(result.get("gc_content_per_seq", []))),
        "avg_gc_balance": r(np.mean(result.get("gc_balance_per_seq", []))),

        "filtered_mean": r(result.get("filtered_mean")),
        "filtered_var": r(result.get("filtered_var")),

        "per_base_summary": per_base_summary
    }


# ------ main

def analyse_and_summarize(name, path):
    print(f"Analysing {name}")
    print(f"Input file: {path}")
    raw = analyse_reactivities(path)
    summary = summarize_results(name, raw)
    print("\n========== Reactivity statistics ==========")
    print(json.dumps(summary, indent=2))
    print("===================================================\n")
    return summary


if __name__ == "__main__":
    print("Analysing reactivities....")

    dataset_paths = {
        "2A3_combined": "data/2A3/combined_cmap_data_2a3.csv",
        "DMS_combined": "data/DMS/combined_cmap_data_dms.csv",
        "2A3_train": "data/2A3/TRAIN/data.csv",
        "DMS_train": "data/DMS/TRAIN/data.csv",
        "2A3_train_norm": "data/2A3_norm/TRAIN/data.csv",
        "DMS_train_norm": "data/DMS_norm/TRAIN/data.csv",
    }

    summaries = {
        name: analyse_and_summarize(name, path)
        for name, path in dataset_paths.items()
    }

    outname = "reactivity_summary.json"
    with open(outname, "w") as f:
        json.dump(list(summaries.values()), f, indent=2)
    print(f"Wrote reactivity analysis results to {outname}")

    # if "sig_to_noise" in summaries["2A3_combined"] and "sig_to_noise" in summaries["DMS_combined"]:
    #     plot_sn_strip_two_datasets(
    #         summaries["2A3_combined"]["sig_to_noise"],
    #         summaries["DMS_combined"]["sig_to_noise"]
    #     )
