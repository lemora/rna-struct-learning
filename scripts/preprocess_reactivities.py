import csv
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
import os

from rnasl.utils.helper import normalize_reactivities_per_base_np


# ------ helper functions

def get_outfile_name(input_path: str, fname: str, subfolder=None, fname_as_suffix=True):
    infile = Path(input_path)
    base = infile.with_suffix('')

    if subfolder:
        out_dir = base.parent / subfolder
        out_dir.mkdir(exist_ok=True)
    else:
        out_dir = base.parent

    if fname_as_suffix:
        outfile_dms = out_dir / (base.name + fname)
    else:
        outfile_dms = out_dir / fname

    return outfile_dms


# ------ transform original file to unique id one

def combined_to_unique_id_file(input_path: str, outfile: str, force=False):
    if os.path.isfile(outfile):
        if force:
            os.remove(outfile)
            print(f"Deleted: {outfile}")
        else:
            print("Csv file with unique file ids already exists. Doing nothing.")
            return

    with (open(input_path, newline='') as infile,
          open(outfile, mode='w', newline='') as outf):
        reader = csv.DictReader(infile)

        reactivity_cols = [col for col in reader.fieldnames if col.startswith('reactivity_0')]
        output_fields = ['sequence_id', 'sequence'] + reactivity_cols

        writer = csv.DictWriter(outf, fieldnames=output_fields)
        writer.writeheader()

        i = 0
        for row in reader:
            dataset_name = row['dataset_name']
            original_id = row['sequence_id']
            unique_id = f"{original_id}_{dataset_name}"  # Combine ID + dataset_name -> new unique ID
            output_row = {
                'sequence_id': unique_id,
                'sequence': row['sequence'], }
            for col in reactivity_cols:
                output_row[col] = row.get(col, '')
            i += 1
            if i % 10000 == 0:
                print(f"processed {i} rows")
            writer.writerow(output_row)

    return


# ------ split original file by experiment type (DMS, 2A3)

def split_by_exp_type(input_path: str, force=False):
    outfile_dms = get_outfile_name(input_path, "_dms.csv", "DMS")
    outfile_2a3 = get_outfile_name(input_path, "_2a3.csv", "2A3")
    if os.path.isfile(outfile_dms) or os.path.isfile(outfile_2a3):
        if force:
            os.remove(outfile_dms)
            os.remove(outfile_2a3)
            print(f"Deleted: {outfile_dms} and {outfile_2a3}")
        else:
            print("Files split by experiment type already exist. Doing nothing.")
            return

    with (open(input_path, newline='') as infile,
          open(outfile_dms, mode='w', newline='') as out_dms,
          open(outfile_2a3, mode='w', newline='') as out_2a3):
        reader = csv.DictReader(infile)

        reactivity_cols = [col for col in reader.fieldnames if col.startswith('reactivity_')]
        output_fields = ['sequence_id', 'sequence', 'signal_to_noise', 'SN_filter'] + reactivity_cols

        writer_dms = csv.DictWriter(out_dms, fieldnames=output_fields)
        writer_dms.writeheader()
        writer_2a3 = csv.DictWriter(out_2a3, fieldnames=output_fields)
        writer_2a3.writeheader()

        i = 0
        for row in reader:
            exp_type = row['experiment_type']
            output_row = {key: row[key] for key in output_fields}
            if exp_type == "2A3_MaP":
                writer_2a3.writerow(output_row)
            elif exp_type == "DMS_MaP":
                writer_dms.writerow(output_row)
            elif exp_type == "unknown":
                writer_2a3.writerow(output_row)
                writer_dms.writerow(output_row)
            else:
                print(f"Unknown experiment type {exp_type}! Not adding anywhere")

            i += 1
            if i % 10000 == 0:
                print(f"processed {i} rows")

    print(f"Split {input_path} by experiment type")
    return


# ------ split train/test/validation by assigning clusters

def create_train_valid_test_split(cluster_map, test_frac=0.10, val_frac=0.10, seed=42):
    """
    Split clusters into train, validation, test sets based on sequence counts (not cluster counts).
    All clusters containing 'eterna' sequences go to test set.
    return: meta_cluster_map (dict): {"train": [...], "validation": [...], "test": [...]} containing cluster ids
    """
    random.seed(seed)
    tvt_cluster_map = {"train": [], "validation": [], "test": []}
    eterna_suffix = "eterna"

    total_seqs = sum(len(members) for members in cluster_map.values())
    target_test = int(total_seqs * test_frac)
    target_val = int(total_seqs * val_frac)

    eterna_clusters = set()
    for cluster_id, members in cluster_map.items():
        if any(seq.endswith(eterna_suffix) for seq in members):
            eterna_clusters.add(cluster_id)

    assigned = set()
    num_test_seqs = sum(len(cluster_map[cid]) for cid in eterna_clusters)
    tvt_cluster_map["test"].extend(eterna_clusters)
    assigned.update(eterna_clusters)

    remaining_clusters = [cid for cid in cluster_map if cid not in assigned]
    random.shuffle(remaining_clusters)

    # Fill up test set if necessary
    for cid in remaining_clusters:
        if num_test_seqs >= target_test:
            break
        tvt_cluster_map["test"].append(cid)
        num_test_seqs += len(cluster_map[cid])
        assigned.add(cid)

    # Fill validation set
    num_val_seqs = 0
    for cid in remaining_clusters:
        if cid in assigned:
            continue
        if num_val_seqs >= target_val:
            break
        tvt_cluster_map["validation"].append(cid)
        num_val_seqs += len(cluster_map[cid])
        assigned.add(cid)

    # Remaining clusters go to train
    for cid in remaining_clusters:
        if cid not in assigned:
            tvt_cluster_map["train"].append(cid)
    num_train_seqs = total_seqs - num_test_seqs - num_val_seqs
    print(
        f"#seqs per split:\n{num_train_seqs} ({(num_train_seqs / total_seqs) * 100:.2f}%) train,\n{num_val_seqs} ("
        f"{(num_val_seqs / total_seqs) * 100:.2f}%) validation,\n{num_test_seqs} ("
        f"{(num_test_seqs / total_seqs) * 100:.2f}%) test\n")

    return tvt_cluster_map


def write_tvt_cluster_map_to_file(tvt_cluster_map, outfile="tvt_clusters.tsv", force=False):
    if os.path.isfile(outfile):
        if force:
            os.remove(outfile)
            print(f"Deleted: {outfile}")
        else:
            print("Train-Validation-Test cluster file already exists. Doing nothing.")
            return
    with open(outfile, "w") as f:
        for split, cluster_ids in tvt_cluster_map.items():
            for cid in cluster_ids:
                f.write(f"{cid}\t{split}\n")

    print(f"Saved cluster split to {outfile}")


def split_csv_by_cluster(input_csv, output_dir, cluster_map, tvt_split, has_header=True, normalise=False):
    """
    Splits input CSV into TRAIN, VALIDATION, TEST folders based on cluster assignment.
    Optionally applies per-base percentile normalization.
    """
    if os.path.isfile(os.path.join(output_dir, "TRAIN", "data.csv")):
        print(f"Train, validation, test split files for {input_csv} already exist. Doing nothing.")
        return

    dataset_type = "DMS" if "/DMS/" in input_csv else "2A3"
    print(f"Splitting data into train, test, validation sets. Normalise = {normalise}")

    # Maps
    seq_to_cluster = {seq_id: cluster_id for cluster_id, seq_ids in cluster_map.items() for seq_id in seq_ids}
    cid_to_split = {cluster_id: split for split, cluster_ids in tvt_split.items() for cluster_id in cluster_ids}

    # Prepare output directories
    for split_name in ["TRAIN", "VALIDATION", "TEST"]:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    writer_paths = {
        "train": os.path.join(output_dir, "TRAIN", "data.csv"),
        "validation": os.path.join(output_dir, "VALIDATION", "data.csv"),
        "test": os.path.join(output_dir, "TEST", "data.csv"),
    }

    with open(input_csv, "r", newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        reactivity_cols = [col for col in fieldnames if col.startswith('reactivity_')]

        # Open and prepare csv writers
        writers = {}
        files = {}
        for split in ["train", "validation", "test"]:
            fh = open(writer_paths[split], "w", newline='')
            files[split] = fh
            writers[split] = csv.DictWriter(fh, fieldnames=fieldnames)
            writers[split].writeheader()

        for i, row in enumerate(reader):
            seq_id = row['sequence_id'].strip()
            seq = row['sequence'].strip()

            cluster_id = seq_to_cluster.get(seq_id)
            if cluster_id is None:
                print(f"Sequence ID not found in cluster map: {seq_id}")
                continue

            split = cid_to_split.get(cluster_id)
            if split not in writers:
                print(f"Unknown split for cluster {cluster_id}: {split}")
                continue

            if normalise:
                reactivities = np.array([
                    float(row[col]) if row[col].strip() not in ('', 'nan') else np.nan
                    for col in reactivity_cols[:len(seq)]
                ])
                normed = normalize_reactivities_per_base_np(seq, reactivities, dataset_type)
                for j, col in enumerate(reactivity_cols[:len(seq)]):
                    row[col] = f"{normed[j]:.5f}" if np.isfinite(normed[j]) else ""

            writers[split].writerow(row)

            if i % 10000 == 0:
                print(f"Processed row {i}")

        for fh in files.values():
            fh.close()

    print("Split complete: TRAIN/, VALIDATION/, TEST/ folders created.")


# ------ printing/plotting dataset statistics


def plot_cluster_histogram(cluster_sizes, log_scale=True, bins=50):
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_sizes, bins=bins, log=log_scale, edgecolor='black')
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster size")
    plt.ylabel("Count (log)" if log_scale else "Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cluster_size_distribution(clusters, eterna_only_cluster_ids):
    all_sizes = [len(seq_ids) for seq_ids in clusters.values()]
    eterna_sizes = [len(clusters[cid]) for cid in eterna_only_cluster_ids]

    count_all = Counter(all_sizes)
    count_eterna = Counter(eterna_sizes)

    sizes = sorted(set(count_all.keys()).union(count_eterna.keys()))

    all_counts = [count_all.get(s, 0) for s in sizes]
    eterna_counts = [count_eterna.get(s, 0) for s in sizes]

    plt.figure(figsize=(12, 6))
    plt.bar(sizes, all_counts, width=0.8, label="All clusters", color="gray", alpha=0.6)
    plt.bar(sizes, eterna_counts, width=0.5, color='orange', alpha=1.0, label="Eterna-only clusters")
    plt.xlabel("Cluster size")
    plt.ylabel("Number of clusters")
    plt.title("Cluster Size Distribution")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.show()


def cluster_statistics(clusters):
    cluster_sizes = [len(members) for members in clusters.values()]
    total_clusters = len(cluster_sizes)
    total_sequences = sum(cluster_sizes)

    print("\n===================================")
    print(f"---- Cluster statistics ----")
    print(f"Total clusters: {total_clusters}")
    print(f"Total sequences: {total_sequences}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Median cluster size: {np.median(cluster_sizes)}")
    print(f"Max cluster size: {max(cluster_sizes)}")
    print(f"90th percentile: {np.percentile(cluster_sizes, 90)}")
    print(f"99th percentile: {np.percentile(cluster_sizes, 99)}")
    print("===================================\n")

    return cluster_sizes


def analyse_eterna_in_clusters(clusters):
    eterna_cluster_ids = []
    mixed_clusters = []
    eterna_only_clusters = []
    non_eterna_only_clusters = []

    for cid, cluster_members in clusters.items():
        eterna = [s for s in cluster_members if s.endswith("eterna")]
        non_eterna = [s for s in cluster_members if not s.endswith("eterna")]

        if eterna and non_eterna:
            mixed_clusters.append((eterna, non_eterna))
        elif eterna:
            eterna_only_clusters.append(eterna)
            eterna_cluster_ids.append(cid)
        else:
            non_eterna_only_clusters.append(non_eterna)

    print("===================================")
    print(f"---- Eterna clusters ----")
    print(f"Eterna-only clusters: {len(eterna_only_clusters)}")
    print(f"Mixed eterna/non-eterna clusters: {len(mixed_clusters)}")
    print(f"Non-eterna-only clusters: {len(non_eterna_only_clusters)}")
    print("===================================\n")

    return eterna_cluster_ids


# ------ helper and main

def load_tvt_cluster_map_from_file(filepath="tvt_clusters.tsv"):
    tvt_cluster_map = {"train": [], "validation": [], "test": []}
    with open(filepath, "r") as f:
        for line in f:
            cluster_id, split = line.strip().split()
            tvt_cluster_map[split].append(cluster_id)
    return tvt_cluster_map


def read_cluster_map(cluster_tsv_path: str, by_sequence=False):
    """Reads mmseqs cluster file (tsv), creates map. if by_sequence then seq:clusterID else clusterID:seq"""
    cluster_map = defaultdict(list)
    with open(cluster_tsv_path, newline='') as f:
        i = 0
        for line in f:
            cluster_id, seq_id = line.strip().split()
            if by_sequence:
                cluster_map[seq_id].append(cluster_id)
            else:
                cluster_map[cluster_id].append(seq_id)
            i += 1
    return cluster_map


if __name__ == "__main__":
    print("Preprocessing dataset....")
    normalise = True

    # infile = "data/combined_cmap_data.csv"
    # split_by_exp_type(infile)

    # datafile_uid = "data/all_cmap_uid.csv"
    # combined_to_unique_id_file(infile, datafile_uid)

    clusterfile = "data/clusters.tsv"
    cluster_map = read_cluster_map(clusterfile)

    cluster_statistics(cluster_map)
    eterna_only_cluster_ids = analyse_eterna_in_clusters(cluster_map)
    # plot_cluster_size_distribution(cluster_map, eterna_only_cluster_ids)

    tvt_clusters_file = "data/tvt_cluster.tsv"
    if not os.path.exists(tvt_clusters_file):
        tvt_split = create_train_valid_test_split(cluster_map)
        write_tvt_cluster_map_to_file(tvt_split, tvt_clusters_file)
    else:
        tvt_split = load_tvt_cluster_map_from_file(tvt_clusters_file)

    print("Processing the 2A3 dataset")
    dataset_suf = "_norm" if normalise else ""
    filename_2a3 = f"data/2A3{dataset_suf}/combined_cmap_data_2a3.csv"
    split_csv_by_cluster(filename_2a3, f"data/2A3{dataset_suf}/", cluster_map, tvt_split, normalise=normalise)

    print("Processing the DMS dataset")
    filename_dms = f"data/DMS{dataset_suf}/combined_cmap_data_dms.csv"
    split_csv_by_cluster(filename_dms, f"data/DMS{dataset_suf}/", cluster_map, tvt_split, normalise=normalise)
