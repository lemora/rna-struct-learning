import csv
import glob
import os
from collections import Counter

import numpy as np
from datasets import concatenate_datasets, load_dataset

from rnasl.utils.formats import encode_seq_jax, vienna_to_pairing

from rnasl.utils.helper import is_valid_non_crossing


def load_reactivity_dataset(filepath: str, max_count=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """ Returns a generator for tuples: (encoded seq, list of float reactivities) """
    dataset = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        reactivity_cols = [col for col in reader.fieldnames if col.startswith('reactivity_0')]

        for i, row in enumerate(reader):
            if max_count and i >= max_count:
                break
            try:
                seq = row['sequence'].strip()
                if not seq:
                    continue

                SN_filter = float(row.get('SN_filter', '1.0').strip())
                if SN_filter < 1:
                    continue

                encoded_seq = encode_seq_jax(seq)
                reactivities_raw = [row[col].strip() for col in reactivity_cols[:len(seq)]]
                reactivities = np.array([
                    float(v) if v and v.lower() != "null" else -np.nan
                    for v in reactivities_raw
                ], dtype=np.float32)

                if len(encoded_seq) != len(reactivities):
                    continue

                dataset.append((encoded_seq, reactivities))
            except Exception:
                continue

    dataset = sorted(dataset, key=lambda x: len(x[0]))  # sorting by seq len for jax speedup
    return dataset


def load_structure_dataset_eterna(struct_dir: str, max_count=None) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
    """ Returns a generator for tuples: (encoded seq, list of integer base pairs) """
    dataset = []
    total_loaded = 0

    for filepath in sorted(glob.glob(os.path.join(struct_dir, "*.bpseq"))):
        if max_count is not None and total_loaded >= max_count:
            break

        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            seq = []
            base_pairs = set()
            for line in lines:
                if not line.strip() or line.startswith("#"):
                    continue
                tokens = line.strip().split()
                if len(tokens) < 3:
                    continue
                i, base, pair_idx = int(tokens[0]) - 1, tokens[1], int(tokens[2]) - 1
                seq.append(base)
                if pair_idx >= 0 and i < pair_idx:
                    base_pairs.add((i, pair_idx))

            if not is_valid_non_crossing(base_pairs):
                print(f"Skipping {filepath}: crossing pairs detected.")
                continue

            encoded_seq = encode_seq_jax("".join(seq))
            dataset.append((encoded_seq, base_pairs))
            total_loaded += 1

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    return dataset


def load_structure_dataset_archiveii(max_count=None) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
    """ Returns a generator for tuples: (encoded seq, list of integer base pairs) """
    ds_dict = load_dataset("multimolecule/archiveii")
    dataset_raw = concatenate_datasets(list(ds_dict.values()))

    dataset = []
    families = []
    total_loaded = 0

    seen = set()
    skipped = 0

    for item in dataset_raw:
        if max_count is not None and total_loaded >= max_count:
            break

        try:
            seq = item["sequence"]
            struct = item["secondary_structure"]
            family = item["family"]
            id = item["id"]

            if seq in seen:
                skipped += 1
                continue  # skip duplicate seq
            seen.add(seq)

            base_pairs = vienna_to_pairing(struct)
            if not is_valid_non_crossing(base_pairs):
                print(f"Has crossing pairs {id}")
                continue

            encoded_seq = encode_seq_jax(seq)
            dataset.append((encoded_seq, base_pairs))
            families.append(family)
            total_loaded += 1

        except Exception as e:
            print(f"Error processing entry {item.get('id', '')}: {e}")
            continue

    family_counts = Counter(families)
    print(f"Unique families in dataset: {family_counts}")
    print(f"Removed {skipped} redundant sequences")
    print(f"Kept {total_loaded} unique sequences")
    dataset = sorted(dataset, key=lambda x: len(x[0]))
    return dataset
