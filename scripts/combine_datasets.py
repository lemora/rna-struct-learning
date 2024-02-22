import os
import csv
import hashlib
# import polars as pl
import shutil

import rnasl.gconst as gc

DATASET_ROOT = "/home/lemora/Dev/data/rna/"
NUM_REACTIVITIES = 206


def pad_reactivities(reactivities):
    if not reactivities:
        return [""] * NUM_REACTIVITIES
    elif len(reactivities) == NUM_REACTIVITIES:
        return reactivities
    elif len(reactivities) > NUM_REACTIVITIES:
        raise Exception("> 206 reactivity values detected. Aborting")

    padding_needed = NUM_REACTIVITIES - len(reactivities)
    left_padding = padding_needed // 2
    right_padding = padding_needed - left_padding
    padded_reactivities = [""] * left_padding + reactivities + [""] * right_padding
    return padded_reactivities


def write_csv_line(outfile: str, seq_id: str, seq: str, exp_type: str, dataset_name: str, reads: int,
                   sig_to_noise: float, sn_filter: int, reactivities: list, react_errs: list = None):
    # csv line format:
    # sequence_id,sequence,experiment_type,dataset_name,reads,signal_to_noise,SN_filter,[206 reactivities],[206 errors]
    if not reactivities:
        raise Exception("Empty reactivities!")
    if not react_errs:
        react_errs = [""] * 206

    new_row = [seq_id, seq, exp_type, dataset_name, reads, sig_to_noise, sn_filter] + reactivities + react_errs
    with open(outfile, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)
    return

def generate_id(prefix: str, sequence: str):
    dig = hashlib.sha256(b"{prefix}{sequence}").hexdigest()
    return dig

def process_bpp_file(filepath):
    bases = []
    reactivities = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            bases.append(parts[1])
            reactivities.append(parts[3])
    sequence = ''.join(bases)
    # print(f"seq: {sequence}, react: {reactivities}")
    return sequence, reactivities


def process_add_eternafold(outfile):
    chemmapdir = f"{DATASET_ROOT}eternafold/ChemMappingData/"
    i = 0
    for folder in ["train", "test", "holdout"]:
        thedir = f"{chemmapdir}{folder}"
        if gc.VERBOSE: print(f"Processing dir {thedir}")

        for filename in os.listdir(thedir):
            if not filename.endswith("bpseq"):
                continue
            filepath = os.path.join(thedir, filename)
            if not os.path.isfile(filepath):
                continue
            if gc.VERBOSE: print(f"Processing file: {filepath}")
            seq, react = process_bpp_file(filepath)
            # id = generate_id("{i}eterna", seq)
            write_csv_line(outfile, f"{i}eterna", seq, exp_type="unknown", dataset_name="eterna", reads=False, sig_to_noise=False,
                           sn_filter=0, reactivities=pad_reactivities(react), react_errs=None)
            i += 1

    print(f"Eternafold data added to {outfile}")
    return


def process_add_ribonanza(outfile):
    datafile = f"{DATASET_ROOT}ribonanza/stanford-ribonanza-rna-folding/train_data.csv"
    with open(datafile, 'r') as file1:
        with open(outfile, 'a') as file2:
            shutil.copyfileobj(file1, file2)
    print(f"Ribonanza data added to {outfile}")
    return

def to_fasta(infile, outfile):
    seen_sequences = set()

    with open(infile, 'r') as file1, open(outfile, 'w') as file2:
        header = True
        for line in file1:
            if header:
                header = False
                continue

            words = line.strip().split(',')
            if len(words) < 2:
                continue

            seq_id = words[0].strip()
            sequence = words[1].strip().upper()

            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            file2.write(f">{seq_id}\n{sequence}\n")

    print(f"Written {outfile} with {len(seen_sequences)} unique sequences.")

                

def combine_datasets():
    outfile = "data/combined_cmap_data"
    if os.path.isfile(f"{outfile}.csv"):
        print("Combined csv file already exists.")
    else:
        process_add_ribonanza(f"{outfile}.csv")
        process_add_eternafold(f"{outfile}.csv")
        print(f"Combined data saved to {outfile}.csv")

    if os.path.isfile(f"{outfile}.fasta"):
        print("Combined fasta file already exists.")
    else:
        to_fasta(f"{outfile}.csv", f"{outfile}.fasta")
        print(f"Combined data saved to {outfile}.fasta")


if __name__ == "__main__":
    print("Combining datasets....")
    combine_datasets()
