from rnasl.jax_setup import jfloat

import argparse
import sys
import os

import rnasl.gconst as gc

from rnasl.utils.helper import ensure_rna_bases, isXNA
import rnasl.folding.nussinov_mfe as nussmfe
import rnasl.folding.nussinov_pf as nusspf
import rnasl.folding.nussinov_pf_jax as nusspfx
import rnasl.folding.nussinov_bpps as nussbpps
import rnasl.folding.pcfg_nltk as pcfg_nltk
from rnasl.folding_primitives.pairing_energies import PairingEnergies
from rnasl.io.experiment_io import energy_mat_from_file, read_in_seq, write_rna_struct_to_file

DEFAULT_RNA = "GGGUCGUUAGCUCAGUUGGUAGAGCAGUUGACUUUUAAUCAAUUGGUCGCAGGUUCGAAUCCUGCACGACCCA"  # E. coli tRNA


def parse_command_line(argv) -> argparse.ArgumentParser:
    """Process command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true', default=False, help='enable verbose mode')
    p.add_argument('-d', '--display', action='store_true', default=False,
                   help='display the visualized intermediate results')

    p.add_argument('-s', '--seq', type=str, required=True,
                   help='The RNA to predict a structure for (accepts *.seq, *.dbn or string)')
    p.add_argument('-a', '--alg', type=str, default="nuss_pfx", choices=["pcfg", "nuss_mfe", "nuss_pf", "nuss_pfx"],
                   help="The algorithm to use for MFE structure prediction.")
    p.add_argument('-m', '--mode', type=str, default="mfe", choices=["mfe", "ensemble"],
                   help="The type of prediction to make.")
    p.add_argument('-p', '--params', type=str, help="Which prameters to use for the prediction.")
    p.add_argument('-t', '--temp', type=float, default=gc.TEMP, help='Temperature in kelvin.')
    p.add_argument('-l', type=int, default=3, help='Minimum loop length.')
    p.add_argument('-o', '--out', type=str, help='The filename to write the predicted structure into.')
    args = p.parse_args(argv)
    return args


def run():
    argv = sys.argv[1:]
    args = parse_command_line(argv)

    gc.VERBOSE = args.verbose
    gc.DISPLAY = args.display

    if gc.VERBOSE:
        print("----------------------")

    if args.temp < 0:
        print(r"ERR: The temperature t must be >= 0 K.")
        quit()
    gc.TEMP = args.temp

    if args.l < 0:
        print(r"ERR: The minimum loop length l must be >= 0.")
        quit()

    rna_seq = read_in_seq(args.seq)
    rna_seq = ensure_rna_bases(rna_seq)
    if not isXNA(rna_seq):
        print(rf"ERR: Not a valid RNA string: {rna_seq}")
        quit()

    print(f"sequence (length: {len(rna_seq)}): {rna_seq}")
    print(f"length: {len(rna_seq)}")
    print("----------------------\n")

    energy_mat = energy_mat_from_file(args.params)
    dirpath = os.path.dirname(os.path.abspath(args.params))

    if args.alg == "pcfg":
        print("Running PCFG")
        vienna_struct = pcfg_nltk.predict_structure(rna_seq)
    elif args.alg == "nuss_mfe":
        print("Running MFE nussinov")
        pairing_energies = PairingEnergies(["A", "C", "G", "U"], energy_mat)
        vienna_struct = nussmfe.predict_structure(rna_seq, pairing_energies, args.l)
    elif args.alg == "nuss_pf":
        print("Running partition function nussinov")
        pairing_energies = PairingEnergies(["A", "C", "G", "U"], energy_mat)
        vienna_struct = nusspf.predict_structure(rna_seq, pairing_energies, args.l)
    elif args.alg == "nuss_pfx":
        print("Running partition function nussinov on jax")
        vienna_struct = nusspfx.predict_structure(rna_seq, energy_mat, args.l, dirpath)
        # nussbpps.compare_np_jax(rna_seq, energy_mat, args.l)
        # return

    print(f"\n-- Predicted structure:\n{rna_seq}\n{vienna_struct}\n")
    outname = args.out if args.out else "struct.dbn"
    write_rna_struct_to_file(rna_seq, vienna_struct, outname)


if __name__ == "__main__":
    run()
