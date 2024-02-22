import argparse
import sys

from rnasl.visualization.visualize import plot_2d_struct


def parse_command_line(argv) -> argparse.ArgumentParser:
    """Process command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("rna")
    p.add_argument("structure")
    p.add_argument('-d', '--display', action='store_true', default=False,
                   help='display the visualized intermediate results')
    args = p.parse_args(argv)
    return args


def run():
    argv = sys.argv[1:]
    args = parse_command_line(argv)
    if not (args.rna and args.structure):
        raise Exception("RNA and structure must be given for visualization. Exiting.")
    print(f"rna: {args.rna} || atruct: {args.structure}")

    plot_2d_struct(args.rna, args.structure, show=args.display)


if __name__ == "__main__":
    run()
