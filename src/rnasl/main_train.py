from rnasl.jax_setup import jfloat

import argparse
import sys

import yaml

import rnasl.gconst as gc
import rnasl.training.nussinov_train as nussinov
from rnasl.training.loss_train_configs import LossConfig, TrainConfig


def parse_command_line(argv) -> argparse.ArgumentParser:
    """Process command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--verbose', action='store_true', default=False, help='enable verbose mode')
    p.add_argument('-d', '--display', action='store_true', default=False,
                   help='display the visualized intermediate results')

    p.add_argument('data', type=str, help='The training data file to train the model on')
    p.add_argument('--validate', type=str, help='The validation data file for training the model')
    p.add_argument('--outdir', type=str, help="The name of the folder in which to save the training results.")
    p.add_argument('--config', type=str, help="YAML config file (overridden by CLI arguments).")

    p.add_argument('--lr', type=float, help='Learning rate.')
    p.add_argument('--steps', type=int, help='Max training step count.')
    p.add_argument('--examples_per_step', type=int, help='Number of examples per training step.')
    p.add_argument('--val_examples_per_step', type=int, help='Number of examples to validate on per training step.')
    p.add_argument('--patience', type=int,
                   help='early stopping patience: number of steps without improvement before early stopping.')

    p.add_argument('-a', '--alg', type=str, default="nussinov", choices=["nussinov"],
                   help="The algorithm to train for structure prediction.")
    args = p.parse_args(argv)
    return args


def merge_config_with_cli(cli_args: argparse.Namespace) -> dict:
    cli = vars(cli_args)
    config_path = cli.get("config")

    yaml_cfg = {}
    if config_path:
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    merged = {
        **yaml_cfg.get("train_config", {}),  # stays nested: no overrrides
        "loss_config": yaml_cfg.get("loss_config", {}).copy(),
        "train_file": yaml_cfg.get("train_file"),
        "val_file": yaml_cfg.get("val_file"),
        "outdir": yaml_cfg.get("outdir"),
    }

    # apply CLI overrides
    for key, value in cli.items():
        if value is None:
            continue
        if key in LossConfig.__annotations__:
            merged["loss_config"][key] = value
        else:
            merged[key] = value

    return merged


def filtered_dataclass_init(cls, raw_dict):
    return cls(**{k: v for k, v in raw_dict.items() if k in cls.__annotations__})


def run():
    argv = sys.argv[1:]
    args = parse_command_line(argv)

    gc.DISPLAY = args.display
    gc.VERBOSE = args.verbose

    merged_args = merge_config_with_cli(args)

    trainconf = filtered_dataclass_init(TrainConfig, merged_args)
    lossconf = filtered_dataclass_init(LossConfig, merged_args.get("loss_config", {}))

    if args.alg == "nussinov":
        print("Running training via partition function nussinov")
        nussinov.train(args.data,
                       args.validate,
                       args.outdir,
                       trainconf,
                       lossconf)


if __name__ == "__main__":
    run()
