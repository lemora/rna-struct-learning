from rnasl.jax_setup import jfloat

import argparse
import os
import sys

import rnasl.gconst as gc
from rnasl.eval.eval_loss_landscape import analyse_loss_landscape
from rnasl.eval.eval_predictions_reactivities import evaluate_reactivity_predictions
from rnasl.eval.eval_predictions_structures import evaluate_structure_predictions
from rnasl.eval.eval_training import evaluate_training, evaluate_training_old


def parse_command_line(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # global toggles
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    # what to run
    parser.add_argument('--eval_model', action='store_true', help='Evaluate model predictions')
    parser.add_argument('--eval_training', action='store_true', help='Evaluate training log')
    parser.add_argument('--eval_lossscape', action='store_true', help='Analyse loss-landscape')
    parser.add_argument('--eval_baselines', action='store_true', help='Run baselines')
    parser.add_argument('--viennarna', action='store_true', help='Run ViennaRNA')
    parser.add_argument('--eternafold', type=str, help='Path to EternaFold repo')

    # data selection
    parser.add_argument('--data_type', choices=['reactivity', 'structure'],
                        help='Dataset domain for evaluation')
    parser.add_argument('--rtest', type=str, help='CSV with reactivity test data')
    parser.add_argument('--rdata_baselinetype', type=str, default=None,
                        choices=['2A3', '2A3_norm', 'DMS', 'DMS_norm'],
                        help='The reactivity dataset type to add baselines for. If not specified, the training '
                             'dataset type will be used based on the configuration.')

    # trained model dir
    parser.add_argument('--resdir', type=str, help='Output folder from training run')

    args = parser.parse_args(argv)

    # validations
    need_resdir = any((args.eval_model, args.eval_training, args.eval_lossscape))
    if need_resdir and not args.resdir:
        parser.error('--resdir required for the chosen operation(s)')

    need_domain = any((args.eval_model, args.eval_baselines,
                       args.viennarna, args.eternafold))
    if need_domain and not args.data_type:
        parser.error('--data_type required for the chosen operation(s)')

    if args.data_type == 'reactivity':
        if args.rtest is None and any((args.eval_model,
                                       args.viennarna, args.eternafold)):
            parser.error('--rtest required for reactivity runs')

    return args


def run() -> None:
    args = parse_command_line(sys.argv[1:])
    gc.DISPLAY = args.display
    gc.VERBOSE = args.verbose
    h = 3

    energy_mat_path = None
    if args.resdir:
        resdir = args.resdir.rstrip('/')
        energy_mat_path = os.path.join(resdir, 'TrainedEnergies.txt')

    # -------- training-related
    if args.eval_training:
        loss_old = os.path.join(resdir, 'loss_history.csv')
        loss_new = os.path.join(resdir, 'training_log.csv')
        e_hist = os.path.join(resdir, 'energy_history.npy')
        if os.path.isfile(loss_old):
            evaluate_training_old(loss_old, e_hist)
        elif os.path.isfile(loss_new):
            evaluate_training(loss_new, e_hist, args.rdata_baselinetype)

    if args.eval_lossscape:
        analyse_loss_landscape(energy_mat_path)

    # -------- model predictions
    if args.eval_model:
        if args.data_type == 'structure':
            evaluate_structure_predictions(energy_mat_path)
        else:  # reactivity
            evaluate_reactivity_predictions(args.rtest, energy_mat_path)

    # -------- baselines / external tools
    if args.eval_baselines and args.data_type == 'structure':
        evaluate_structure_predictions(None, 3, "baseline_nopairs")
        evaluate_structure_predictions(None, 3, "baseline_fullstem")

    if args.viennarna:
        if args.data_type == 'structure':
            evaluate_structure_predictions(None, 3, "viennarna")
        else:
            evaluate_reactivity_predictions(args.rtest, energy_mat_path=None, predictor='viennarna')

    if args.eternafold:
        if args.data_type == 'structure':
            evaluate_structure_predictions(None, 3, "eternafold", args.eternafold)
        else:
            evaluate_reactivity_predictions(args.rtest, energy_mat_path=None, predictor='eternafold',
                                            predictor_path=args.eternafold)


if __name__ == "__main__":
    run()
