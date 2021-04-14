import argparse
import os.path as osp

from guacamol.utils.helpers import setup_default_logger

from LambdaZero.examples.baselines.guacamol_baseline.smiles_lstm_hc.smiles_rnn_directed_generator import SmilesRnnDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
# from LambdaZero.examples.baselines.guacamol_baseline.assess_goal_directed_generation import assess_goal_directed_generation
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if __name__ == '__main__':
    setup_default_logger()

    parser = argparse.ArgumentParser(description='Goal-directed generation benchmark for SMILES RNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default=None, help='Full path to the pre-trained SMILES RNN model')
    parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default=None, help='Output directory for results')
    parser.add_argument('--number_repetitions', default=1, type=int, help='Number of re-training runs to average')
    parser.add_argument('--keep_top', default=512, type=int, help='Molecules kept each step')
    parser.add_argument('--n_epochs', default=20, type=int, help='Epochs to sample')
    parser.add_argument('--mols_to_sample', default=1024, type=int, help='Molecules sampled at each step')
    parser.add_argument('--optimize_batch_size', default=256, type=int, help='Batch size for the optimization')
    parser.add_argument('--optimize_n_epochs', default=2, type=int, help='Number of epochs for the optimization')
    parser.add_argument('--benchmark_num_samples', default=4096, type=int,
                        help='Number of molecules to generate from final model for the benchmark')
    parser.add_argument('--benchmark_trajectory', action='store_true',
                        help='Take molecules generated during re-training into account for the benchmark')
    parser.add_argument('--smiles_file', default=osp.join(datasets_dir, 'guacamol_baseline/data/guacamol_v1_all.smiles'))
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--suite', default='proxy')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = osp.dirname(osp.realpath(__file__))

    if args.model_path is None:
        dir_path = osp.dirname(osp.realpath(__file__))
        args.model_path = osp.join(datasets_dir, 'guacamol_baseline/smiles_lstm_hc/pretrained_model', 'model_final_0.473.pt')

    optimizer = SmilesRnnDirectedGenerator(pretrained_model_path=args.model_path,
                                           n_epochs=args.n_epochs,
                                           mols_to_sample=args.mols_to_sample,
                                           keep_top=args.keep_top,
                                           optimize_n_epochs=args.optimize_n_epochs,
                                           max_len=args.max_len,
                                           optimize_batch_size=args.optimize_batch_size,
                                           number_final_samples=args.benchmark_num_samples,
                                           random_start=args.random_start,
                                           smi_file=args.smiles_file,
                                           n_jobs=args.n_jobs)

    json_file_path = osp.join(args.output_dir, 'goal_directed_results.json')
    assess_goal_directed_generation(optimizer, json_output_file=json_file_path, benchmark_version=args.suite)
