from __future__ import print_function

from argparse import ArgumentParser
import os
import sys

import matplotlib
matplotlib.use('agg')

THIS_FILE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_FILE_DIR,
                        'deep_disfluency',
                        'deep_disfluency',
                        'data',
                        'disfluency_detection',
                        'switchboard')
sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))
DEFAULT_HELDOUT_DATASET = DATA_DIR + '/swbd_disf_heldout_data_timings.csv'

from model import AWD_LSTM_DisfluencyDetector, MultitaskDisfluencyDetector, save, load
from eval_utils import eval_babi, eval_deep_disfluency


def configure_argument_parser():
    parser = ArgumentParser(description='Evaluate the LSTM dialogue filter')
    parser.add_argument('model_folder')
    parser.add_argument('dataset')
    parser.add_argument('mode', help='[deep_disfluency/babi]')

    return parser


def main(in_dataset_file, in_model_folder, in_mode):
    model, actual_config, vocab, pos_vocab, char_vocab, label_vocab = load(in_model_folder)
    rev_vocab = {word_id: word for word, word_id in vocab.items()} 
    rev_label_vocab = {label_id: label for label, label_id in label_vocab.items()}
    if in_mode == 'deep_disfluency':
        eval_result = eval_deep_disfluency(model,
                                           [(vocab, pos_vocab, label_vocab), (vocab, pos_vocab, vocab)],
                                           in_dataset_file,
                                           actual_config)
    elif in_mode == 'babi':
        eval_result = eval_babi(model,
                                [(vocab, pos_vocab, label_vocab), (vocab, vocab, rev_vocab)],
                                in_dataset_file,
                                actual_config)
    else:
        raise NotImplementedError
    for key, value in eval_result.items():
        print('{}:\t{}'.format(key, value))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset, args.model_folder, args.mode)
