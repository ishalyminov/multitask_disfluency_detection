from argparse import ArgumentParser

import tensorflow as tf
import pandas as pd

from dialogue_denoiser_lstm import make_dataset, load, evaluate
from deep_disfluency_utils import get_tag_mapping


def configure_argument_parser():
    parser = ArgumentParser(description='Evaluate the LSTM dialogue filter')
    parser.add_argument('model_folder')
    parser.add_argument('dataset')

    return parser


def main(in_dataset, in_model_folder):
    with tf.Session() as sess:
        model, vocab, char_vocab, label_vocab = load(in_model_folder, sess)
        X_test, y_test = make_dataset(in_dataset, vocab, char_vocab, label_vocab)

        tag_map = get_tag_mapping(label_vocab)
        eval_map = evaluate(model, (X_test[0], y_test), tag_map, sess)
        print 'Evaluation results:'
        print ' '.join(['{}: {:.3f}'.format(key, value) for key, value in eval_map.iteritems()])

if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset), args.model_folder)
