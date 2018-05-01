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
        model, vocab, char_vocab, label_vocab, eval_label_vocab = load(in_model_folder, sess)
        X_test, y_test = make_dataset(in_dataset, vocab, char_vocab, label_vocab)

        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}
        _, eval_map = evaluate(model,
                               (X_test, y_test),
                               get_tag_mapping(label_vocab),
                               sess)
        print 'Evaluation results:'
        print ' '.join(['{}: {:.3f}'.format(key, value) for key, value in eval_map.iteritems()])


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset), args.model_folder)
