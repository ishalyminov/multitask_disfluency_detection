import json
from argparse import ArgumentParser

import os

from dialogue_denoiser_lstm import (load_dataset,
                                    make_dataset,
                                    load,
                                    make_tagger_data_points,
                                    evaluate,
                                    MAX_INPUT_LENGTH)


def configure_argument_parser():
    parser = ArgumentParser(description='Evaluate the LSTM dialogue filter')
    parser.add_argument('from_file')
    parser.add_argument('to_file')
    parser.add_argument('model_folder')

    return parser


def main(in_from, in_to, in_model_folder):
    encoder_lines, decoder_lines = load_dataset(in_from, in_to)
    data_points = make_tagger_data_points(encoder_lines, decoder_lines)
    model, vocab = load(in_model_folder)
    X, y = make_dataset(data_points, vocab)

    print 'Accuracy: {:.3f}'.format(evaluate(model, X, y))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.from_file, args.to_file, args.model_folder)
