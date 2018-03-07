import json
from argparse import ArgumentParser

import os

from dialogue_denoiser_lstm import (load_dataset,
                                    make_training_data,
                                    create_model,
                                    train,
                                    evaluate,
                                    MAX_INPUT_LENGTH)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('from_file')
    parser.add_argument('to_file')
    parser.add_argument('model_folder')

    return parser


def main(in_from, in_to, in_model_folder):
    encoder_lines, decoder_lines = load_dataset(in_from, in_to)
    vocab, train_data, dev_data, test_data = make_training_data(encoder_lines, decoder_lines)

    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    with open(os.path.join(in_model_folder, 'vocab.json'), 'w') as vocab_out:
        json.dump(vocab, vocab_out)

    model = create_model(len(vocab), 128, MAX_INPUT_LENGTH, 2, 0.01)
    train(model, train_data, dev_data, test_data, os.path.join(in_model_folder, 'model.h5'))
    print 'Testset accuracy: {:.3f}'.format(evaluate(model, *test_data))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.from_file, args.to_file, args.model_folder)
