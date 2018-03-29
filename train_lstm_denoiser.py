from argparse import ArgumentParser
import os

import pandas as pd

from dialogue_denoiser_lstm import (make_training_data,
                                    create_model,
                                    train,
                                    evaluate,
                                    MAX_INPUT_LENGTH, save)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset')
    parser.add_argument('model_folder')

    return parser


def main(in_dataset, in_model_folder):
    vocab, label_vocab, train_data, dev_data, test_data = make_training_data(in_dataset['utterance'].values, in_dataset['tags'].values)
    save(None, vocab, label_vocab, in_model_folder)

    model = create_model(len(vocab), 128, MAX_INPUT_LENGTH, len(label_vocab), 0.01)
    train(model, train_data, dev_data, test_data, os.path.join(in_model_folder, 'model.h5'))
    print 'Testset accuracy: {:.3f}'.format(evaluate(model, *test_data))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset), args.model_folder)
