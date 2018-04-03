import random
from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np

from dialogue_denoiser_lstm import (evaluate,
                                    load,
                                    MODEL_NAME,
                                    save,
                                    make_dataset,
                                    train)

random.seed(273)
np.random.seed(273)


def configure_argument_parser():
    parser = ArgumentParser(description='Fine-tune a LSTM dialogue filter model')
    parser.add_argument('model_folder')
    parser.add_argument('dataset')
    parser.add_argument('result_model_folder')
    parser.add_argument('--epochs_number', type=int, default=1)
    parser.add_argument('--trainset_sample_size', type=int, default=10)

    return parser


def main(in_dataset, in_model_folder, in_trainset_size, in_epochs_number, in_result_folder):
    model, vocab, label_vocab = load(in_model_folder)
    in_dataset = in_dataset.sample(frac=1).reset_index(drop=True)
    trainset, testset = in_dataset[:in_trainset_size], in_dataset[in_trainset_size:]
    train_data_points = [(tokens, tags) for tokens, tags in zip(trainset['utterance'], trainset['tags'])]
    test_data_points = [(tokens, tags) for tokens, tags in zip(testset['utterance'], testset['tags'])]
    train_data = make_dataset(train_data_points, vocab, label_vocab)
    test_data = make_dataset(test_data_points, vocab, label_vocab)

    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)

    train(model,
          train_data,
          test_data,
          test_data,
          os.path.join(in_result_folder, MODEL_NAME),
          epochs=in_epochs_number,
          batch_size=1)
    save(model, vocab, label_vocab, in_result_folder, save_model=False)

    print 'Testset accuracy: {:.3f}'.format(evaluate(model, *test_data))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset),
         args.model_folder,
         args.trainset_sample_size,
         args.epochs_number,
         args.result_model_folder)

