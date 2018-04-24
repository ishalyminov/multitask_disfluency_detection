from argparse import ArgumentParser
import os

import pandas as pd

from data_utils import make_vocabulary, make_char_vocabulary, PAD
from dialogue_denoiser_lstm import (create_simple_model,
                                    train,
                                    evaluate,
                                    save,
                                    make_dataset,
                                    MAX_INPUT_LENGTH,
                                    VOCABULARY_SIZE,
                                    MODEL_NAME, MAX_CHAR_INPUT_LENGTH)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')

    return parser


def main(in_dataset_folder, in_model_folder):
    table_to_tuples = lambda x: [(tokens, tags) for tokens, tags in zip(x['utterance'], x['tags'])]

    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    vocab, _ = make_vocabulary(trainset['utterance'].values, VOCABULARY_SIZE)
    char_vocab = make_char_vocabulary()
    label_vocab, _ = make_vocabulary(trainset['tags'].values, VOCABULARY_SIZE, special_tokens=[PAD])
    X_train, y_train, weights_train = make_dataset(table_to_tuples(trainset),
                                                   vocab,
                                                   char_vocab,
                                                   label_vocab)
    X_dev, y_dev, weights_dev = make_dataset(table_to_tuples(devset),
                                              vocab,
                                              char_vocab,
                                              label_vocab)
    X_test, y_test, weights_test = make_dataset(table_to_tuples(testset),
                                                vocab,
                                                char_vocab,
                                                label_vocab)
    save(None, vocab, char_vocab, label_vocab, in_model_folder, save_model=False)

    model = create_simple_model(len(vocab),
                                len(char_vocab),
                                128,  # word embedding size
                                32,  # char embedding size
                                MAX_INPUT_LENGTH,
                                MAX_CHAR_INPUT_LENGTH,
                                len(label_vocab),
                                0.01)
    train(model,
          ([X_train[0]], y_train, weights_train),
          ([X_dev[0]], y_dev, weights_dev),
          ([X_test[0]], y_test, weights_test),
          os.path.join(in_model_folder, MODEL_NAME),
          label_vocab,
          batch_size=32,
          epochs=100,
          steps_per_epoch=10)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder)

