import string
from argparse import ArgumentParser
import os

import pandas as pd

from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    evaluate,
                                    save,
                                    make_vocabulary,
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
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    vocab, _ = make_vocabulary(trainset['utterance'].values, VOCABULARY_SIZE)
    label_vocab, _ = make_vocabulary(trainset['tags'].values, VOCABULARY_SIZE)
    X_train, y_train = make_dataset([(tokens, tags) for tokens, tags in zip(trainset['utterance'], trainset['tags'])],
                                    vocab,
                                    label_vocab)
    X_dev, y_dev = make_dataset([(tokens, tags) for tokens, tags in zip(devset['utterance'], devset['tags'])], 
                                vocab, 
                                label_vocab)
    X_test, y_test = make_dataset([(tokens, tags) for tokens, tags in zip(testset['utterance'], testset['tags'])], 
                                  vocab, 
                                  label_vocab)
    save(None, vocab, label_vocab, in_model_folder, save_model=False)

    model = create_model(len(vocab),
                         len(string.printable),
                         128,
                         MAX_INPUT_LENGTH,
                         MAX_CHAR_INPUT_LENGTH,
                         len(label_vocab),
                         0.01)
    train(model,
          (X_train, y_train),
          (X_dev, y_dev),
          (X_test, y_test),
          os.path.join(in_model_folder, MODEL_NAME))
    print 'Testset accuracy: {:.3f}'.format(evaluate(model, X_test, y_test))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder)

