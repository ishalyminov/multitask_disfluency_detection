from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd

from data_utils import make_vocabulary, make_char_vocabulary
from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    save,
                                    make_dataset,
                                    MAX_INPUT_LENGTH,
                                    MAX_VOCABULARY_SIZE,
                                    MODEL_NAME,
                                    get_class_weight_proportional)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')

    return parser


def main(in_dataset_folder, in_model_folder):
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    vocab, _ = make_vocabulary(trainset['utterance'].values, MAX_VOCABULARY_SIZE)
    char_vocab = make_char_vocabulary()
    label_vocab, _ = make_vocabulary(trainset['tags'].values, MAX_VOCABULARY_SIZE, special_tokens=[])
    # label_vocab = {key: idx for idx, key in enumerate(filter(lambda x: x.startswith('<rm-4'), label_vocab.keys()))}
    X_train, y_train = make_dataset(trainset, vocab, char_vocab, label_vocab)
    X_dev, y_dev = make_dataset(devset, vocab, char_vocab, label_vocab)
    X_test, y_test = make_dataset(testset, vocab, char_vocab, label_vocab)
    class_weight = get_class_weight_proportional(np.argmax(y_train, axis=-1))
    save(None, vocab, char_vocab, label_vocab, in_model_folder, save_model=False)

    model = create_model(len(vocab), 256, MAX_INPUT_LENGTH, len(label_vocab))
    train(model,
          (X_train[0], y_train),
          (X_dev[0], y_dev),
          (X_test[0], y_test),
          os.path.join(in_model_folder, 'ckpt'),
          label_vocab,
          class_weight,
          learning_rate=0.01,
          batch_size=32,
          epochs=100,
          steps_per_epoch=1000)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder)

