from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd

from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary
from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    save,
                                    make_dataset,
                                    MODEL_NAME,
                                    get_class_weight_proportional)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)

    return parser


def main(in_dataset_folder, in_model_folder, in_config):
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    if in_config['use_pos_tags']:
        utterances = []
        for utterance, postags in zip(trainset['utterance'], trainset['pos']):
            utterance_augmented = ['{}_{}'.format(token, pos)
                                   for token, pos in zip(utterance, postags)]
            utterances.append(utterance_augmented)
    else:
        utterances = trainset['utterance']
    vocab, _ = make_vocabulary(utterances, in_config['max_vocabulary_size'])
    char_vocab = make_char_vocabulary()
    label_vocab, _ = make_vocabulary(trainset['tags'].values,
                                     in_config['max_vocabulary_size'],
                                     special_tokens=[])
    rev_label_vocab = {label_id: label
                       for label, label_id in label_vocab.iteritems()}
    X_train, y_train = make_dataset(trainset, vocab, label_vocab, in_config)
    X_dev, y_dev = make_dataset(devset, vocab, label_vocab, in_config)
    X_test, y_test = make_dataset(testset, vocab, label_vocab, in_config)
    class_weight = get_class_weight_proportional(np.argmax(y_train, axis=-1))

    save(in_config, vocab, char_vocab, label_vocab, in_model_folder)

    model = create_model(len(vocab),
                         in_config['embedding_size'],
                         in_config['max_input_length'],
                         len(label_vocab))
    train(model,
          (X_train, y_train),
          (X_dev, y_dev),
          (X_test, y_test),
          os.path.join(in_model_folder, MODEL_NAME),
          vocab,
          label_vocab,
          rev_label_vocab,
          config)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.dataset_folder, args.model_folder, config)
