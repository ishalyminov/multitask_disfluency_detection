from argparse import ArgumentParser
import os
from operator import itemgetter

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary
from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    save,
                                    make_dataset,
                                    load,
                                    get_class_weight_proportional,
                                    get_sample_weight)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--resume', action='store_true', default=False)

    return parser


def init_model(trainset, in_model_folder, resume, in_config, in_session):
    model = None
    if not resume:
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
        model = create_model(len(vocab),
                             in_config['embedding_size'],
                             in_config['max_input_length'],
                             len(label_vocab))
        init = tf.global_variables_initializer()
        in_session.run(init)
        save(in_config, vocab, char_vocab, label_vocab, in_model_folder, in_session)
    model, actual_config, vocab, char_vocab, label_vocab = load(in_model_folder,
                                                                in_session,
                                                                existing_model=model)
    return model, actual_config, vocab, char_vocab, label_vocab


def main(in_dataset_folder, in_model_folder, resume, in_config):
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    with tf.Session() as sess:
        model, actual_config, vocab, char_vocab, label_vocab = init_model(trainset,
                                                                          in_model_folder,
                                                                          resume,
                                                                          in_config,
                                                                          sess)
        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}
        X_train, y_train = make_dataset(trainset, vocab, label_vocab, actual_config)
        X_dev, y_dev = make_dataset(devset, vocab, label_vocab, actual_config)
        X_test, y_test = make_dataset(testset, vocab, label_vocab, actual_config)

        y_train_flattened = np.argmax(y_train, axis=-1)
        class_weight = get_class_weight_proportional(y_train_flattened,
                                                     smoothing_coef=actual_config['class_weight_smoothing_coef'])
        sample_weights = get_sample_weight(y_train_flattened, class_weight)

        scaler = MinMaxScaler(feature_range=(1, 5))
        class_weight_vector = scaler.fit_transform(np.array(map(itemgetter(1), sorted(class_weight.items(), key=itemgetter(0)))).reshape(-1, 1)).flatten()

        train(model,
              (X_train, y_train),
              (X_dev, y_dev),
              (X_test, y_test),
              vocab,
              label_vocab,
              rev_label_vocab,
              in_model_folder,
              actual_config['epochs_number'],
              actual_config,
              sess,
              class_weight=class_weight_vector)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.dataset_folder, args.model_folder, args.resume, config)
