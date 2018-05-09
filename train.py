from argparse import ArgumentParser
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary
from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    save,
                                    make_dataset,
                                    load)


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--resume', action='store_true', default=False)

    return parser


def init_model(trainset, in_model_folder, resume, in_config, in_session):
    model = None
    with tf.Session() as sess:
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


def filter_rms(in_X, in_y, in_rev_label_vocab):
    labels = map(in_rev_label_vocab.get, in_y)
    sample_index = filter(lambda x: x.startswith('<rm') or x.startswith('<rp'), labels)

    X_result, y_result = np.take(in_X, sample_index, axis=0), np.take(in_y, sample_index, axis=0)
    return X_result, y_result


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
        X_train_full, y_train_full = make_dataset(trainset, vocab, label_vocab, actual_config)
        X_dev_full, y_dev_full = make_dataset(devset, vocab, label_vocab, actual_config)
        X_test, y_test = make_dataset(testset, vocab, label_vocab, actual_config)

        print 'Stage 1: pre-training the model on full data'
        train(model,
              (X_train_full, y_train_full),
              (X_dev_full, y_dev_full),
              (X_test, y_test),
              vocab,
              label_vocab,
              rev_label_vocab,
              in_model_folder,
              actual_config['pretraining_epochs_number'],
              actual_config,
              sess)


        X_train_rm, y_train_rm = filter_rms(X_train_full, y_train_full, rev_label_vocab)

        print 'Stage 2: training the model on disfluencies'
        train(model,
              (X_train_rm, y_train_rm),
              (X_dev_full, y_dev_full),
              (X_test, y_test),
              vocab,
              label_vocab,
              rev_label_vocab,
              in_model_folder,
              actual_config['epochs_number'],
              actual_config,
              sess)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.dataset_folder, args.model_folder, args.resume, config)
