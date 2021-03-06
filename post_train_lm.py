from argparse import ArgumentParser
import os
from operator import itemgetter

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary, make_multitask_dataset
from training_utils import get_class_weight_proportional
from dialogue_denoiser_lstm import create_model, train, save, load, post_train_lm


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('main_dataset_folder')
    parser.add_argument('lm_dataset_folder')
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
        task_output_dimensions = []
        for task in in_config['tasks']:
            if task == 'tag':
                task_output_dimensions.append(len(label_vocab))
            elif task == 'lm':
                task_output_dimensions.append(len(vocab))
            else:
                raise NotImplementedError

        model = create_model(len(vocab),
                             in_config['embedding_size'],
                             in_config['max_input_length'],
                             task_output_dimensions)
        init = tf.global_variables_initializer()
        in_session.run(init)
        save(in_config, vocab, char_vocab, label_vocab, in_model_folder, in_session)
    model, actual_config, vocab, char_vocab, label_vocab = load(in_model_folder,
                                                                in_session,
                                                                existing_model=model)
    return model, actual_config, vocab, char_vocab, label_vocab


def main(in_main_dataset_folder, in_lm_dataset_folder, in_model_folder, resume, in_config):
    trainset_main = pd.read_json(os.path.join(in_main_dataset_folder, 'trainset.json'))
    devset_main = pd.read_json(os.path.join(in_main_dataset_folder, 'devset.json'))
    testset_main = pd.read_json(os.path.join(in_main_dataset_folder, 'testset.json'))

    trainset_lm = pd.read_json(os.path.join(in_lm_dataset_folder, 'trainset.json'))
    devset_lm = pd.read_json(os.path.join(in_lm_dataset_folder, 'devset.json'))
    testset_lm = pd.read_json(os.path.join(in_lm_dataset_folder, 'testset.json'))

    with tf.Session() as sess:
        model, actual_config, vocab, char_vocab, label_vocab = init_model(trainset_main,
                                                                          in_model_folder,
                                                                          resume,
                                                                          in_config,
                                                                          sess)
        rev_vocab = {word_id: word
                     for word, word_id in vocab.iteritems()}
        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}
        _, ys_train_main = make_multitask_dataset(trainset_main,
                                                  vocab,
                                                  label_vocab,
                                                  actual_config)
        X_dev_main, ys_dev_main = make_multitask_dataset(devset_main,
                                                         vocab,
                                                         label_vocab,
                                                         actual_config)
        X_test_main, ys_test_main = make_multitask_dataset(testset_main,
                                                           vocab,
                                                           label_vocab,
                                                           actual_config)

        y_train_flattened = np.argmax(ys_train_main[0], axis=-1)
        smoothing_coef = actual_config['class_weight_smoothing_coef']
        class_weight = get_class_weight_proportional(y_train_flattened,
                                                     smoothing_coef=smoothing_coef)

        scaler = MinMaxScaler(feature_range=(1, 5))
        class_weight_vector = scaler.fit_transform(np.array(map(itemgetter(1), sorted(class_weight.items(), key=itemgetter(0)))).reshape(-1, 1)).flatten()

        post_train_lm(model,
                      (X_train, ys_train),
                      (X_dev, ys_dev),
                      (X_test, ys_test),
                      [(vocab, label_vocab, rev_label_vocab), (vocab, vocab, rev_vocab)],
                      in_model_folder,
                      actual_config['epochs_number'],
                      actual_config,
                      sess,
                      class_weights=[class_weight_vector, np.ones(len(vocab))])


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.main_dataset_folder, args.lm_dataset_folder, args.model_folder, args.resume, config)
