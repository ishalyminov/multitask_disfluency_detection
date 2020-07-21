from argparse import ArgumentParser
import os
from operator import itemgetter

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import model
from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary, make_multitask_dataset, reverse_dict
from training_utils import get_class_weight_proportional, train


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--resume', action='store_true', default=False)

    return parser


def init_model(trainset, in_model_folder, resume, in_config):
    model = None
    if not resume:
        utterances = trainset['utterance']
        postags = trainset['pos']
        vocab, _ = make_vocabulary(utterances, in_config['max_vocabulary_size'])
        pos_vocab, _ = make_vocabulary(postags, in_config['max_vocabulary_size'])
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

        rev_vocab = reverse_dict(vocab)
        rev_label_vocab = reverse_dict(label_vocab)
        rev_pos_vocab = reverse_dict(pos_vocab)
        model = getattr(model, in_config['model_class'])(vocab,
                                                         rev_vocab,
                                                         pos_vocab,
                                                         reverse_pos_vocab,
                                                         label_vocab,
                                                         rev_label_vocab,
                                                         in_config)

        model.save(model, in_config, vocab, pos_vocab, char_vocab, label_vocab, in_model_folder)
    model, actual_config, vocab, pos_vocab, char_vocab, label_vocab = model.load(in_model_folder,
                                                                                 existing_model=model)
    return (model,
            actual_config,
            {'word_vocab': vocab,
             'rev_word_vocab': rev_vocab,
             'pos_vocab': pos_vocab,
             'rev_pos_vocab': rev_pos_vocab,
             'char_vocab': char_vocab,
             'label_vocab': label_vocab,
             'rev_label_vocab': rev_label_vocab})


def main(in_dataset_folder, in_model_folder, resume, in_config):
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    model, actual_config, vocabs = init_model(trainset, in_model_folder, resume, in_config)
    X_train, ys_train = make_multitask_dataset(trainset, vocabs['word_vocab'], vocabs['label_vocab'], actual_config)
    X_dev, ys_dev = make_multitask_dataset(devset, vocabs['word_vocab'], vocabs['label_vocab'], actual_config)
    X_test, ys_test = make_multitask_dataset(testset, vocabs['word_vocab'], vocabs['label_vocab'], actual_config)

    class_weight = get_class_weight_proportional(ys_train[0],
                                                 smoothing_coef=actual_config['class_weight_smoothing_coef'])

    scaler = MinMaxScaler(feature_range=(1, 5))
    label_freqs = list(map(itemgetter(1), sorted(class_weight.items(), key=itemgetter(0))))
    class_weight_vector = scaler.fit_transform(np.array(label_freqs).reshape(-1, 1)).flatten()

    train(model,
          (X_train, ys_train),
          (X_dev, ys_dev),
          (X_test, ys_test),
          [(vocabs['word_vocab'], vocabs['label_vocab'], vocabs['rev_label_vocab']),
           (vocabs['word_vocab'], vocabs['label_vocab'], vocabs['rev_label_vocab'])],
          in_model_folder,
          actual_config['epochs_number'],
          actual_config,
          class_weights=[class_weight_vector, np.ones(len(vocabs['word_vocab']))],
          task_weights=config['task_weights'])


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.dataset_folder, args.model_folder, args.resume, config)
