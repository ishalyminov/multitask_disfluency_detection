from argparse import ArgumentParser

import pandas as pd
import tensorflow as tf

from dialogue_denoiser_lstm import (predict,
                                    load,
                                    make_dataset)


def configure_argument_parser():
    parser = ArgumentParser(description='Run the LSTM denoiser on a dataset')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('result_file')
    return parser


def main(in_dataset_file, in_model_folder, in_result_file):
    dataset = pd.read_json(in_dataset_file)

    with tf.Session() as sess:
       model, actual_config, vocab, char_vocab, label_vocab = load(in_model_folder, sess)
       rev_label_vocab = {label_id: label
                          for label, label_id in label_vocab.iteritems()}
       print 'Done loading'
       X, y = make_dataset(dataset, vocab, label_vocab)
       y_pred = predict(model, (X, y), rev_label_vocab, sess)
    tags_predicted = []
    tag_idx = 0
    for tag_seq in dataset['tags']:
        tags_predicted.append(y_pred[tag_idx: tag_idx + len(tag_seq)])
        tag_idx += len(tag_seq)
    result = pd.DataFrame({'utterance': dataset['utterance'],
                           'tags_gold': dataset['tags'],
                           'tags_predicted': tags_predicted})
    result.to_json(in_result_file)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset_folder, args.model_folder, args.result_file)

