from argparse import ArgumentParser

import tensorflow as tf

from dialogue_denoiser_lstm import load, filter_line


def configure_argument_parser():
    parser = ArgumentParser(description='Interactive LSTM dialogue filter')
    parser.add_argument('model_folder')

    return parser


def run(in_model_folder):
    with tf.Session() as sess:
        model, actual_config, vocab, char_vocab, label_vocab = load(in_model_folder,
                                                                    sess)
        rev_vocab = {word_id: word
                     for word, word_id in vocab.iteritems()}
        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}
        print 'Done loading'
        try:
            line = raw_input().strip()
            while line:
                print filter_line(line,
                                  model,
                                  [(vocab, label_vocab, rev_label_vocab),
                                   (vocab, vocab, rev_vocab)],
                                  actual_config,
                                  sess)
                line = raw_input().strip()
        except EOFError as e:
            pass


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    run(args.model_folder)

