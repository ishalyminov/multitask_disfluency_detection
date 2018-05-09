from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn as sk
from catboost import CatBoostClassifier

from config import read_config, DEFAULT_CONFIG_FILE
from data_utils import make_vocabulary, make_char_vocabulary
from dialogue_denoiser_lstm import (create_model,
                                    train,
                                    save,
                                    make_dataset,
                                    load_encoder,
                                    batch_generator,
                                    get_class_weight_proportional,
                                    get_sample_weight)
from deep_disfluency_utils import get_tag_mapping


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_folder')
    parser.add_argument('model_folder')
    parser.add_argument('--config', default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--resume', action='store_true', default=False)

    return parser


def main(in_dataset_folder, in_model_folder, resume, in_config):
    trainset, devset, testset = (pd.read_json(os.path.join(in_dataset_folder, 'trainset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'devset.json')),
                                 pd.read_json(os.path.join(in_dataset_folder, 'testset.json')))
    with tf.Session() as sess:
        model, actual_config, vocab, char_vocab, label_vocab = load_encoder(in_model_folder, sess, existing_model=None)
        X, y, encoding_op = model
        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}
        print 'Creating datasets'
        X_train, y_train = make_dataset(trainset, vocab, label_vocab, actual_config)
        X_dev, y_dev = make_dataset(devset, vocab, label_vocab, actual_config)
        X_test, y_test = make_dataset(testset, vocab, label_vocab, actual_config)

        batch_size = actual_config['batch_size']
        batch_gen = batch_generator(X_train, y_train, batch_size)

        print 'Creating embbedings for GBDT'
        X_emb = np.zeros(shape=(y_train.shape[0], actual_config['embedding_size']))
        for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
            X_emb_batch = sess.run([encoding_op],
                                   feed_dict={X: batch_x})
            X_emb[batch_idx * batch_size: (batch_idx + 1) * batch_size] = X_emb_batch[0][:,-1]

        y_train_flat = np.argmax(y_train, axis=-1)
        class_weight = get_class_weight_proportional(y_train_flat, smoothing_coef=1.05)
        sample_weight = get_sample_weight(y_train_flat, class_weight)
        print 'GBDT training started'
        clf = CatBoostClassifier(learning_rate=0.1, depth=6, task_type='GPU')
        trained_model = clf.fit(X_emb, y_train_flat)

        trained_model.save_model('dialogue_denoiser_lstm_gbdt.npy')
        print 'Creating test embbedings for GBDT'
        batch_gen = batch_generator(X_test, y_test, batch_size)
        X_test_emb = np.zeros(shape=(y_test.shape[0], actual_config['embedding_size']))
        for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
            X_emb_batch = sess.run([encoding_op],
                                   feed_dict={X: batch_x})
            X_test_emb[batch_idx * batch_size: (batch_idx + 1) * batch_size] = X_emb_batch[0][:,-1]
        y_pred = trained_model.predict(X_emb)

        tag_map = get_tag_mapping(label_vocab)
        result_map = {}

        for class_name, class_ids in tag_map.iteritems():
            result_map['f1_' + class_name] = sk.metrics.f1_score(y_true=np.argmax(y_train, axis=-1),
                                                                 y_pred=y_pred,
                                                                 labels=class_ids,
                                                                 average='micro')
        print result_map


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    config = read_config(args.config)
    main(args.dataset_folder, args.model_folder, args.resume, config)
