from __future__ import print_function

import json
import random
import os
import sys
from copy import deepcopy

import sklearn as sk
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd

from data_utils import make_multitask_dataset
from pos_tag_dataset import pos_tag
from training_utils import get_loss_function, batch_generator

THIS_FILE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))

from deep_disfluency.utils.tools import (convert_from_eval_tags_to_inc_disfluency_tags,
                                         convert_from_inc_disfluency_tags_to_eval_tags)
from deep_disfluency.utils.tools import convert_from_inc_disfluency_tags_to_eval_tags
from deep_disfluency.evaluation.disf_evaluation import incremental_output_disfluency_eval_from_file
from deep_disfluency.evaluation.disf_evaluation import final_output_disfluency_eval_from_file
from deep_disfluency.evaluation.eval_utils import get_tag_data_from_corpus_file
from deep_disfluency.evaluation.eval_utils import rename_all_repairs_in_line_with_index
from deep_disfluency_utils import get_tag_mapping



MODEL_NAME = 'ckpt'

DATA_DIR = os.path.join(THIS_FILE_DIR,
                        'deep_disfluency',
                        'deep_disfluency',
                        'data',
                        'disfluency_detection',
                        'switchboard')
DEFAULT_HELDOUT_DATASET = DATA_DIR + '/swbd_disf_heldout_data_timings.csv'


def post_train_lm(in_model,
                  train_data,
                  dev_data,
                  test_data,
                  in_task_vocabs,
                  in_model_folder,
                  in_epochs_number,
                  config,
                  session,
                  class_weights=None,
                  task_weights=None,
                  **kwargs):
    X_train, y_train_for_tasks = train_data

    tag_mapping = get_tag_mapping(in_task_vocabs[0][1])

    X, ys_for_tasks, logits_for_tasks = in_model

    for v in tf.trainable_variables():
        v_initial = tf.Variable(v, name=v.name + '_initial', trainable=False)
        session.run(v_initial.initializer)

    if class_weights is None:
        class_weights = [np.ones(y_train_i.shape[1]) for y_train_i in y_train_for_tasks]
    if task_weights is None:
        task_weights = {'lm': 1.0, 'tag': 0.0}
    # Define loss and optimizer
    loss_op = get_loss_function(logits_for_tasks,
                                ys_for_tasks,
                                class_weights,
                                l2_coef=config['l2_coef'],
                                task_weights=task_weights,
                                weight_trajectory_coef=0.99)

    starting_lr = config['lr']
    lr_decay = config['lr_decay']
    global_step = tf.Variable(0, trainable=False)
    session.run(tf.assign(global_step, 0))
    learning_rate = lr = tf.train.cosine_decay(starting_lr,
                                               global_step,
                                               2000000,
                                               alpha=0.001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step)

    saver = tf.train.Saver(tf.global_variables())

    _, dev_eval = evaluate(in_model,
                           dev_data,
                           tag_mapping,
                           class_weights,
                           task_weights,
                           config,
                           session)
    best_dev_f1_rm = dev_eval['f1_rm']
    epochs_without_improvement = 0
    for epoch_counter in xrange(in_epochs_number):
        batch_gen = batch_generator(X_train,
                                    y_train_for_tasks,
                                    config['batch_size'])
        train_batch_losses = []
        for batch_x, batch_y in batch_gen:
            _, train_batch_loss = session.run([train_op, loss_op],
                                              feed_dict={X: batch_x, ys_for_tasks: batch_y})
            train_batch_losses.append(train_batch_loss)
        _, dev_eval = evaluate(in_model,
                               dev_data,
                               tag_mapping,
                               class_weights,
                               task_weights,
                               config,
                               session)
        print('Epoch {} out of {} results'.format(epoch_counter, in_epochs_number))
        print('train loss: {:.3f}'.format(np.mean(train_batch_losses)))
        print('; '.join(['dev {}: {:.3f}'.format(key, value)
                         for key, value in dev_eval.iteritems()]) + ' @lr={}'.format(session.run(learning_rate)))
        if best_dev_f1_rm < dev_eval['f1_rm']:
            best_dev_f1_rm = dev_eval['f1_rm']
            saver.save(session, os.path.join(in_model_folder, MODEL_NAME))
            print('New best loss. Saving checkpoint')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if config['early_stopping_threshold'] < epochs_without_improvement:
            print('Early stopping after {} epochs'.format(epoch_counter))
            break

    print('Optimization Finished!')


def create_model(in_vocab_size, in_cell_size, in_max_input_length, in_task_output_dimensions):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        X = tf.placeholder(tf.int32, [None, in_max_input_length], name='X')
        ys_for_tasks = [tf.placeholder(tf.float32, [None, task_i_output_dimensions], name='y_{}'.format(task_idx))
                        for task_idx, task_i_output_dimensions in enumerate(in_task_output_dimensions)]
        embeddings = tf.Variable(tf.random_uniform([in_vocab_size, in_cell_size], -1.0, 1.0),
                                 name='emb')
        emb = tf.nn.embedding_lookup(embeddings, X)

        lstm_cell = rnn.BasicLSTMCell(in_cell_size, forget_bias=1.0, name='lstm')
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, emb, dtype=tf.float32)

        W_for_tasks = [tf.Variable(tf.random_normal([in_cell_size, task_i_output_dim]),
                                   name='W_{}'.format(task_idx))
                       for task_idx, task_i_output_dim in enumerate(in_task_output_dimensions)]
        b_for_tasks = [tf.Variable(tf.random_normal([task_i_output_dim]),
                                   name='bias_{}'.format(task_idx))
                       for task_idx, task_i_output_dim in enumerate(in_task_output_dimensions)]

        task_outputs = [tf.add(tf.matmul(outputs[:, -1, :], W_task), b_task)
                        for W_task, b_task in zip(W_for_tasks, b_for_tasks)]
    return X, tuple(ys_for_tasks), task_outputs
