from __future__ import print_function

from collections import defaultdict, deque
from math import sin, pi

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


def get_sample_weight(in_labels, in_class_weight_map):
    sample_weight = np.vectorize(in_class_weight_map.get)(in_labels)
    return sample_weight


def get_class_weight_sqrt(in_labels):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / np.power(freq, 1 / 3.) for label, freq in label_freqs.iteritems()}
    return label_weights


def get_class_weight_proportional(in_labels, smoothing_coef=1.0):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / np.power(float(freq), 1.0 / smoothing_coef)
                     for label, freq in label_freqs.iteritems()}
    return label_weights


def get_class_weight_auto(in_labels):
    class_weight = compute_class_weight('balanced', np.unique(in_labels), in_labels)
    class_weight_map = {class_id: weight
                        for class_id, weight in zip(np.unique(in_labels), class_weight)}

    return class_weight_map


def random_batch_generator(data, labels, batch_size, sample_probabilities=None):
    data_idx = range(labels.shape[0])
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, p=sample_probabilities)
        batch = (np.take(data, batch_idx, axis=0), np.take(labels, batch_idx, axis=0))
        yield batch


def dynamic_importance_sampling_random_batch_generator(data,
                                                       labels,
                                                       batch_size,
                                                       smoothing_coef_min,
                                                       smoothing_coef_max):
    labels_flat = np.argmax(labels, axis=-1)

    sample_probs = np.ones(labels.shape[0])
    data_idx = range(labels.shape[0])
    x = 0.0
    delta = smoothing_coef_min - smoothing_coef_max
    batch_counter = 0
    while True:
        if batch_counter == 0:
            class_weight = get_class_weight_proportional(labels_flat, smoothing_coef=smoothing_coef_min + delta * abs(sin(x)))
            sample_weight = get_sample_weight(labels_flat, class_weight)
            sample_probs = sample_weight / sum(sample_weight)
            x = (x + 0.1) % (2 * pi)
        batch_counter = (batch_counter + 1) % 1000
        batch_idx = np.random.choice(data_idx, size=batch_size, p=sample_probs)
        batch = (np.take(data, batch_idx, axis=0), np.take(labels, batch_idx, axis=0))
        yield batch


def batch_generator(X, y_for_tasks, batch_size):
    batch_start_idx = 0
    total_batches_number = X.shape[0] / batch_size
    batch_counter = 0
    while batch_start_idx < X.shape[0]:
        if batch_counter % 1000 == 0:
            print('Processed {} out of {} batches'.format(batch_counter, total_batches_number))
        batch = (X[batch_start_idx: batch_start_idx + batch_size],
                 [y_i[batch_start_idx: batch_start_idx + batch_size] for y_i in y_for_tasks])
        batch_start_idx += batch_size
        batch_counter += 1
        yield batch


def get_loss_function(in_logits_for_tasks,
                      in_labels_for_tasks,
                      in_class_weights_for_tasks,
                      l2_coef=0.0,
                      weight_change_penalization_coef=0.0,
                      task_weights=None):
    assert len(in_logits_for_tasks) == len(in_labels_for_tasks) == len(task_weights)
    if task_weights == None:
        task_weights = np.ones(len(in_logits_for_tasks))
    eps = tf.constant(value=np.finfo(np.float32).eps, dtype=tf.float32)

    losses = []
    for logits, labels, class_weights in zip(in_logits_for_tasks,
                                             in_labels_for_tasks,
                                             in_class_weights_for_tasks):
        class_weights_i = tf.constant(value=class_weights, dtype=tf.float32)
        logits_i = logits + eps
        softmax_i = tf.nn.softmax(logits_i)
        loss_xent_i = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax_i + eps), class_weights_i),
                                     reduction_indices=[1])
        losses.append(loss_xent_i)
    # loss_xent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=in_labels, logits=in_logits)
    # Add regularization loss as well
    loss_l2 = tf.reduce_sum([tf.nn.l2_loss(v)
                             for v in tf.trainable_variables()
                             if 'bias' not in v.name]) * l2_coef

    graph = tf.get_default_graph()
    # L2 regularization on model weights trajectory
    weight_change_l2s = []
    all_tensor_names = [v.name for v in tf.trainable_variables()]
    for v in tf.trainable_variables():
        v_initial_name = v.name.partition(':')[0] + '_initial:' + v.name.partition(':')[-1]
        if v_initial_name not in all_tensor_names:
            continue
        v_initial = graph.get_tensor_by_name(v_initial_name)
        weight_change_l2s.append(tf.nn.l2_loss(tf.subtract(v, v_initial)))
    loss_weight_change = None if not len(weight_change_l2s) \
                         else tf.reduce_sum(weight_change_l2s) * weight_change_penalization_coef

    losses_weighted = [loss_i * task_weight_i
                       for loss_i, task_weight_i in zip(losses, task_weights)]
    aux_losses = [l for l in [loss_l2, loss_weight_change] if l is not None]
    cost = tf.reduce_mean(tf.add(tf.reduce_sum(losses_weighted), *aux_losses), name='cost')

    return cost
