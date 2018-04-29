import json
import random
import os
from collections import deque, defaultdict

import keras
import sklearn as sk
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_utils import vectorize_sequences, pad_sequences
from deep_disfluency_utils import get_tag_mapping

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

MAX_VOCABULARY_SIZE = 15000
# we have dependencies up to 8 tokens back, so this should do
MAX_INPUT_LENGTH = 20 
MEAN_WORD_LENGTH = 8
CNN_CONTEXT_LENGTH = 3
MAX_CHAR_INPUT_LENGTH = CNN_CONTEXT_LENGTH * (MEAN_WORD_LENGTH + 1)

MODEL_NAME = 'model.h5'
VOCABULARY_NAME = 'vocab.json'
CHAR_VOCABULARY_NAME = 'char_vocab.json'
LABEL_VOCABULARY_NAME = 'label_vocab.json'


def get_sample_weight(in_labels, in_class_weight_map):
    sample_weight = np.vectorize(in_class_weight_map.get)(in_labels)
    return sample_weight


def get_class_weight_sqrt(in_labels):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / np.sqrt(freq) for label, freq in label_freqs.iteritems()}
    return label_weights


def get_class_weight_proportional(in_labels):
    label_freqs = defaultdict(lambda: 0)
    for label in in_labels:
        label_freqs[label] += 1.0
    label_weights = {label: 1.0 / float(freq) for label, freq in label_freqs.iteritems()}
    return label_weights


def get_class_weight_auto(in_labels):
    class_weight = compute_class_weight('balanced', np.unique(in_labels), in_labels)
    class_weight_map = {class_id: weight
                        for class_id, weight in zip(np.unique(in_labels), class_weight)}

    return class_weight_map


def make_data_points(in_tokens, in_tags):
    contexts, tags = [], []
    context = deque([], maxlen=MAX_INPUT_LENGTH)
    for token, tag in zip(in_tokens, in_tags):
        context.append(token)
        contexts.append(list(context))
        tags.append(tag)
    return contexts, tags


def make_dataset(in_dataset, in_vocab, in_char_vocab, in_label_vocab):
    contexts, tags = [], []
    for idx, row in in_dataset.iterrows():
        current_contexts, current_tags = make_data_points(row['utterance'], row['tags'])
        for context, tag in zip(current_contexts, current_tags):
            if tag in in_label_vocab:
                contexts.append(context)
                tags.append(tag)
        #contexts += current_contexts
        #tags += current_tags
    tokens_vectorized = vectorize_sequences(contexts, in_vocab)
    tokens_padded = pad_sequences(tokens_vectorized, MAX_INPUT_LENGTH)

    chars_vectorized = []
    for utterance_tokenized in contexts:
        char_contexts = [' '.join(utterance_tokenized[max(i - CNN_CONTEXT_LENGTH + 1, 0): i + 1])
                         for i in xrange(len(utterance_tokenized))]
        char_contexts_vectorized = vectorize_sequences(char_contexts, in_char_vocab)
        char_contexts_padded = pad_sequences(char_contexts_vectorized, MAX_CHAR_INPUT_LENGTH)
        chars_vectorized += [char_contexts_padded]

    chars_padded = pad_sequences(chars_vectorized, MAX_INPUT_LENGTH)
    labels = vectorize_sequences([tags], in_label_vocab)

    y = keras.utils.to_categorical(labels[0], num_classes=len(in_label_vocab))
 
    return [tokens_padded, chars_padded], y


def random_batch_generator(data, labels, batch_size, sample_probabilities=None):
    """Generator used by `keras.models.Sequential.fit_generator` to yield batches
    of pairs.
    Such a generator is required by the parallel nature of the aforementioned
    Keras function. It can theoretically feed batches of pairs indefinitely
    (looping over the dataset). Ideally, it would be called so that an epoch ends
    exactly with the last batch of the dataset.
    """

    data_idx = range(labels.shape[0])
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, p=sample_probabilities)
        batch = (np.take(data, batch_idx, axis=0), np.take(labels, batch_idx, axis=0))
        yield batch


def batch_generator(X, y, batch_size):
    batch_start_idx = 0
    while batch_start_idx < y.shape[0]:
        batch = (X[batch_start_idx: batch_start_idx + batch_size],
                 y[batch_start_idx: batch_start_idx + batch_size])
        batch_start_idx += batch_size
        yield batch


def train(in_model,
          train_data,
          dev_data,
          test_data,
          in_checkpoint_folder,
          label_vocab,
          class_weight,
          learning_rate=0.01,
          epochs=100,
          batch_size=32,
          steps_per_epoch=1000,
          **kwargs):
    X_train, y_train = train_data
    y_train_flattened = np.argmax(y_train, -1)
    sample_weights = get_sample_weight(y_train_flattened,
                                       get_class_weight_proportional(y_train_flattened))
    tag_mapping = get_tag_mapping(label_vocab)

    X, y, logits = in_model

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    y_pred_op = tf.argmax(logits, 1)
    y_true_op = tf.argmax(y, 1)
    correct_pred = tf.equal(y_pred_op, y_true_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    # Start training
    with tf.Session() as sess:
        sample_probs = sample_weights / np.sum(sample_weights)
        batch_gen = random_batch_generator(X_train,
                                           y_train,
                                           batch_size,
                                           sample_probabilities=sample_probs)
        # Run the initializer
        sess.run(init)

        step, best_dev_loss = 0, np.inf
        for batch_x, batch_y in batch_gen:
            step += 1
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, y: batch_y})
            if step % steps_per_epoch == 0:
                print 'Step {} eval'.format(step) 

                dev_eval = evaluate(in_model, dev_data, tag_mapping, sess)
                print '; '.join(['dev {}: {:.3f}'.format(key, value)
                                 for key, value in dev_eval.iteritems()])
                if dev_eval['loss'] < best_dev_loss:
                    best_dev_loss = dev_eval['loss']
                    saver.save(sess, in_checkpoint_folder)
    print "Optimization Finished!"


def evaluate(in_model, in_dataset, in_tag_map, in_session, batch_size=32):
    X_test, y_test = in_dataset
    X, y, logits = in_model

    # Evaluate model (with test logits, for dropout to be disabled)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    y_pred_op = tf.argmax(logits, 1)
    y_true_op = tf.argmax(y, 1)
    correct_pred = tf.equal(y_pred_op, y_true_op)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Start training
    batch_gen = batch_generator(X_test, y_test, batch_size)

    batch_losses, batch_accuracies = [], []
    y_pred = np.zeros(y_test.shape[0])
    for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
        y_pred_batch, loss_batch, acc_batch = in_session.run([y_pred_op, loss_op, accuracy],
                                                              feed_dict={X: batch_x, y: batch_y})
        y_pred[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred_batch
        batch_losses.append(loss_batch)
        batch_accuracies.append(acc_batch)

    result_matrix = {'loss': np.mean(batch_losses), 'acc': np.mean(batch_accuracies)}
    for class_name, class_ids in in_tag_map.iteritems():
        result_matrix['f1_' + class_name] = sk.metrics.f1_score(y_true=np.argmax(y_test, -1),
                                                                y_pred=y_pred,
                                                                labels=class_ids,
                                                                average='micro')
    return result_matrix


def denoise_line(in_line, in_model, in_vocab, in_char_vocab, in_rev_label_vocab):
    tokens = [in_line.lower().split()]
    tokens_vectorized = vectorize_sequences(tokens, in_vocab)
    chars_vectorized = []
    for utterance_tokenized in tokens:
        contexts = [' '.join(utterance_tokenized[max(i - CONTEXT_LENGTH + 1, 0): i + 1])
                    for i in xrange(len(utterance_tokenized))]
        contexts_vectorized = vectorize_sequences(contexts, in_char_vocab)
        chars_vectorized += [contexts_vectorized]
    chars_vectorized = pad_sequences(chars_vectorized, MAX_INPUT_LENGTH)

    predicted = predict(in_model, tokens_vectorized)[0]
    result_tokens = map(lambda x: in_rev_label_vocab[x], predicted[:len(tokens[0])])
    return ' '.join(result_tokens)


def create_model(in_vocab_size, in_cell_size, in_max_input_length, in_classes_number):
    X = tf.placeholder(tf.int32, [None, in_max_input_length])
    y = tf.placeholder(tf.int32, [None, in_classes_number])
    embeddings = tf.Variable(tf.random_uniform([in_vocab_size, in_cell_size], -1.0, 1.0))
    emb = tf.nn.embedding_lookup(embeddings, X)

    lstm_cell = rnn.BasicLSTMCell(in_cell_size, forget_bias=1.0)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, emb, dtype=tf.float32)

    W = tf.Variable(tf.random_normal([in_cell_size, in_classes_number]))
    b = tf.Variable(tf.random_normal([in_classes_number]))

    return X, y, tf.add(tf.matmul(outputs[:,-1,:], W), b)


def load(in_model_folder):
    with open(os.path.join(in_model_folder, VOCABULARY_NAME)) as vocab_in:
        vocab = json.load(vocab_in)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME)) as char_vocab_in:
        char_vocab = json.load(char_vocab_in)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME)) as label_vocab_in:
        label_vocab = json.load(label_vocab_in)
    model = keras.models.load_model(os.path.join(in_model_folder, MODEL_NAME))
    return model, vocab, char_vocab, label_vocab


def save(in_model, in_vocab, in_char_vocab, in_label_vocab, in_model_folder, save_model=False):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    if save_model:
        in_model.save(os.path.join(in_model_folder, MODEL_NAME))
    with open(os.path.join(in_model_folder, VOCABULARY_NAME), 'w') as vocab_out:
        json.dump(in_vocab, vocab_out)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME), 'w') as char_vocab_out:
        json.dump(in_char_vocab, char_vocab_out)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME), 'w') as label_vocab_out:
        json.dump(in_label_vocab, label_vocab_out)
