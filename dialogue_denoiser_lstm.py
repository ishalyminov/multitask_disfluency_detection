import json
import random
import os
from collections import deque

import keras
from keras.layers import (TimeDistributed,
                          Embedding,
                          Conv1D,
                          Conv2D,
                          MaxPool1D,
                          MaxPool2D,
                          Flatten,
                          LSTM,
                          Reshape)
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from data_utils import vectorize_sequences, pad_sequences, PAD_ID
from deep_disfluency_utils import make_tag_mapping
from metrics import DisfluencyDetectionF1Score

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


def get_class_weight(in_labels):
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


def char_cnn_module(in_char_input, in_vocab_size, in_emb_size):
    """
        Zhang and LeCun, 2015
    """
    model = TimeDistributed(Embedding(in_vocab_size, in_emb_size, mask_zero=True))(in_char_input)
    model = TimeDistributed(Reshape((27, in_emb_size, 1)))(model)
    model = TimeDistributed(Conv2D(32, (3, 3), activation='relu', name='chars'))(model)
    model = TimeDistributed(MaxPool2D((3, 3)))(model)

    model = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(model)
    model = TimeDistributed(MaxPool2D((1, 1)))(model)

    model = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(model)
    model = TimeDistributed(MaxPool2D((1, 1)))(model)

    model = TimeDistributed(Flatten())(model)

    return model


def char_cnn_1d_module(in_char_input, in_vocab_size, in_emb_size):
    """
        Zhang and LeCun, 2015
    """
    model = TimeDistributed(Embedding(in_vocab_size, in_emb_size, mask_zero=True))(in_char_input)
    # model = TimeDistributed(Reshape((27, in_emb_size, 1)))(model)
    model = TimeDistributed(Conv1D(32, 3, activation='relu', name='chars'))(model)
    model = TimeDistributed(MaxPool1D(3))(model)

    model = TimeDistributed(Conv1D(32, 3, activation='relu'))(model)
    model = TimeDistributed(MaxPool1D(1))(model)

    model = TimeDistributed(Conv1D(32, 3, activation='relu'))(model)
    model = TimeDistributed(MaxPool1D(1))(model)

    model = TimeDistributed(Flatten())(model)

    return model


def rnn_module(in_word_input, in_vocab_size, in_cell_size):
    embedding = Embedding(in_vocab_size, in_cell_size, mask_zero=True)(in_word_input)
    lstm = LSTM(in_cell_size, return_sequences=False)(embedding)
    return lstm


def create_model(in_vocab_size,
                 in_char_vocab_size,
                 in_cell_size,
                 in_char_cell_size,
                 in_max_input_length,
                 in_max_char_input_length,
                 in_classes_number,
                 lr):
    word_input = keras.layers.Input(shape=(in_max_input_length,))
    # char_input = keras.layers.Input(shape=(in_max_input_length, in_max_char_input_length))
    rnn = rnn_module(word_input, in_vocab_size, in_cell_size)
    # char_cnn = char_cnn_module(char_input, in_char_vocab_size, in_char_cell_size)

    rnn_cnn_combined = rnn # keras.layers.Concatenate()([rnn, char_cnn])
    output = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(rnn_cnn_combined)
    # output = keras.layers.Dropout(0.5)(output)
    # output = keras.layers.Dense(128, activation='relu')(output)
    # output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.TimeDistributed(keras.layers.Dense(in_classes_number,
                                                             activation='softmax',
                                                             name='labels'))(output)
    model = keras.Model(inputs=[word_input], outputs=[output])

    opt = keras.optimizers.Adam(lr=lr, clipnorm=5.0)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')
    return model


def batch_generator(data, labels, batch_size, sample_probabilities=None):
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
        batch = ([np.take(feature, batch_idx, axis=0) for feature in data],
                 np.take(labels, batch_idx, axis=0))
        yield batch


def train(in_model,
          train_data,
          dev_data,
          test_data,
          in_checkpoint_filepath,
          label_vocab,
          class_weight,
          epochs=100,
          batch_size=32,
          steps_per_epoch=1000,
          **kwargs):
    X_train, y_train = train_data
    X_dev, y_dev = dev_data
    X_test, y_test = test_data

    x, y, logits = in_model
    prediction = tf.nn.softmax(logits)

    y = tf.placeholder("float", [None, 3])
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        batch_gen = batch_generator(X_train, y_train, batch_size)
        # Run the initializer
        sess.run(init)

        step = 0
        for batch_x, batch_y in batch_gen:
            step += 1
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={x: batch_x[0], y: batch_y})
            if step % 100 == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x: X_train[0],
                                                                 y: y_train})
                print "Step " + str(step) + ", Training set Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc)

        print "Optimization Finished!"


def predict(in_model, X):
    model_out = in_model.predict(X)
    return np.argmax(model_out, axis=-1)


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


def evaluate(in_model, X, y):
    y_pred = np.argmax(in_model.predict(X), axis=-1)
    y_gold = np.argmax(y, axis=-1)
    return sum([int(np.array_equal(y_pred_i, y_gold_i))
                for y_pred_i, y_gold_i in zip(y_pred, y_gold)]) / float(y.shape[0])

import tensorflow as tf
from tensorflow.contrib import rnn

def create_simple_model(in_vocab_size,
                        in_char_vocab_size,
                        in_cell_size,
                        in_char_cell_size,
                        in_max_input_length,
                        in_max_char_input_length,
                        in_classes_number,
                        lr):
    x = tf.placeholder(tf.int32, [None, in_max_input_length])
    y = tf.placeholder(tf.int32, [None, in_classes_number])
    embeddings = tf.Variable(tf.random_uniform([in_vocab_size, in_cell_size], -1.0, 1.0))
    emb = tf.nn.embedding_lookup(embeddings, x)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(in_cell_size, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, emb, dtype=tf.float32)

    weights = tf.Variable(tf.random_normal([in_cell_size, in_classes_number]))
    biases = tf.Variable(tf.random_normal([in_classes_number]))
    # Linear activation, using rnn inner loop last output
    return (x, y, tf.add(tf.matmul(outputs[:,-1,:], weights), biases))

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
