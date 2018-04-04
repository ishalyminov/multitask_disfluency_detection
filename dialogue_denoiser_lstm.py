import json
from random import shuffle
import random
from operator import itemgetter
import os

import keras
import numpy as np

from data_utils import make_vocabulary, vectorize_sequences, to_one_hot

random.seed(273)
np.random.seed(273)


TRAINSET_RATIO = 0.8
VOCABULARY_SIZE = 15000
MAX_INPUT_LENGTH = 80
MEAN_WORD_LENGTH = 8
MAX_CHAR_INPUT_LENGTH = MAX_INPUT_LENGTH * (MEAN_WORD_LENGTH + 1)

MODEL_NAME = 'model.h5'
VOCABULARY_NAME = 'vocab.json'
LABEL_VOCABULARY_NAME = 'label_vocab.json'


def load_dataset(in_encoder_input, in_decoder_input):
    with open(in_encoder_input) as encoder_in:
        with open(in_decoder_input) as decoder_in:
            encoder_lines, decoder_lines = [map(lambda x: x.strip(), encoder_in.readlines()),
                                            map(lambda x: x.strip(), decoder_in.readlines())]
    return encoder_lines, decoder_lines


def make_tagger_data_point(in_src, in_tgt):
    source, target = in_src.lower().split(), in_tgt.lower().split()
    tags = []
    src_index, tgt_index = 0, 0
    while src_index < len(source):
        if tgt_index < len(target) and source[src_index] == target[tgt_index]:
            tags.append(1)
            tgt_index += 1
        else:
            tags.append(0)
        src_index += 1
    assert len(tags) == len(source)
    return source, tags


def make_tagger_data_points(in_encoder_lines, in_decoder_lines):
    result = []
    for src_line, tgt_line in zip(in_encoder_lines, in_decoder_lines):
        result.append(make_tagger_data_point(src_line, tgt_line))
    return result


def make_dataset(in_data_points, in_vocab, in_label_vocab):
    X = vectorize_sequences(map(itemgetter(0), in_data_points), in_vocab, MAX_INPUT_LENGTH)
    labels = vectorize_sequences(map(itemgetter(1), in_data_points), in_label_vocab, MAX_INPUT_LENGTH)
    y = np.asarray([to_one_hot(label, len(in_label_vocab)) for label in labels]) 
    return X, y


def make_training_data(in_encoder_lines, in_decoder_lines):
    # data_points = make_tagger_data_points(in_encoder_lines, in_decoder_lines)
    data_points = [(enc_line, dec_line)
                   for enc_line, dec_line in zip(in_encoder_lines, in_decoder_lines)]
    train, dev, test = make_dataset_split(data_points, TRAINSET_RATIO)
    vocab, _ = make_vocabulary(map(itemgetter(0), train), VOCABULARY_SIZE)
    label_vocab, _ = make_vocabulary(map(itemgetter(1), train), VOCABULARY_SIZE)
    X_train, y_train = make_dataset(train, vocab, label_vocab)
    X_dev, y_dev = make_dataset(dev, vocab, label_vocab)
    X_test, y_test = make_dataset(test, vocab, label_vocab)
    return vocab, label_vocab, (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


def make_dataset_split(in_data_points, trainset_ratio):
    shuffle(in_data_points)
    trainset_size = int(TRAINSET_RATIO * len(in_data_points))
    devset_size = int((len(in_data_points) - trainset_size) / 2.0)
    train, dev, test = (in_data_points[:trainset_size],
                        in_data_points[trainset_size: trainset_size + devset_size],
                        in_data_points[trainset_size + devset_size:])
    return train, dev, test


def char_cnn_module(in_char_input):
    """
        Zhang and LeCun, 2015
    """

    # time_dist_input = keras.layers.TimeDistributed(keras.layers.Input(shape=in_char_input.shape[1:]))(in_char_input)
    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 7, activation='relu', name='chars'))(in_char_input)
    model = keras.layers.TimeDistributed(keras.layers.MaxPool1D(3))(model)

    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 7, activation='relu'))(model)
    model = keras.layers.TimeDistributed(keras.layers.MaxPool1D(3))(model)

    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 3, activation='relu'))(model)
    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 3, activation='relu'))(model)
    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 3, activation='relu'))(model)
    model = keras.layers.TimeDistributed(keras.layers.Conv1D(256, 3, activation='relu'))(model)
    model = keras.layers.TimeDistributed(keras.layers.MaxPool1D(3))(model)

    model = keras.layers.TimeDistributed(keras.layers.Flatten())(model)

    return model


def rnn_module(in_word_input, in_vocab_size, in_cell_size):
    embedding = keras.layers.Embedding(in_vocab_size, in_cell_size)(in_word_input)
    lstm = keras.layers.LSTM(in_cell_size, return_sequences=True)(embedding)
    return lstm


def create_model(in_vocab_size,
                 in_char_vocab_size,
                 in_cell_size,
                 in_max_input_length,
                 in_max_char_input_length,
                 in_classes_number,
                 lr):
    word_input = keras.layers.Input(shape=(in_max_input_length,))
    char_input = keras.layers.Input(shape=(in_max_input_length, in_max_char_input_length, in_char_vocab_size,))
    rnn = rnn_module(word_input, in_vocab_size, in_cell_size)
    char_cnn = char_cnn_module(char_input)

    rnn_cnn_combined = keras.layers.Concatenate()([rnn, char_cnn])
    output = keras.layers.Dense(1024, activation='relu')(rnn_cnn_combined)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.Dense(1024, activation='relu')(output)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.TimeDistributed(keras.layers.Dense(in_classes_number,
                                                             activation='softmax',
                                                             name='labels'))(output)
    model = keras.Model(inputs=[word_input, char_input], outputs=[output])

    # mean absolute error, accuracy
    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    return model


def train(in_model,
          train_data,
          dev_data,
          test_data,
          in_checkpoint_filepath,
          epochs=100,
          batch_size=32,
          **kwargs):
    X_train, y_train = train_data
    X_dev, y_dev = dev_data
    X_test, y_test = test_data

    in_model.fit(X_train,
                 y_train,
                epochs=epochs,
                shuffle=True,
                validation_data=(X_dev, y_dev),
                callbacks=[keras.callbacks.ModelCheckpoint(in_checkpoint_filepath,
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto',
                                                           period=1),
                           keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0,
                                                         patience=10,
                                                         verbose=1,
                                                        mode='auto')])
    test_loss = in_model.evaluate(x=X_test, y=y_test)
    print 'Testset loss after {} epochs: {:.3f}'.format(epochs, test_loss)


def predict(in_model, X):
    return np.argmax(in_model.predict(np.asarray([X])), axis=-1)


def denoise_line(in_line, in_model, in_vocab):
    tokens = in_line.lower().split()
    line_vectorized = vectorize_sequences([tokens], in_vocab, MAX_INPUT_LENGTH)
    predicted = predict(in_model, line_vectorized)[0]
    result_tokens = [example_token
                     for example_token, keep_flag in zip(tokens, predicted)
                     if keep_flag]
    return ' '.join(result_tokens)


def evaluate(in_model, X, y):
    y_pred = np.argmax(in_model.predict(X), axis=-1)
    y_gold = np.argmax(y, axis=-1)
    return sum([int(np.array_equal(y_pred_i, y_gold_i))
                for y_pred_i, y_gold_i in zip(y_pred, y_gold)]) / float(y.shape[0])


def load(in_model_folder):
    model = keras.models.load_model(os.path.join(in_model_folder, MODEL_NAME))
    with open(os.path.join(in_model_folder, VOCABULARY_NAME)) as vocab_in:
        vocab = json.load(vocab_in)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME)) as label_vocab_in:
        label_vocab = json.load(label_vocab_in)
    return model, vocab, label_vocab


def save(in_model, in_vocab, in_label_vocab, in_model_folder, save_model=False):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    if save_model:
        in_model.save(os.path.join(in_model_folder, MODEL_NAME))
    with open(os.path.join(in_model_folder, VOCABULARY_NAME), 'w') as vocab_out:
        json.dump(in_vocab, vocab_out)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME), 'w') as label_vocab_out:
        json.dump(in_label_vocab, label_vocab_out)

