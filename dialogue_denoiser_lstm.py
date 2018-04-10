import json
from random import shuffle
import random
from operator import itemgetter
import os

import keras
import numpy as np
from keras.layers import TimeDistributed, Embedding, Conv1D, MaxPool1D, Flatten, LSTM
from keras import backend as K

from data_utils import make_vocabulary, vectorize_sequences, to_one_hot, PAD_ID

random.seed(273)
np.random.seed(273)


TRAINSET_RATIO = 0.8
VOCABULARY_SIZE = 15000
MAX_INPUT_LENGTH = 80
MEAN_WORD_LENGTH = 8
MAX_CHAR_INPUT_LENGTH = MAX_INPUT_LENGTH * (MEAN_WORD_LENGTH + 1)

MODEL_NAME = 'model.h5'
VOCABULARY_NAME = 'vocab.json'
CHAR_VOCABULARY_NAME = 'char_vocab.json'
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


def make_dataset(in_data_points, in_vocab, in_char_vocab, in_label_vocab):
    utterances_tokenized, tags = (map(itemgetter(0), in_data_points),
                                  map(itemgetter(1), in_data_points))
    tokens_vectorized = vectorize_sequences(utterances_tokenized, in_vocab, MAX_INPUT_LENGTH)
    chars_vectorized = []
    for utterance_tokenized in utterances_tokenized:
        substrings = [' '.join(utterance_tokenized[:i + 1])
                      for i in xrange(len(utterance_tokenized))]
        substrings_vectorized = vectorize_sequences(substrings, in_char_vocab, MAX_CHAR_INPUT_LENGTH)
        chars_vectorized += [substrings_vectorized]
    chars_vectorized = keras.preprocessing.sequence.pad_sequences(chars_vectorized,
                                                                  value=PAD_ID,
                                                                  maxlen=MAX_INPUT_LENGTH)
    labels = vectorize_sequences(map(itemgetter(1), in_data_points), in_label_vocab, MAX_INPUT_LENGTH)
    y = np.asarray([to_one_hot(label, len(in_label_vocab)) for label in labels])
    return [tokens_vectorized, chars_vectorized], y


def make_training_data(in_encoder_lines, in_decoder_lines):
    data_points = [(enc_line, dec_line)
                   for enc_line, dec_line in zip(in_encoder_lines, in_decoder_lines)]
    train, dev, test = make_dataset_split(data_points, TRAINSET_RATIO)
    vocab, _ = make_vocabulary(map(itemgetter(0), train), VOCABULARY_SIZE)
    label_vocab, _ = make_vocabulary(map(itemgetter(1), train), VOCABULARY_SIZE)
    X_train, y_train = make_dataset(train, vocab, label_vocab)
    X_dev, y_dev = make_dataset(dev, vocab, label_vocab)
    X_test, y_test = make_dataset(test, vocab, label_vocab)
    return vocab, label_vocab, (X_train, y_train), (X_dev, y_dev), (X_test, y_test)


def make_dataset_split(in_data_points, trainset_ratio=TRAINSET_RATIO):
    shuffle(in_data_points)
    trainset_size = int(trainset_ratio * len(in_data_points))
    devset_size = int((len(in_data_points) - trainset_size) / 2.0)
    train, dev, test = (in_data_points[:trainset_size],
                        in_data_points[trainset_size: trainset_size + devset_size],
                        in_data_points[trainset_size + devset_size:])
    return train, dev, test


def char_cnn_module(in_char_input, in_vocab_size, in_emb_size):
    """
        Zhang and LeCun, 2015
    """

    model = TimeDistributed(Embedding(in_vocab_size, in_emb_size))(in_char_input)
    model = TimeDistributed(Conv1D(16, 5, activation='relu', name='chars'))(model)
    model = TimeDistributed(MaxPool1D(3))(model)

    model = TimeDistributed(Conv1D(4, 3, activation='relu'))(model)
    model = TimeDistributed(MaxPool1D(3))(model)

    model = TimeDistributed(Flatten())(model)

    return model


def rnn_module(in_word_input, in_vocab_size, in_cell_size):
    embedding = Embedding(in_vocab_size, in_cell_size)(in_word_input)
    lstm = LSTM(in_cell_size, return_sequences=True)(embedding)
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
    char_input = keras.layers.Input(shape=(in_max_input_length, in_max_char_input_length))
    rnn = rnn_module(word_input, in_vocab_size, in_cell_size)
    char_cnn = char_cnn_module(char_input, in_char_vocab_size, in_char_cell_size)

    rnn_cnn_combined = char_cnn  # keras.layers.Concatenate()([rnn, char_cnn])
    output = keras.layers.Dense(128, activation='relu')(rnn_cnn_combined)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.TimeDistributed(keras.layers.Dense(in_classes_number,
                                                             activation='softmax',
                                                             name='labels'))(output)
    model = keras.Model(inputs=[word_input, char_input], outputs=[output])

    opt = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[f1])
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

    in_model.fit(x=X_train,
                 y=y_train,
                 epochs=epochs,
                 shuffle=True,
                 batch_size=batch_size,
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
                                                          mode='auto'),
                            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.2,
                                                              patience=5,
                                                              min_lr=0.001)])
    test_loss = in_model.evaluate(x=X_test, y=y_test)
    print 'Evaluation results on testset:', test_loss


def predict(in_model, X):
    return np.argmax(in_model.predict(X), axis=-1)


def denoise_line(in_line, in_model, in_vocab, in_char_vocab, in_label_vocab):
    tokens = [in_line.lower().split()]
    tokens_vectorized = vectorize_sequences(tokens, in_vocab, MAX_INPUT_LENGTH)
    chars_vectorized = []
    for utterance_tokenized in tokens:
        substrings = [' '.join(utterance_tokenized[:i + 1])
                      for i in xrange(len(utterance_tokenized))]
        substrings_vectorized = vectorize_sequences(substrings, in_char_vocab, MAX_CHAR_INPUT_LENGTH)
        chars_vectorized += [substrings_vectorized]
    chars_vectorized = keras.preprocessing.sequence.pad_sequences(chars_vectorized,
                                                                  value=PAD_ID,
                                                                  maxlen=MAX_INPUT_LENGTH)

    predicted = predict(in_model, chars_vectorized)[0]
    result_tokens = [example_token
                     for example_token, keep_flag in zip(tokens, predicted)
                     if keep_flag]
    import pdb; pdb.set_trace()
    return ' '.join(result_tokens)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def evaluate(in_model, X, y):
    y_pred = np.argmax(in_model.predict(X), axis=-1)
    y_gold = np.argmax(y, axis=-1)
    return sum([int(np.array_equal(y_pred_i, y_gold_i))
                for y_pred_i, y_gold_i in zip(y_pred, y_gold)]) / float(y.shape[0])


def load(in_model_folder):
    model = keras.models.load_model(os.path.join(in_model_folder, MODEL_NAME),
                                    custom_objects={'f1': f1})
    with open(os.path.join(in_model_folder, VOCABULARY_NAME)) as vocab_in:
        vocab = json.load(vocab_in)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME)) as char_vocab_in:
        char_vocab = json.load(char_vocab_in)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME)) as label_vocab_in:
        label_vocab = json.load(label_vocab_in)
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

