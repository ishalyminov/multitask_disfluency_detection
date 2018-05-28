import string
from collections import defaultdict, deque
from operator import itemgetter
import logging
from itertools import chain

import tensorflow as tf

PAD_ID = 0
UNK_ID = 1
PAD = '_PAD'
UNK = '_UNK'


def make_char_vocabulary():
    alphabet = filter(lambda x: x not in string.uppercase, string.printable)
    tokens = [PAD, UNK] + list(alphabet)
    vocab = {token: index for (index, token) in enumerate(tokens)}
    return vocab


def make_vocabulary(in_lines,
                    max_vocabulary_size,
                    special_tokens=(PAD, UNK),
                    frequency_threshold=3,
                    ngram_sizes=(1,)):
    freqdict = defaultdict(lambda: 0)

    for line in in_lines:
        ngram_windows = [deque([], maxlen=size) for size in ngram_sizes]
        for token in line:
            for window in ngram_windows:
                window.append(token)
                if len(window) == window.maxlen:
                    freqdict[' '.join(window)] += 1
    vocab = sorted(freqdict.items(), key=itemgetter(1), reverse=True)
    vocab = filter(lambda x: frequency_threshold < x[1], vocab)
    logging.info('{} tokens ({}% of the vocabulary) were filtered due to the frequency threshold'
                 .format(len(freqdict) - len(vocab), 100.0 * len(vocab) / float(len(freqdict))))
    rev_vocab = (list(special_tokens) + map(itemgetter(0), vocab))[:max_vocabulary_size]
    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sequences(in_sequences, in_vocab):
    sequences_vectorized = []
    for sequence in in_sequences:
        sequences_vectorized.append([in_vocab.get(token, UNK_ID) for token in sequence])
    return sequences_vectorized


def pad_sequences(in_sequences, in_max_input_length, value=PAD_ID):
    return tf.keras.preprocessing.sequence.pad_sequences(in_sequences,
                                                         value=value,
                                                         maxlen=in_max_input_length,
                                                         padding='pre')


def create_contexts(in_tokens, in_max_input_length):
    contexts = []
    context = deque([], maxlen=in_max_input_length)
    for token in in_tokens:
        context.append(token)
        contexts.append(list(context))
    return contexts


def make_multitask_dataset(in_dataset, in_vocab, in_label_vocab, in_config):
    utterances, contexts = [], []
    for idx, row in in_dataset.iterrows():
        if in_config['use_pos_tags']:
            utterance = ['{}_{}'.format(token, pos)
                         for token, pos in zip(row['utterance'], row['pos'])]
        else:
            utterance = row['utterance']
        utterances.append(utterance)
        current_contexts = create_contexts(utterance,
                                           in_config['max_input_length'])
        contexts += current_contexts
    tokens_vectorized = vectorize_sequences(contexts, in_vocab)
    tokens_padded = pad_sequences(tokens_vectorized, in_config['max_input_length'])

    ys_for_tasks = []
    for task in in_config['tasks']:
        if task == 'tag':
            labels = vectorize_sequences(in_dataset['tags'], in_label_vocab)
            labels = list(chain(*labels))
            y_i = tf.keras.utils.to_categorical(labels, num_classes=len(in_label_vocab))
        elif task == 'lm':
            label_sequences = [utterance[1:] + [PAD] for utterance in utterances]
            labels = vectorize_sequences(label_sequences, in_vocab)
            labels = list(chain(*labels))
            y_i = tf.keras.utils.to_categorical(labels, num_classes=len(in_vocab))
        else:
            raise NotImplementedError
        ys_for_tasks.append(y_i)
    return tokens_padded, ys_for_tasks


def make_dataset(in_dataset, in_vocab, in_label_vocab, in_config):
    contexts, tags = [], []
    for idx, row in in_dataset.iterrows():
        if in_config['use_pos_tags']:
            utterance = ['{}_{}'.format(token, pos)
                         for token, pos in zip(row['utterance'], row['pos'])]
        else:
            utterance = row['utterance']
        current_contexts, current_tags = (create_contexts(utterance,
                                                          in_config['max_input_length']),
                                          row['tags'])
        contexts += current_contexts
        tags += current_tags
    tokens_vectorized = vectorize_sequences(contexts, in_vocab)
    tokens_padded = pad_sequences(tokens_vectorized, in_config['max_input_length'])

    labels = vectorize_sequences([tags], in_label_vocab)
    y = tf.keras.utils.to_categorical(labels[0], num_classes=len(in_label_vocab))
    return tokens_padded, y


