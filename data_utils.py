import string
from collections import defaultdict
from operator import itemgetter
import logging

import keras

PAD_ID = 0
UNK_ID = 1
PAD = '_PAD'
UNK = '_UNK'


def make_char_vocabulary():
    alphabet = filter(lambda x: x not in string.uppercase, string.printable)
    tokens = [PAD, UNK] + list(alphabet)
    vocab = {token: index for (index, token) in enumerate(tokens)}
    return vocab


def make_vocabulary(in_lines, max_vocabulary_size, special_tokens=[PAD, UNK], frequency_threshold=3):
    freqdict = defaultdict(lambda: 0)
    for line in in_lines:
        for token in line:
            freqdict[token] += 1
    vocab = sorted(freqdict.items(), key=itemgetter(1), reverse=True)
    vocab = filter(lambda x: frequency_threshold < x[1], vocab)
    logging.info('{} tokens ({}% of the vocabulary) were filtered due to the frequency threshold'
                 .format(len(freqdict) - len(vocab), 100.0 * len(vocab) / float(len(freqdict))))
    rev_vocab = (special_tokens + map(itemgetter(0), vocab))[:max_vocabulary_size]
    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sequences(in_sequences, in_vocab):
    sequences_vectorized = []
    for sequence in in_sequences:
        sequences_vectorized.append([in_vocab.get(token, UNK_ID) for token in sequence])
    return sequences_vectorized


def pad_sequences(in_sequences, in_max_input_length, value=PAD_ID):
    return keras.preprocessing.sequence.pad_sequences(in_sequences,
                                                      value=value,
                                                      maxlen=in_max_input_length,
                                                      padding='pre')


