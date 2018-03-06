from collections import defaultdict
from operator import itemgetter

import keras
import numpy as np

PAD_ID = 0
UNK_ID = 1
PAD = '_PAD'
UNK = '_UNK'


def make_vocabulary(in_lines, max_vocabulary_size):
    freqdict = defaultdict(lambda: 0)
    for line in in_lines:
        for token in line:
            freqdict[token] += 1
    vocab = sorted(freqdict.items(), key=itemgetter(1), reverse=True)
    rev_vocab = ([PAD, UNK] + map(itemgetter(0), vocab))[:max_vocabulary_size]
    vocab = {word: idx for idx, word in enumerate(rev_vocab)}
    return vocab, rev_vocab


def vectorize_sequences(in_sequences, in_vocab, max_input_length):
    sequences_vectorized = []
    for sequence in in_sequences:
        sequences_vectorized.append([in_vocab.get(token, UNK_ID) for token in sequence])
    return keras.preprocessing.sequence.pad_sequences(sequences_vectorized,
                                                      value=PAD_ID,
                                                      maxlen=max_input_length)


def to_one_hot(in_sequence, in_classes_number):
    result = np.zeros((len(in_sequence), in_classes_number))
    for idx, element in enumerate(in_sequence):
        result[idx][element] = 1
    return result