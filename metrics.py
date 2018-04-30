from keras import backend as K

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score


class DisfluencyDetectionF1Score(Callback):
    def __init__(self, in_tag_clusters={}):
        self.tag_clusters = in_tag_clusters

    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.argmax(self.model.predict(self.validation_data[:1]), axis=-1)
        y_true = np.argmax(self.validation_data[1], axis=-1)
        val_f1_map = {}
        for name, tags in self.tag_clusters.iteritems():
            val_f1 = zero_padded_f1(y_true,
                                    y_pred,
                                    labels=tags)
            val_f1_map[name] = val_f1
        self.val_f1s.append(val_f1_map)
        print u' - val_f1: {}'.format(u' - '.join([u'{}: {:.3f}'.format(name, f1)
                                                   for name, f1 in val_f1_map.iteritems()]))


def zero_padded_f1(y_true, y_pred, labels=None):
    result = f1_score(y_true, y_pred, labels=labels, average='macro')
    return result 


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


def f1_binary(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def f1(classes_number, y_true, y_pred):
    result = 0.0
    for class_id in xrange(1, classes_number + 1):
        y_true_single_class = y_true[:,:,class_id]
        y_pred_single_class = y_pred[:,:,class_id]
        f1_single = f1_binary(y_true_single_class, y_pred_single_class)
        result += f1_single / float(classes_number)
    return result
