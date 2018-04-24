from keras import backend as K

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score


class ZeroPaddedF1Score(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []


    def on_epoch_end(self, epoch, logs={}):
        val_f1 = zero_padded_f1(self.validation_data[1],
                                self.model.predict(self.validation_data[:1]))
        self.val_f1s.append(val_f1)
        print u' - val_f1: {}'.format(u' - '.join([u'{}: {:.3f}'.format(class_id, class_f1)
                                                   for class_id, class_f1 in enumerate(val_f1)]))


def zero_padded_f1(y_true, y_pred):
    y_true_flat = np.argmax(y_true, axis=-1).flatten()
    y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
    result = f1_score(y_true_flat, y_pred_flat, labels=range(1, y_true.shape[-1]), average=None)
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
