
# coding: utf-8

# In[20]:


import random
import os
from collections import deque

import sklearn as sk
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

from data_utils import make_vocabulary
from deep_disfluency_utils import get_tag_mapping


# In[2]:
from dialogue_denoiser_lstm import get_class_weight_proportional, get_sample_weight

DATASET_FOLDER = 'deep_disfluency_dataset_timings'
random.seed(273)
np.random.seed(273)


# In[3]:


def make_gbdt_dataset(in_utterances, in_tags, in_vocab, in_label_vocab, ngram_sizes=[1, 2, 3]):
    data_points, y = [], []
    for utterance, tags in zip(in_utterances, in_tags):
        ngram_windows = [deque([], maxlen=size) for size in ngram_sizes]
        data_point = np.zeros(len(in_vocab), dtype=np.int32)
        for token, tag in zip(utterance, tags):
            for ngram_window in ngram_windows:
                ngram_window.append(token)
                if len(ngram_window) == ngram_window.maxlen:
                    ngram_id = in_vocab.get(' '.join(ngram_window))
                    if ngram_id:
                        data_point[ngram_id] = 1
            data_points.append(np.array(data_point))
            y.append(in_label_vocab[tag])
    return np.array(data_points), y


# In[4]:


trainset, devset, testset = (pd.read_json(os.path.join(DATASET_FOLDER, 'trainset.json')),
                             pd.read_json(os.path.join(DATASET_FOLDER, 'devset.json')),
                             pd.read_json(os.path.join(DATASET_FOLDER, 'testset.json')))


# In[5]:


vocab, _ = make_vocabulary(trainset['utterance'], 20000, ngram_sizes=[1, 2, 3])
label_vocab, _ = make_vocabulary(trainset['tags'].values,
                                 20000,
                                 special_tokens=[])


# In[6]:


X, y = make_gbdt_dataset(trainset['utterance'], trainset['tags'], vocab, label_vocab)
X_test, y_test = make_gbdt_dataset(testset['utterance'], testset['tags'], vocab, label_vocab)
class_weight = get_class_weight_proportional(y, smoothing_coef=1.05)
sample_weight = get_sample_weight(y, class_weight)

# In[7]:


cls = GradientBoostingClassifier(n_estimators=256, max_depth=10)


# In[9]:


cls.fit(X, y)


# In[14]:


y_pred = cls.predict(X_test)


# In[21]:


tag_map = get_tag_mapping(label_vocab)
result_map = {}

for class_name, class_ids in tag_map.iteritems():
    result_map['f1_' + class_name] = sk.metrics.f1_score(y_true=y_test,
                                                         y_pred=y_pred,
                                                         labels=class_ids,
                                                         average='micro')


# In[23]:


print result_map

