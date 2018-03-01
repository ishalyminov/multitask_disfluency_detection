
# coding: utf-8

# In[189]:


import re
from collections import defaultdict
import string
import os

import numpy as np
import nltk

from deep_disfluency.corpus import swda


# In[46]:


reader = swda.CorpusReader('../swda/swda')


# In[92]:


def extract_nonspeech(in_utterance):
    return map(lambda x: ('nonspeech', x), re.findall('<[^<]+?>', in_utterance))


def filter_nonspeech(in_utterance):
    return re.sub('<[^<]+?>', '', in_utterance).strip()


# In[101]:


NONSENTENCE_RE = '({(\w) ([^{]+?)})'


def extract_nonsentence(in_utterance):
    return map(lambda (body, disfluency_type, text): ('{{{}...}}'.format(disfluency_type), (body, text)),
               re.findall(NONSENTENCE_RE, in_utterance))


def filter_nonsentence(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for disfluency_type, (body, text) in extract_nonsentence(in_utterance):
        disfluent = disfluent.replace(body, text)
        clean = clean.replace(body, '')
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


# In[102]:


print extract_nonsentence('Actually, {F uh, }')
print filter_nonsentence('Actually, {F uh, }')


# In[134]:


def extract_restarts_with_repair_and_nonsentece(in_utterance):
    return map(lambda x: ('[RM + {} RR]', x),
               re.findall('(\[([^[]+?)\s+\+\s+{}\s+([^[]+?)\])'.format(NONSENTENCE_RE), in_utterance))


def filter_restarts_with_repair_and_nonsentece(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm, nonsent_body, nonsent_type, nonsent_text, rr) in extract_restarts_with_repair_and_nonsentece(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm, nonsent_text, rr]))
        clean = clean.replace(body, rr)
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


# In[135]:


print extract_restarts_with_repair_and_nonsentece('Actually, [ I, + {F uh, } I] guess I am [I, + I], {F uh, }')
print filter_restarts_with_repair_and_nonsentece('Actually, [ I, + {F uh, } I] guess I am [I, + I], {F uh, }')


# In[138]:


def extract_restarts_with_repair(in_utterance):
    return map(lambda x: ('[RM + RR]', x),
               re.findall('(\[([^[]+?)\s+\+\s+([^[{]+?)\])', in_utterance))


def filter_restarts_with_repair(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm, rr) in extract_restarts_with_repair(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm, rr]))
        clean = clean.replace(body, rr)
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


# In[139]:


print extract_restarts_with_repair('Actually, [ I, + {F uh, } I] guess I am [I, + I], {F uh, }')
print filter_restarts_with_repair('Actually, [ I, + {F uh, } I] guess I am [I, + I], {F uh, }')


# In[144]:


def extract_restarts_without_repair(in_utterance):
    return map(lambda x: ('[RM +]', x),
               re.findall('(\[([^[]+?)\s+\+\s+\])', in_utterance))


def filter_restarts_without_repair(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm) in extract_restarts_without_repair(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm]))
        clean = clean.replace(body, '')
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


# In[145]:


print extract_restarts_without_repair('Actually, [ I + ] guess I am [I, + I], {F uh, }')
print filter_restarts_without_repair('Actually, [ I + ] guess I am [I, + I], {F uh, }')


# In[172]:


pipeline = [
    (extract_restarts_with_repair_and_nonsentece, filter_restarts_with_repair_and_nonsentece),
    (extract_restarts_with_repair, filter_restarts_with_repair),
    (extract_restarts_without_repair, filter_restarts_without_repair),
    (extract_nonsentence, filter_nonsentence),
]
disfluency_stats = defaultdict(lambda: 0)
parallel_corpus = []
for utt in reader.iter_utterances(display_progress=False):
    local_disfluencies = defaultdict(lambda: 0)
    utt_filtered = filter_nonspeech(utt.text)
    utt_original, utt_clean = utt_filtered, utt_filtered
    for extract_step, filter_step in pipeline:
        disfluencies = extract_step(utt_original)
        if not len(disfluencies):
            continue
        utt_original, utt_clean = filter_step(utt_original)[0], filter_step(utt_clean)[1]
        if not len(re.findall('\w+', utt_clean)):
            break
        for disfluency_type, disfluency in disfluencies:
            local_disfluencies[disfluency_type] += 1
    if local_disfluencies and len(re.findall('\w+', utt_clean)):
        for disfluency_type, count in local_disfluencies.iteritems():
            disfluency_stats[disfluency_type] += count
        parallel_corpus.append((utt_original, utt_clean))


# In[171]:


# 30% of utternaces in the final corpus will be fluent
fluent_corpus_size = int(0.3 * len(parallel_corpus) / 0.7)
parallel_corpus_fluent = []
for utt in reader.iter_utterances(display_progress=False):
    utt_filtered = filter_nonspeech(utt.text)
    utt_original, utt_clean = utt_filtered, utt_filtered
    fluent = True
    for extract_step, filter_step in pipeline:
        disfluencies = extract_step(utt_original)
        if not len(disfluencies):
            fluent = False
            break
    if fluent and len(re.findall('\w+', utt_clean)):
        parallel_corpus_fluent.append((utt_clean, utt_clean))
    if fluent_corpus_size == len(parallel_corpus_fluent):
        break


# Corpus stats
# ==

# In[186]:


final_corpus = [(nltk.word_tokenize(utt_from.translate(None, string.punctuation)), nltk.word_tokenize(utt_to.translate(None, string.punctuation)))
                for utt_from, utt_to in parallel_corpus + parallel_corpus_fluent]


# In[188]:


print 'Total number of utterances with disfluencies: {}'.format(len(parallel_corpus))
print 'Total number of utterances without disfluencies: {}'.format(len(parallel_corpus_fluent))
print 'Mean utterance length (utterance_from): {:.3f}'.format(np.mean([len(utt_from) for utt_from, utt_to in final_corpus]))
print 'Mean utterance length (utterance_to): {:.3f}'.format(np.mean([len(utt_to) for utt_from, utt_to in final_corpus]))
print 'Dislfuency stats by type:'
for key, value in disfluency_stats.iteritems():
    print '{}:\t{}'.format(key, value)


# In[192]:


out_folder = 'swda_parallel_corpus'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
with open(os.path.join(out_folder, 'encoder.txt'), 'w') as encoder_out,      open(os.path.join(out_folder, 'decoder.txt'), 'w') as decoder_out:
    for utt_from, utt_to in final_corpus:
        print >>encoder_out, ' '.join(utt_from)
        print >>decoder_out, ' '.join(utt_to)

