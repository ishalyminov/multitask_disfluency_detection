import json
import os
import random
import pickle

import numpy as np
import torch
import fastai
from fastai.text import AWD_LSTM, get_language_model, untar_data, convert_weights, LinearDecoder, awd_lstm_lm_config

from data_utils import reverse_dict

VOCABULARY_NAME = 'vocab.json'
CHAR_VOCABULARY_NAME = 'char_vocab.json'
POS_VOCABULARY_NAME = 'pos_vocab.json'
LABEL_VOCABULARY_NAME = 'label_vocab.json'
EVAL_LABEL_VOCABULARY_NAME = 'eval_label_vocab.json'
CONFIG_NAME = 'config.json'
MODEL_FILE = 'model.pth'

random.seed(273)
np.random.seed(273)
torch.manual_seed(273)


def get_ulmfit_model(in_vocab):
    model = get_language_model(AWD_LSTM, len(in_vocab))
    model_path = untar_data(fastai.text.learner._model_meta[AWD_LSTM]['url'], data=False)
    fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]

    wgts_fname, itos_fname = fnames
    "Load a pretrained model and adapts it to the data vocabulary."
    old_itos = pickle.load(open(itos_fname, 'rb'))
    old_stoi = {v: k for k, v in enumerate(old_itos)}
    wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
    if 'model' in wgts:
        wgts = wgts['model']
    wgts = convert_weights(wgts, old_stoi, in_vocab)
    model.load_state_dict(wgts)
    return model


def load(in_model_folder, existing_model=None):
    with open(os.path.join(in_model_folder, VOCABULARY_NAME)) as vocab_in:
        vocab = json.load(vocab_in)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME)) as char_vocab_in:
        char_vocab = json.load(char_vocab_in)
    with open(os.path.join(in_model_folder, POS_VOCABULARY_NAME)) as pos_vocab_in:
        pos_vocab = json.load(pos_vocab_in)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME)) as label_vocab_in:
        label_vocab = json.load(label_vocab_in)
    with open(os.path.join(in_model_folder, CONFIG_NAME)) as config_in:
        config = json.load(config_in)
    task_output_dimensions = []
    for task in config['tasks']:
        if task == 'tag':
            task_output_dimensions.append(len(label_vocab))
        elif task == 'lm':
            task_output_dimensions.append(len(vocab))
        else:
            raise NotImplementedError
    if not existing_model:
        model = locals()(config['model_class'])(vocab, reverse_dict(vocab), label_vocab, reverse_dict(label_vocab), config)
    else:
        model = existing_model
    model.load_state_dict(torch.load(os.path.join(in_model_folder, MODEL_FILE)))
    return model, config, vocab, char_vocab, label_vocab


def save(in_model, in_config, in_vocab, in_pos_vocab, in_char_vocab, in_label_vocab, in_model_folder):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    with open(os.path.join(in_model_folder, CONFIG_NAME), 'w') as config_out:
        json.dump(in_config, config_out)
    with open(os.path.join(in_model_folder, VOCABULARY_NAME), 'w') as vocab_out:
        json.dump(in_vocab, vocab_out)
    with open(os.path.join(in_model_folder, POS_VOCABULARY_NAME), 'w') as pos_vocab_out:
        json.dump(in_pos_vocab, pos_vocab_out)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME), 'w') as char_vocab_out:
        json.dump(in_char_vocab, char_vocab_out)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME), 'w') as label_vocab_out:
        json.dump(in_label_vocab, label_vocab_out)
    torch.save(in_model.state_dict(), os.path.join(in_model_folder, MODEL_FILE))


class AWD_LSTM_DisfluencyDetector(torch.nn.Module):
    def __init__(self, in_word_vocab, in_word_rev_vocab, in_disf_tag_vocab, in_disf_tag_rev_vocab, in_config):
        torch.nn.Module.__init__(self)
        self.word_rev_vocab = in_word_rev_vocab
        self.disf_tag_rev_vocab = in_disf_tag_rev_vocab
        self.awd_lstm_lm = get_ulmfit_model(in_word_vocab)
        self.disf_tag_decoder = LinearDecoder(len(in_disf_tag_vocab), awd_lstm_lm_config['emb_sz'], output_p=0.1)
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def forward(self, in_x):
        self.awd_lstm_lm.reset()
        lm_decoded, raw_lstm_outputs, lstm_outputs_dropped_out = self.awd_lstm_lm(in_x)
        disf_decoded, _, _ = self.disf_tag_decoder((raw_lstm_outputs, lstm_outputs_dropped_out))
        return [disf_decoded[:, -1, :], lm_decoded[:, -1, :]]

    def predict(self, in_x):
        logits_for_tasks = self.forward(in_x)
        predicted_ids = [torch.argmax(logits_i, dim=1) for logits_i in logits_for_tasks]
        predicted_tokens = []
        for rev_label_vocab, predicted_ids_i in zip([self.disf_tag_rev_vocab, self.word_rev_vocab], predicted_ids):
            predicted_tokens_i = [rev_label_vocab.get(token)
                                  for token in predicted_ids_i.cpu().numpy()]
            predicted_tokens.append(predicted_tokens_i)
        return predicted_tokens


class DualEncoder_AWD_LSTM_DisfluencyDetector(torch.nn.Module):
    def __init__(self, in_word_vocab, in_word_rev_vocab, in_disf_tag_vocab, in_disf_tag_rev_vocab, in_config):
        torch.nn.Module.__init__(self)
        self.word_rev_vocab = in_word_rev_vocab
        self.disf_tag_rev_vocab = in_disf_tag_rev_vocab
        self.awd_lstm_lm = get_ulmfit_model(in_word_vocab)
        self.pos_tag_embedding = torch.nn.Embedding(in_config['pos_embedding_size'])
        self.pos_tag_encoder = torch.nn.LSTM(in_config['pos_embedding_size'])
        self.disf_tag_decoder = LinearDecoder(len(in_disf_tag_vocab),
                                              awd_lstm_lm_config['emb_sz'] + in_config['pos_embedding_size'],
                                              output_p=0.1)
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def forward(self, in_x):
        word_input, pos_input = in_x
        self.awd_lstm_lm.reset()
        lm_decoded, raw_lstm_outputs, lstm_outputs_dropped_out = self.awd_lstm_lm(word_input)
        pos_tag_encoded = self.pos_tag_encoder(self.pos_tag_embedding(pos_input))

        combined_input = torch.cat([lstm_outputs_dropped_out, pos_tag_encoded], dim=-1)
        disf_decoded, _, _ = self.disf_tag_decoder((None, combined_input))
        return [disf_decoded[:, -1, :], lm_decoded[:, -1, :]]

    def predict(self, in_x):
        logits_for_tasks = self.forward(in_x)
        predicted_ids = [torch.argmax(logits_i, dim=1) for logits_i in logits_for_tasks]
        predicted_tokens = []
        for rev_label_vocab, predicted_ids_i in zip([self.disf_tag_rev_vocab, self.word_rev_vocab], predicted_ids):
            predicted_tokens_i = [rev_label_vocab.get(token)
                                  for token in predicted_ids_i.cpu().numpy()]
            predicted_tokens.append(predicted_tokens_i)
        return predicted_tokens


class MultitaskDisfluencyDetector(torch.nn.Module):
    def __init__(self, in_word_vocab, in_disf_vocab, in_config):
        torch.nn.Module.__init__(self)
        self.emb = torch.nn.Embedding(len(in_word_vocab), 256, padding_idx=1)
        self.lstm = torch.nn.LSTM(256, 256, 1, batch_first=True)
        self.lm_head = torch.nn.Linear(256, len(in_word_vocab))
        self.disf_head = torch.nn.Linear(256, len(in_disf_vocab))
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def forward(self, in_x):
        encoding = self.lstm(self.emb(in_x))[1][0]
        disf_tag = self.disf_head(encoding).squeeze(0)
        lm_tag = self.lm_head(encoding).squeeze(0)
        return [disf_tag, lm_tag]
