import torch
import fastai
from fastai.text import language_model_learner, AWD_LSTM, get_language_model, untar_data, pickle, convert_weights, \
    LinearDecoder, SequentialRNN


def get_ulmfit_model(in_rev_vocab):
    model = get_language_model(AWD_LSTM, len(in_rev_vocab))
    model_path = untar_data(fastai.text.learner._model_meta[AWD_LSTM]['url'], data=False)
    fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]

    wgts_fname, itos_fname = fnames
    "Load a pretrained model and adapts it to the data vocabulary."
    old_itos = pickle.load(open(itos_fname, 'rb'))
    old_stoi = {v: k for k, v in enumerate(old_itos)}
    wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
    if 'model' in wgts:
        wgts = wgts['model']
    wgts = convert_weights(wgts, old_stoi, in_rev_vocab)
    model.load_state_dict(wgts)
    return model


class AWD_LSTM_DisfluencyDetector(torch.nn.Module):
    def __init__(self, in_word_vocab, in_word_rev_vocab, in_disf_tag_vocab, in_disf_tag_rev_vocab, in_config):
        torch.nn.Module.__init__(self)
        self.awd_lstm_lm = get_ulmfit_model(in_word_rev_vocab)
        awd_lstm = self.awd_lstm_lm._modules['0']
        disfluency_decoder = LinearDecoder(len(in_disf_tag_vocab), in_config['embedding_size'], output_p=0.1)
        self.disfluency_tag_predictor = SequentialRNN(awd_lstm, disfluency_decoder)

    def forward(self):
        pass

    def create_model(in_vocab_size, in_cell_size, ):
        X = tf.placeholder(tf.int32, [None, in_max_input_length], name='X')
        ys_for_tasks = [tf.placeholder(tf.float32, [None, task_i_output_dimensions], name='y_{}'.format(task_idx))
                        for task_idx, task_i_output_dimensions in enumerate(in_task_output_dimensions)]
        embeddings = tf.Variable(tf.random_uniform([in_vocab_size, in_cell_size], -1.0, 1.0),
                                 name='emb')
        emb = tf.nn.embedding_lookup(embeddings, X)

        lstm_cell = rnn.BasicLSTMCell(in_cell_size, forget_bias=1.0, name='lstm')
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, emb, dtype=tf.float32)

        W_for_tasks = [tf.Variable(tf.random_normal([in_cell_size, task_i_output_dim]),
                                   name='W_{}'.format(task_idx))
                       for task_idx, task_i_output_dim in enumerate(in_task_output_dimensions)]
        b_for_tasks = [tf.Variable(tf.random_normal([task_i_output_dim]),
                                   name='bias_{}'.format(task_idx))
                       for task_idx, task_i_output_dim in enumerate(in_task_output_dimensions)]

        task_outputs = [tf.add(tf.matmul(outputs[:, -1, :], W_task), b_task)
                        for W_task, b_task in zip(W_for_tasks, b_for_tasks)]
        return X, tuple(ys_for_tasks), task_outputs
