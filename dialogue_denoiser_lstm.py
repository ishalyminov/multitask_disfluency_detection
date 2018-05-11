import json
import random
import os
import sys
from copy import deepcopy

import sklearn as sk
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd

from data_utils import make_multitask_dataset
from training_utils import get_loss_function, batch_generator

THIS_FILE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))

from deep_disfluency.utils.tools import (convert_from_eval_tags_to_inc_disfluency_tags,
                                         convert_from_inc_disfluency_tags_to_eval_tags)
from deep_disfluency.utils.tools import convert_from_inc_disfluency_tags_to_eval_tags
from deep_disfluency.evaluation.disf_evaluation import incremental_output_disfluency_eval_from_file
from deep_disfluency.evaluation.disf_evaluation import final_output_disfluency_eval_from_file
from deep_disfluency.evaluation.eval_utils import get_tag_data_from_corpus_file
from deep_disfluency.evaluation.eval_utils import rename_all_repairs_in_line_with_index
from deep_disfluency_utils import get_tag_mapping

random.seed(273)
np.random.seed(273)
tf.set_random_seed(273)

MODEL_NAME = 'ckpt'
VOCABULARY_NAME = 'vocab.json'
CHAR_VOCABULARY_NAME = 'char_vocab.json'
LABEL_VOCABULARY_NAME = 'label_vocab.json'
EVAL_LABEL_VOCABULARY_NAME = 'eval_label_vocab.json'
CONFIG_NAME = 'config.json'

DATA_DIR = os.path.join(THIS_FILE_DIR,
                        'deep_disfluency',
                        'deep_disfluency',
                        'data',
                        'disfluency_detection',
                        'switchboard')
DEFAULT_HELDOUT_DATASET = DATA_DIR + '/swbd_disf_heldout_data_timings.csv'


def train(in_model,
          train_data,
          dev_data,
          test_data,
          in_task_vocabs,
          in_model_folder,
          in_epochs_number,
          config,
          session,
          class_weights=None,
          task_weights=None,
          **kwargs):
    X_train, y_train_for_tasks = train_data

    tag_mapping = get_tag_mapping(in_task_vocabs[0][1])

    X, ys_for_tasks, logits_for_tasks = in_model

    if class_weights is None:
        class_weights = [np.ones(y_train_i.shape[1]) for y_train_i in y_train_for_tasks]
    if task_weights is None:
        task_weights = np.ones(len(y_train_for_tasks))
    # Define loss and optimizer
    loss_op = get_loss_function(logits_for_tasks,
                                ys_for_tasks,
                                class_weights,
                                l2_coef=config['l2_coef'],
                                task_weights=task_weights)

    starting_lr = config['lr']
    lr_decay = config['lr_decay']
    global_step = tf.Variable(0, trainable=False)
    session.run(tf.assign(global_step, 0))
    learning_rate = tf.train.exponential_decay(starting_lr,
                                               global_step,
                                               100000,
                                               lr_decay,
                                               staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step)

    saver = tf.train.Saver(tf.global_variables())

    best_dev_loss = np.inf
    epochs_without_improvement = 0
    for epoch_counter in xrange(in_epochs_number):
        batch_gen = batch_generator(X_train,
                                    y_train_for_tasks,
                                    config['batch_size'])
        train_batch_losses = []
        for batch_x, batch_y in batch_gen:
            _, train_batch_loss = session.run([train_op, loss_op],
                                              feed_dict={X: batch_x, ys_for_tasks: batch_y})
            train_batch_losses.append(train_batch_loss)
        _, dev_eval = evaluate(in_model,
                               dev_data,
                               tag_mapping,
                               class_weights,
                               task_weights,
                               config,
                               session)
        print 'Epoch {} out of {} results'.format(epoch_counter, in_epochs_number)
        print 'train loss: {:.3f}'.format(np.mean(train_batch_losses))
        print '; '.join(['dev {}: {:.3f}'.format(key, value)
                         for key, value in dev_eval.iteritems()])
        if dev_eval['loss'] < best_dev_loss:
            best_dev_loss = dev_eval['loss']
            saver.save(session, os.path.join(in_model_folder, MODEL_NAME))
            print 'New best loss. Saving checkpoint'
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if config['early_stopping_threshold'] < epochs_without_improvement:
            print 'Early stopping after {} epochs'.format(epoch_counter)
            break

    print 'Optimization Finished!'


def evaluate(in_model,
             in_dataset,
             in_tag_map,
             in_class_weights,
             in_task_weights,
             in_config,
             in_session,
             batch_size=32):
    X_test, y_test_for_tasks = in_dataset
    X, ys_for_tasks, logits_for_tasks = in_model

    # Evaluate model (with test logits, for dropout to be disabled)
    loss_op = get_loss_function(logits_for_tasks,
                                ys_for_tasks,
                                in_class_weights,
                                l2_coef=in_config['l2_coef'],
                                task_weights=in_task_weights)
    y_pred_op = [tf.argmax(logits_i, 1) for logits_i in logits_for_tasks]
    y_true_op = [tf.argmax(y_i, 1) for y_i in ys_for_tasks]

    correct_pred = tf.equal(y_pred_op, y_true_op)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Start training
    batch_gen = batch_generator(X_test, y_test_for_tasks, batch_size)

    batch_losses, batch_accuracies = [], []
    y_pred_main_task = np.zeros(X_test.shape[0])
    for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
        y_pred_batch, loss_batch, acc_batch = in_session.run([y_pred_op, loss_op, accuracy],
                                                              feed_dict={X: batch_x,
                                                                         ys_for_tasks: batch_y})
        y_pred_main_task[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred_batch[0]
        batch_losses.append(loss_batch)
        batch_accuracies.append(acc_batch)

    y_gold_main_task = np.argmax(y_test_for_tasks[0], -1)
    result_map = {'loss': np.mean(batch_losses), 'acc': np.mean(batch_accuracies)}
    for class_name, class_ids in in_tag_map.iteritems():
        result_map['f1_' + class_name] = sk.metrics.f1_score(y_true=y_gold_main_task
                                                             ,
                                                             y_pred=y_pred_main_task,
                                                             labels=class_ids,
                                                             average='micro')
    return y_pred_main_task, result_map


def predict(in_model, in_dataset, in_vocabs_for_tasks, in_session, batch_size=32):
    X_test, y_test_for_tasks = in_dataset
    X, ys_for_tasks, logits_for_tasks = in_model

    y_pred_op = [tf.argmax(logits_i, 1) for logits_i in logits_for_tasks]

    # Start training
    batch_gen = batch_generator(X_test, y_test_for_tasks, batch_size)

    y_pred_main_task = np.zeros(X_test.shape[0])
    for batch_idx, (batch_x, batch_ys) in enumerate(batch_gen):
        y_pred_batch = in_session.run(y_pred_op,
                                       feed_dict={X: batch_x})
        y_pred_main_task[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred_batch[0]

    rev_label_vocab_main_task = in_vocabs_for_tasks[0][2]
    predictions = map(rev_label_vocab_main_task.get, y_pred_main_task) 
    return predictions


def predict_increco_file(in_model,
                         vocabs_for_tasks,
                         source_file_path,
                         in_config,
                         in_session,
                         target_file_path=None,
                         is_asr_results_file=False):
    """Return the incremental output in an increco style
    given the incoming words + POS. E.g.:

    Speaker: KB3_1

    Time: 1.50
    KB3_1:1    0.00    1.12    $unc$yes    NNP    <f/><tc/>

    Time: 2.10
    KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
    KB3_1:2    1.12    2.00     because    IN    <rps id="1"/><cc/>

    Time: 2.5
    KB3_1:2    1.12    2.00     because    IN    <rps id="1"/><rpndel id="1"/><cc/>

    from an ASR increco style input without the POStags:

    or a normal style disfluency dectection ground truth corpus:

    Speaker: KB3_1
    KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
    KB3_1:2    1.12    2.00     $because    IN    <rps id="1"/><cc/>
    KB3_1:3    2.00    3.00    because    IN    <f/><cc/>
    KB3_1:4    3.00    4.00    theres    EXVBZ    <f/><cc/>
    KB3_1:6    4.00    5.00    a    DT    <f/><cc/>
    KB3_1:7    6.00    7.10    pause    NN    <f/><cc/>


    :param source_file_path: str, file path to the input file
    :param target_file_path: str, file path to output in the above format
    :param is_asr_results_file: bool, whether the input is increco style
    """
    if target_file_path:
        target_file = open(target_file_path, "w")
    if 'timings' in source_file_path:
        print "input file has timings"
        if not is_asr_results_file:
            dialogues = []
            IDs, timings, words, pos_tags, labels = \
                get_tag_data_from_corpus_file(source_file_path)
            for dialogue, a, b, c, d in zip(IDs,
                                            timings,
                                            words,
                                            pos_tags,
                                            labels):
                dialogues.append((dialogue, (a, b, c, d)))
    else:
        print "no timings in input file, creating fake timings"
        raise NotImplementedError

    # collecting a single dataset for the model to predict in batches
    utterances, tags, pos = [], [], []
    for speaker, speaker_data in dialogues:
        timing_data, lex_data, pos_data, labels = speaker_data
        # iterate through the utterances
        # utt_idx = -1

        for i in range(0, len(timing_data)):
            # print i, timing_data[i]
            _, end = timing_data[i]
            if "<t" in labels[i]:
                utterances.append([])
                tags.append([])
                pos.append([])
            utterances[-1].append(lex_data[i])
            tags[-1].append(labels[i])
            pos[-1].append(pos_data[i])

    # eval tags --> RNN tags
    dataset = pd.DataFrame({'utterance': utterances,
                            'tags': [convert_from_eval_tags_to_inc_disfluency_tags(tags_i,
                                                                                   words_i,
                                                                                   representation="disf1")
                                     for tags_i, words_i in zip(tags, utterances)],
                            'pos': pos})
    X, ys_for_tasks = make_multitask_dataset(dataset,
                                             vocabs_for_tasks[0][0],
                                             vocabs_for_tasks[0][1],
                                             in_config)
    predictions = predict(in_model,
                          (X, ys_for_tasks),
                          vocabs_for_tasks,
                          in_session)
    predictions_eval = []
    global_word_index = 0
    broken_sequences_number = 0
    # RNN tags --> eval tags
    for utterance in utterances:
        current_tags = predictions[global_word_index: global_word_index + len(utterance)]
        try:
            current_tags_eval = convert_from_inc_disfluency_tags_to_eval_tags(current_tags,
                                                                              utterance,
                                                                              representation="disf1")
        except:
            current_tags_eval = current_tags
            broken_sequences_number += 1
        predictions_eval += current_tags_eval
        global_word_index += len(utterance)
    print '#broken sequences after RNN --> eval conversion: {} out of {}'.format(broken_sequences_number, len(utterances))

    predictions_eval_iter = iter(predictions_eval) 
    for speaker, speaker_data in dialogues:
        if target_file_path:
            target_file.write("Speaker: " + str(speaker) + "\n\n")
        timing_data, lex_data, pos_data, labels = speaker_data

        for i in range(0, len(timing_data)):
            # print i, timing_data[i]
            _, end = timing_data[i]
            word = lex_data[i]
            pos = pos_data[i]
            predicted_tags = [next(predictions_eval_iter)]
            current_time = end
            if target_file_path:
                target_file.write("Time: " + str(current_time) + "\n")
                new_words = lex_data[i - (len(predicted_tags) - 1):i + 1]
                new_pos = pos_data[i - (len(predicted_tags) - 1):i + 1]
                new_timings = timing_data[i - (len(predicted_tags) - 1):i + 1]
                for t, w, p, tag in zip(new_timings,
                                        new_words,
                                        new_pos,
                                        predicted_tags):
                    target_file.write("\t".join([str(t[0]),
                                                 str(t[1]),
                                                 w,
                                                 p,
                                                 tag]))
                    target_file.write("\n")
                target_file.write("\n")
        target_file.write("\n")


def eval_deep_disfluency(in_model,
                         in_vocabs_for_tasks,
                         source_file_path,
                         in_config,
                         in_session,
                         verbose=True):
    increco_file = 'swbd_disf_heldout_data_output_increco.text'
    predict_increco_file(in_model,
                         in_vocabs_for_tasks,
                         source_file_path,
                         in_config,
                         in_session,
                         target_file_path=increco_file)
    IDs, timings, words, pos_tags, labels = get_tag_data_from_corpus_file(source_file_path)
    gold_data = {}  # map from the file name to the data
    for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        gold_data[dialogue] = (a, b, c, d)
    final_output_name = increco_file.replace("_increco", "_final")
    incremental_output_disfluency_eval_from_file(increco_file,
                                                 gold_data,
                                                 utt_eval=True,
                                                 error_analysis=True,
                                                 word=True,
                                                 interval=False,
                                                 outputfilename=final_output_name)
    # hyp_dir = experiment_dir
    IDs, timings, words, pos_tags, labels = get_tag_data_from_corpus_file(source_file_path)
    gold_data = {}  # map from the file name to the data
    for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        d = rename_all_repairs_in_line_with_index(list(d))
        gold_data[dialogue] = (a, b, c, d)

    # the below does just the final output evaluation, assuming a final output file, faster
    hyp_file = "swbd_disf_heldout_data_output_final.text"
    word = True  # world-level analyses
    error = True  # get an error analysis
    results, speaker_rate_dict, error_analysis = final_output_disfluency_eval_from_file(
        hyp_file,
        gold_data,
        utt_eval=False,
        error_analysis=error,
        word=word,
        interval=False,
        outputfilename=None
    )
    # the below does incremental and final output in one, also outputting the final outputs
    # derivable from the incremental output, takes quite a while
    if verbose:
        for k, v in results.items():
            print k, v
    all_results = deepcopy(results)

    return {'f1_<rm_word': all_results['f1_<rm_word'],
            'f1_<rps_word': all_results['f1_<rps_word'],
            'f1_<e_word': all_results['f1_<e_word']}


def filter_line(in_line, in_model, in_vocab, in_label_vocab, in_rev_label_vocab, in_session):
    tokens = in_line.lower().split()
    dataset = pd.DataFrame({'utterance': [tokens], 'tags': [['<f/>'] * len(tokens)]})
    X_line, y_line = make_dataset(dataset, in_vocab, in_label_vocab)
    X, y, logits = in_model
    y_pred_op = tf.argmax(logits, 1)
    y_pred  = in_session.run([y_pred_op],
                             feed_dict={X: X_line[0]})
    result_tokens = map(in_rev_label_vocab.get, y_pred[0])
    return ' '.join(result_tokens)


def predict_single_tag(in_line, in_model, in_vocab, in_label_vocab, in_rev_label_vocab, in_session):
    tokens = in_line.lower().split()
    dataset = pd.DataFrame({'utterance': [tokens], 'tags': [['<f/>'] * len(tokens)]})
    X_line, y_line = make_dataset(dataset, in_vocab, in_label_vocab)
    X_last = [X_line[-1]]
    X, y, logits = in_model
    y_pred_op = tf.argmax(logits, 1)
    y_pred  = in_session.run([y_pred_op],
                             feed_dict={X: X_last})
    tags = map(in_rev_label_vocab.get, y_pred[0])
    return tags


def create_model(in_vocab_size, in_cell_size, in_max_input_length, in_task_output_dimensions):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
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


def load(in_model_folder, in_session, existing_model=None):
    with open(os.path.join(in_model_folder, VOCABULARY_NAME)) as vocab_in:
        vocab = json.load(vocab_in)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME)) as char_vocab_in:
        char_vocab = json.load(char_vocab_in)
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
        model = create_model(len(vocab),
                             config['embedding_size'],
                             config['max_input_length'],
                             task_output_dimensions)
    else:
        model = existing_model
    loader = tf.train.Saver()
    loader.restore(in_session, os.path.join(in_model_folder, MODEL_NAME))
    return model, config, vocab, char_vocab, label_vocab


def save(in_config, in_vocab, in_char_vocab, in_label_vocab, in_model_folder, in_session):
    if not os.path.exists(in_model_folder):
        os.makedirs(in_model_folder)
    with open(os.path.join(in_model_folder, CONFIG_NAME), 'w') as config_out:
        json.dump(in_config, config_out)
    with open(os.path.join(in_model_folder, VOCABULARY_NAME), 'w') as vocab_out:
        json.dump(in_vocab, vocab_out)
    with open(os.path.join(in_model_folder, CHAR_VOCABULARY_NAME), 'w') as char_vocab_out:
        json.dump(in_char_vocab, char_vocab_out)
    with open(os.path.join(in_model_folder, LABEL_VOCABULARY_NAME), 'w') as label_vocab_out:
        json.dump(in_label_vocab, label_vocab_out)
    saver = tf.train.Saver()
    saver.save(in_session, os.path.join(in_model_folder, MODEL_NAME))
