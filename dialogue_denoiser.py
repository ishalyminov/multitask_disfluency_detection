# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import shutil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from copy_seq2seq import data_utils, seq2seq_model, seq2seq

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")  # 8
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")  # 32
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 100, "English vocabulary size.")  # 100
tf.app.flags.DEFINE_integer("to_vocab_size", 100, "French vocabulary size.")  # 100
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size",
                            0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",
                            200,
                            "How many training steps to do per checkpoint.")  # 200
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("evaluate", False, "Set to True for evaluation.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("combined_vocabulary",
                            False,
                            "Using a combined encoder/decoder vocabulary")
tf.app.flags.DEFINE_integer("early_stopping_checkpoints",
                            10,
                            "Terminating training after this number of checkpoints of loss increase")
tf.app.flags.DEFINE_boolean("force_make_data",
                            False,
                            "Create datasets even if corresponding files exist")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(20, 30)]


def read_data(encoder_input, decoder_input, decoder_targets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(encoder_input, mode="r") as encoder_input_file, \
            tf.gfile.GFile(decoder_input, mode="r") as decoder_input_file, \
            tf.gfile.GFile(decoder_targets, mode="r") as decoder_targets_file:
        encoder_input, decoder_input, decoder_target = (encoder_input_file.readline(),
                                                        decoder_input_file.readline(),
                                                        decoder_targets_file.readline())
        counter = 0
        while encoder_input and decoder_input and (not max_size or counter < max_size):
            counter += 1
            if counter % 100 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            encoder_ids = [int(x) for x in encoder_input.split()]
            decoder_ids = [int(x) for x in decoder_input.split()]
            decoder_ids.append(data_utils.EOS_ID)
            decoder_target_ids = [
                [int(elem) for elem in x.split(';')] for x in decoder_target.split()
            ]
            decoder_target_ids.append([data_utils.EOS_ID])
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(encoder_ids) < source_size and len(decoder_ids) < target_size:
                    data_set[bucket_id].append([encoder_ids, decoder_ids, decoder_target_ids])
                    break
            encoder_input, decoder_input, decoder_target = (encoder_input_file.readline(),
                                                            decoder_input_file.readline(),
                                                            decoder_targets_file.readline())
    return data_set


def create_model(session, from_vocab_size, to_vocab_size, forward_only, force_create_fresh=False):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(from_vocab_size,
                                       to_vocab_size,
                                       _buckets,
                                       FLAGS.size,
                                       FLAGS.num_layers,
                                       FLAGS.max_gradient_norm,
                                       FLAGS.batch_size,
                                       FLAGS.learning_rate,
                                       FLAGS.learning_rate_decay_factor,
                                       forward_only=forward_only,
                                       dtype=dtype)
    if force_create_fresh:
        if os.path.exists(FLAGS.train_dir):
            shutil.rmtree(FLAGS.train_dir)
            os.makedirs(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = FLAGS.from_dev_data
    to_dev_data = FLAGS.to_dev_data

    from_train, to_train, targets_train, from_dev, to_dev, targets_dev, _, _ = \
        data_utils.prepare_data(FLAGS.data_dir,
                                from_train_data,
                                to_train_data,
                                from_dev_data,
                                to_dev_data,
                                FLAGS.from_vocab_size,
                                FLAGS.to_vocab_size,
                                copy_tokens_number=_buckets[0][1],
                                combined_vocabulary=FLAGS.combined_vocabulary,
                                force=FLAGS.force_make_data)

    enc_vocab_path = os.path.join(FLAGS.data_dir, "vocab.from")
    dec_vocab_path = os.path.join(FLAGS.data_dir, "vocab.to")
    enc_vocab, rev_enc_vocab = data_utils.initialize_vocabulary(enc_vocab_path)
    dec_vocab, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)
    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess,
                             len(enc_vocab),
                             len(dec_vocab),
                             False,
                             force_create_fresh=FLAGS.force_make_data)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        dev_set = read_data(from_dev, to_dev, targets_dev)
        train_set = read_data(from_train, to_train, targets_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        best_loss = None 
        suboptimal_loss_steps = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, decoder_targets, decoder_target_1hots, target_weights = \
                model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         decoder_targets,
                                         decoder_target_1hots,
                                         target_weights,
                                         bucket_id,
                                         False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f loss %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                loss, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                total_eval_loss = 0.0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, decoder_targets, decoder_target_1hots, target_weights = \
                        model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess,
                                                 encoder_inputs,
                                                 decoder_inputs,
                                                 decoder_targets,
                                                 decoder_target_1hots,
                                                 target_weights,
                                                 bucket_id,
                                                 True)
                    total_eval_loss += eval_loss
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d loss %.2f perplexity %.2f" % (
                    bucket_id, eval_loss, eval_ppx))
                    # print("Per-utterance accuracy: {}".format(eval_model(sess, model, from_dev, to_dev, targets_dev)))
                sys.stdout.flush()
                if best_loss is None or total_eval_loss < best_loss:
                    suboptimal_loss_steps = 0
                    best_loss = total_eval_loss
                else:
                    suboptimal_loss_steps += 1
                    if FLAGS.early_stopping_checkpoints <= suboptimal_loss_steps:
                        print("Early stopping after %d checkpoints" % FLAGS.early_stopping_checkpoints)
                        break


def eval_model(in_session, in_model, from_dev_ids_path, to_dev_ids_path, to_dev_target_ids_path):
    original_batch_size = in_model.batch_size
    in_model.batch_size = 64

    # Load vocabularies.
    enc_vocab_path = os.path.join(FLAGS.data_dir, "vocab.from")
    dec_vocab_path = os.path.join(FLAGS.data_dir, "vocab.to")
    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    dec_vocab, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    from_dev_data = FLAGS.from_dev_data
    to_dev_data = FLAGS.to_dev_data
    dataset = read_data(from_dev_ids_path, to_dev_ids_path, to_dev_target_ids_path, max_size=None)
    results = []
    for bucket_id in xrange(len(dataset)):
        bucket_data = dataset[bucket_id]
        for index in xrange(0, len(bucket_data), in_model.batch_size):
            # Get a 1-element batch to feed the sentence to the model.
            enc_in, dec_in, dec_tgt, dec_tgt_1hots, target_weights = \
                in_model.get_batch({bucket_id: bucket_data}, bucket_id, start_index=index)
                #in_model.get_batch({bucket_id: [(encoder_inputs, [], [])] * in_model.batch_size},
                #                   bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits_and_attentions = in_model.step(in_session,
                                                               enc_in,
                                                               dec_in,
                                                               dec_tgt,
                                                               dec_tgt_1hots,
                                                               target_weights,
                                                               bucket_id,
                                                               True)
            #encoder_input_tensors = [tf.convert_to_tensor(enc_input)
            #                         for enc_input in enc_in]
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits_and_attentions[0]]
            #output_tensors = \
            #    [seq2seq.extract_copy_augmented_argmax(logit, attention_dist, encoder_input_tensors)
            #     for logit, attention_dist in output_logits_and_attentions]
            #outputs = [tf.to_int64(output_tensor) for output_tensor in output_tensors]

            # If there is an EOS symbol in outputs, cut them at that point.
            #if data_utils.EOS_ID in outputs:
            #    outputs = outputs[:outputs.index(data_utils.EOS_ID) + 1]
            #print('Gold: ', ' '.join(map(str, decoder_inputs)))
            #print('Pred: ', ' '.join(map(str, outputs)))
            #results.append(int(outputs == decoder_inputs))
            print("Processed {} out of {} data points".format(index, len(bucket_data)))
    in_model.batch_size = original_batch_size
    return sum(results) / float(len(results))


def decode():
    with tf.Session() as sess:
        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.from')
        fr_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.to')
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        fr_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        # Create model and load parameters.
        model = create_model(sess, len(en_vocab), len(fr_vocab), True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning('Sentence truncated: %s', sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_targets, decoder_target_1hots, target_weights = \
                model.get_batch({bucket_id: [(token_ids, [], [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits_and_attentions = model.step(sess,
                                                            encoder_inputs,
                                                            decoder_inputs,
                                                            decoder_targets,
                                                            decoder_target_1hots,
                                                            target_weights,
                                                            bucket_id,
                                                            True)
            encoder_input_tensors = [tf.convert_to_tensor(enc_input) for enc_input in
                                     encoder_inputs]
            output_tensors = \
                [seq2seq.extract_copy_augmented_argmax(logit, attention_dist, encoder_input_tensors)
                 for logit, attention_dist in output_logits_and_attentions]
            outputs = [int(output_tensor.eval()) for output_tensor in output_tensors]
            # This is a greedy decoder - outputs are just argmaxes of output_logits.

            # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            tokens = []
            for output in outputs:
                tokens.append(tf.compat.as_str(rev_fr_vocab[output]))
            print(" ".join(tokens))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def evaluate():
    with tf.Session() as sess:
        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.from')
        fr_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.to')
        from_dev_path = FLAGS.from_dev_data
        to_dev_path = FLAGS.to_dev_data
        from_dev, to_dev, targets_dev = data_utils.make_dataset(from_dev_path,
                                                                to_dev_path,
                                                                en_vocab_path,
                                                                fr_vocab_path,
                                                                tokenizer=None,
                                                                force=FLAGS.force_make_data)
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        fr_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)
        model = create_model(sess, len(en_vocab), len(fr_vocab), True)

        accuracy = eval_model(sess, model, from_dev, to_dev, targets_dev)
        print("Per-utterance accuracy: %.2f" % accuracy)


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10,
                                           10,
                                           [(3, 3), (6, 6)],
                                           32,
                                           2,
                                           5.0,
                                           32,
                                           0.3,
                                           0.99,
                                           num_samples=8)
        sess.run(tf.global_variables_initializer())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    elif FLAGS.evaluate:
        evaluate()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
