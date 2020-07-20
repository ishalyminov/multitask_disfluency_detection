from copy import deepcopy

import torch
import pandas as pd

from deep_disfluency.evaluation.eval_utils import (get_tag_data_from_corpus_file,
                                                   rename_all_repairs_in_line_with_index)
from deep_disfluency.evaluation.disf_evaluation import (incremental_output_disfluency_eval_from_file,
                                                        final_output_disfluency_eval_from_file)
from deep_disfluency.utils.tools import (convert_from_eval_tags_to_inc_disfluency_tags,
                                         convert_from_inc_disfluency_tags_to_eval_tags)
from data_utils import make_multitask_dataset
from training_utils import batch_generator


def create_fake_timings(in_tokens_number):
    return list(zip(map(float, range(0, in_tokens_number)),
                    map(float, range(1, in_tokens_number + 1))))


def predict_increco_file(in_model,
                         vocabs_for_tasks,
                         source_file_path,
                         in_config,
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
        print("input file has timings")
        if not is_asr_results_file:
            dialogues = []
            IDs, timings, words, pos_tags, labels = \
                get_tag_data_from_corpus_file(source_file_path)
            for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
                dialogues.append((dialogue, (a, b, c, d)))
    else:
        print("no timings in input file, creating fake timings")
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
    predictions = predict_dataset(in_model, (X, ys_for_tasks))
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
    print('#broken sequences after RNN --> eval conversion: {} out of {}'.format(broken_sequences_number, len(utterances)))

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

                for t, w, p, tag in zip(new_timings, new_words, new_pos, predicted_tags):
                    target_file.write("\t".join([str(t[0]),
                                                 str(t[1]),
                                                 w,
                                                 p,
                                                 tag]))
                    target_file.write("\n")
                target_file.write("\n")
        target_file.write("\n")


def predict_babi_file(in_model,
                      vocabs_for_tasks,
                      dataset,
                      in_config,
                      target_file_path=None):
    if target_file_path:
        target_file = open(target_file_path, "w")

    # eval tags --> RNN tags
    X, ys_for_tasks = make_multitask_dataset(dataset,
                                             vocabs_for_tasks[0][0],
                                             vocabs_for_tasks[0][1],
                                             in_config)
    predictions = predict_dataset(in_model, (X, ys_for_tasks))
    predictions_eval = []
    global_word_index = 0
    broken_sequences_number = 0
    # RNN tags --> eval tags
    for utterance in dataset['utterance']:
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
    print('#broken sequences after RNN --> eval conversion: {} out of {}'.format(broken_sequences_number,
                                                                                 dataset.shape[0]))

    predictions_eval_iter = iter(predictions_eval)
    for speaker, (_, speaker_data) in enumerate(dataset.iterrows()):
        if target_file_path:
            target_file.write("Speaker: " + str(speaker) + "\n\n")
        timing_data, lex_data, pos_data, labels = (create_fake_timings(len(speaker_data['utterance'])),
                                                   speaker_data['utterance'],
                                                   speaker_data['pos'],
                                                   speaker_data['tags'])

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

                for t, w, p, tag in zip(new_timings, new_words, new_pos, predicted_tags):
                    target_file.write("\t".join([str(t[0]),
                                                 str(t[1]),
                                                 w,
                                                 p,
                                                 tag]))
                    target_file.write("\n")
                target_file.write("\n")
        target_file.write("\n")


def eval_deep_disfluency(in_model, in_vocabs_for_tasks, source_file_path, in_config, verbose=True):
    increco_file = 'swbd_disf_heldout_data_output_increco.text'
    predict_increco_file(in_model,
                         in_vocabs_for_tasks,
                         source_file_path,
                         in_config,
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
            print(k, v)
    all_results = deepcopy(results)

    return {'f1_<rm_word': all_results['f1_<rm_word'],
            'f1_<rps_word': all_results['f1_<rps_word'],
            'f1_<e_word': all_results['f1_<e_word']}


def eval_babi(in_model, in_vocabs_for_tasks, source_file_path, in_config, verbose=True):
    increco_file = 'swbd_disf_heldout_data_output_increco.text'
    dataset = pd.read_json(source_file_path)
    predict_babi_file(in_model,
                      in_vocabs_for_tasks,
                      dataset,
                      in_config,
                      target_file_path=increco_file)
    IDs, timings, words, pos_tags, labels = (map(str, range(dataset.shape[0])),
                                       [None] * dataset.shape[0],
                                       dataset['utterance'],
                                       dataset['pos'],
                                       dataset['tags']) 
    gold_data = {}  # map from the file name to the data
    for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        current_tags_eval = convert_from_inc_disfluency_tags_to_eval_tags(d,
                                                                          b,
                                                                          representation="disf1")
        gold_data[dialogue] = (create_fake_timings(len(b)), b, c, current_tags_eval)
    final_output_name = increco_file.replace("_increco", "_final")
    incremental_output_disfluency_eval_from_file(increco_file,
                                                 gold_data,
                                                 utt_eval=True,
                                                 error_analysis=False,
                                                 word=True,
                                                 interval=False,
                                                 outputfilename=final_output_name)
    # hyp_dir = experiment_dir
    IDs, timings, words, pos_tags, labels = (map(str, range(dataset.shape[0])),
                                       [None] * dataset.shape[0],
                                       dataset['utterance'],
                                       dataset['pos'],
                                       dataset['tags'])
    gold_data = {}  # map from the file name to the data
    for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        current_tags_eval = convert_from_inc_disfluency_tags_to_eval_tags(d,
                                                                          b,
                                                                          representation="disf1")
        d = rename_all_repairs_in_line_with_index(current_tags_eval)
        gold_data[dialogue] = (create_fake_timings(len(b)), b, c, d)

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
            print(k, v)
    all_results = deepcopy(results)

    return {'f1_<rm_word': all_results['f1_<rm_word'],
            'f1_<rps_word': all_results['f1_<rps_word'],
            'f1_<e_word': all_results['f1_<e_word']}


def filter_line(in_line, in_model, in_vocabs_for_tasks, in_config):
    tokens = in_line.lower().split()
    dataset = pd.DataFrame({'utterance': [tokens],
                            'tags': [['<f/>'] * len(tokens)],
                            'pos': [pos_tag(tokens)]})
    (tag_vocab, tag_label_vocab, tag_rev_label_vocab) = in_vocabs_for_tasks[0]
    X_line, ys_line = make_multitask_dataset(dataset, tag_vocab, tag_label_vocab, in_config)
    result_tokens = predict(in_model, (X_line, ys_line), in_vocabs_for_tasks, batch_size=1)
    return ' '.join(result_tokens)


def predict_dataset(in_model, in_dataset, batch_size=32):
    X_test, y_test_for_tasks = in_dataset

    batch_gen = batch_generator(X_test, y_test_for_tasks, batch_size)

    y_pred_main_task = []
    for batch_idx, (batch_x, batch_ys) in enumerate(batch_gen):
        batch_x = torch.LongTensor(batch_x)
        if torch.cuda.is_available():
            batch_x = batch_x.to(torch.device('cuda'))
        y_pred_main_task += in_model.predict(batch_x)[0]
    return y_pred_main_task
