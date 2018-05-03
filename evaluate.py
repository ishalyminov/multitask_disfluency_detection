from argparse import ArgumentParser
from copy import deepcopy

import tensorflow as tf
import pandas as pd

from deep_disfluency.evaluation.disf_evaluation import incremental_output_disfluency_eval_from_file
from deep_disfluency.evaluation.disf_evaluation import final_output_disfluency_eval_from_file
from deep_disfluency.evaluation.eval_utils import get_tag_data_from_corpus_file
from deep_disfluency.evaluation.eval_utils import rename_all_repairs_in_line_with_index
from deep_disfluency.evaluation.results_utils import convert_to_latex
from dialogue_denoiser_lstm import load, predict_increco_file

DEFAULT_HELDOUT_DATASET = 'deep_disfluency/deep_disfluency/data/disfluency_detection/switchboard/swbd_disf_heldout_data_timings.csv'


def configure_argument_parser():
    parser = ArgumentParser(description='Evaluate the LSTM dialogue filter')
    parser.add_argument('model_folder')
    parser.add_argument('dataset', default=DEFAULT_HELDOUT_DATASET)

    return parser


def eval_deep_disfluency(in_model,
                         in_vocab,
                         in_label_vocab,
                         in_rev_label_vocab,
                         source_file_path,
                         in_session):
    increco_file = 'swbd_disf_heldout_data_output_increco.text'
    predict_increco_file(in_model,
                         in_vocab,
                         in_label_vocab,
                         in_rev_label_vocab,
                         source_file_path,
                         in_session,
                         target_file_path=increco_file)
    IDs, timings, words, pos_tags, labels = get_tag_data_from_corpus_file(source_file_path)
    gold_data = {}  # map from the file name to the data
    for dialogue, a, b, c, d in zip(IDs, timings, words, pos_tags, labels):
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        gold_data[dialogue] = (a, b, c, d)
    final_output_name = increco_file.replace("_increco", "_final")
    incremental_output_disfluency_eval_from_file(
        increco_file,
        gold_data,
        utt_eval=True,
        error_analysis=True,
        word=True,
        interval=False,
        outputfilename=final_output_name)

    all_results = {}
    all_error_dicts = {}

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
    for k, v in results.items():
        print k, v
    all_results = deepcopy(results)
    display_results = dict()
    display_results['RNN (window length=2) (+ POS)'] = all_results['test_021/epoch_40']
    display_results['LSTM (window length=2) (+ POS)'] = all_results['test_041/epoch_16']
    final = convert_to_latex(display_results, eval_level=['word'], inc=False, utt_seg=False,
                             only_include=['f1_<rm_word', 'f1_<rps_word', 'f1_<e_word'])


def main(in_dataset_file, in_model_folder):
    with tf.Session() as sess:
        model, vocab, char_vocab, label_vocab, eval_label_vocab = load(in_model_folder, sess)
        rev_label_vocab = {label_id: label
                           for label, label_id in label_vocab.iteritems()}


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset), args.model_folder)
