import os
import sys
from argparse import ArgumentParser
from copy import deepcopy

import pandas as pd
import matplotlib

from dialogue_denoiser_lstm import create_fake_timings

matplotlib.use('agg')

THIS_FILE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))

from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger
from deep_disfluency.utils.tools import convert_from_inc_disfluency_tags_to_eval_tags
from deep_disfluency.evaluation.disf_evaluation import incremental_output_disfluency_eval_from_file
from deep_disfluency.evaluation.disf_evaluation import final_output_disfluency_eval_from_file
from deep_disfluency.evaluation.eval_utils import rename_all_repairs_in_line_with_index

def configure_argument_parser():
    parser = ArgumentParser(description='Evaluate the deep_disfluency dialogue filter')
    parser.add_argument('dataset')
    parser.add_argument('mode', help='[deep_disfluency/babi]')

    return parser


def predict_babi_file_deep_disfluency(in_tagger, in_dataset, target_file_path=None):
    if target_file_path:
        target_file = open(target_file_path, "w")
    predictions_eval = []
    # RNN tags --> eval tags
    for utterance, tags, pos in zip(in_dataset['utterance'], in_dataset['tags'], in_dataset['pos']):
        current_tags = in_tagger.tag_utterance([(token, pos, tag)
                                                for token, pos, tag in zip(utterance, pos, tags)])
        predictions_eval += current_tags

    predictions_eval_iter = iter(predictions_eval)
    for speaker, (_, speaker_data) in enumerate(in_dataset.iterrows()):
        if target_file_path:
            target_file.write("Speaker: " + str(speaker) + "\n\n")
        timing_data, lex_data, pos_data, labels = (
        create_fake_timings(len(speaker_data['utterance'])),
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


def eval_babi(in_tagger,
              source_file_path,
              verbose=True):
    increco_file = 'swbd_disf_heldout_data_output_increco.text'
    dataset = pd.read_json(source_file_path)
    predict_babi_file_deep_disfluency(in_tagger, dataset, target_file_path=increco_file)
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
        current_tags_eval = convert_from_inc_disfluency_tags_to_eval_tags(d,
                                                                          b,
                                                                          representation="disf1")
        # if "asr" in division and not dialogue[:4] in good_asr: continue
        d = rename_all_repairs_in_line_with_index(list(current_tags_eval))
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
            print k, v
    all_results = deepcopy(results)

    return {'f1_<rm_word': all_results['f1_<rm_word'],
            'f1_<rps_word': all_results['f1_<rps_word'],
            'f1_<e_word': all_results['f1_<e_word']}


def main(in_dataset, in_mode):
    # Hough and Schlangen 2015 config
    disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/deep_disfluency/experiments/experiment_configs.csv",
        config_number=21,
        saved_model_dir="deep_disfluency/deep_disfluency/experiments/021/epoch_40"
    )
    for key, value in eval_babi(disf, in_dataset).iteritems():
        print '{}:\t{:.3f}'.format(key, value)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset, args.mode)
