import sys
import os
from argparse import ArgumentParser

import pandas as pd

THIS_FILE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_FILE_DIR,
                        'deep_disfluency',
                        'deep_disfluency',
                        'data',
                        'disfluency_detection',
                        'swbd_Interspeech_2015',
                        'switchboard')

sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))

from deep_disfluency.feature_extraction.feature_utils import load_data_from_disfluency_corpus_file_no_timings


def deep_disfluency_dataset_to_data_frame(in_dataset):
    return pd.DataFrame({'utterance': in_dataset[1],
                         'pos_tags': in_dataset[2],
                         'labels': in_dataset[3]})


def main(in_result_folder):
    train = load_data_from_disfluency_corpus_file_no_timings(os.path.join(DATA_DIR, 'swbd_train_data.csv'),
                                                             limit=8,
                                                             convert_to_dnn_format=True)
    dev = load_data_from_disfluency_corpus_file_no_timings(os.path.join(DATA_DIR, 'swbd_heldout_data.csv'),
                                                           limit=8,
                                                           convert_to_dnn_format=True)
    test = load_data_from_disfluency_corpus_file_no_timings(os.path.join(DATA_DIR, 'swbd_test_data.csv'),
                                                            limit=8,
                                                            convert_to_dnn_format=True)
    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)
    (deep_disfluency_dataset_to_data_frame(train)).to_json(os.path.join(in_result_folder, 'trainset.json'))
    (deep_disfluency_dataset_to_data_frame(dev)).to_json(os.path.join(in_result_folder, 'devset.json'))
    (deep_disfluency_dataset_to_data_frame(test)).to_json(os.path.join(in_result_folder, 'testset.json'))


def configure_argument_parser():
    parser = ArgumentParser(description='Make dataset from deep_disfluency (Hough, Schlangen 2015)')
    parser.add_argument('result_folder')

    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.result_folder)