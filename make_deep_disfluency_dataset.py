import sys
import os
from argparse import ArgumentParser

import pandas as pd

from deep_disfluency_utils import load_dataset

THIS_FILE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_FILE_DIR,
                        'deep_disfluency',
                        'deep_disfluency',
                        'data',
                        'disfluency_detection',
                        'switchboard')

sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))


def deep_disfluency_dataset_to_data_frame(in_dataset):
    return pd.DataFrame({'utterance': in_dataset[2],
                         'pos': in_dataset[3],
                         'tags': in_dataset[4]})


def get_unique_elements(in_list):
    result = set([])
    for sequence in in_list:
        result.update(sequence)
    return result


def main(in_result_folder, in_format):
    if in_format == 'disfluency':
        trainset, devset, testset = (os.path.join(DATA_DIR, 'swbd_disf_train_1_data.csv'),
                                     os.path.join(DATA_DIR, 'swbd_disf_heldout_data.csv'),
                                     os.path.join(DATA_DIR, 'swbd_disf_test_data.csv'))
    elif in_format == 'timings':
        trainset, devset, testset = (os.path.join(DATA_DIR, 'swbd_disf_train_1_data_timings.csv'),
                                     os.path.join(DATA_DIR, 'swbd_disf_heldout_data_timings.csv'),
                                     os.path.join(DATA_DIR, 'swbd_disf_test_data_timings.csv'))
    else:
        raise NotImplementedError
    train = load_dataset(trainset, convert_to_dnn_format=True)
    dev = load_dataset(devset, convert_to_dnn_format=True)
    test = load_dataset(testset, convert_to_dnn_format=True)
    print 'Trainset size: {} utterances'.format(len(train[2]))
    print 'Devset size: {} utterances'.format(len(dev[2]))
    print 'Testset size: {} utterances'.format(len(test[2]))
    unique_train_tags = get_unique_elements(train[4])
    print 'Unique #tags in trainset: {}'.format(len(unique_train_tags))

    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)
    (deep_disfluency_dataset_to_data_frame(train)).to_json(os.path.join(in_result_folder, 'trainset.json'))
    (deep_disfluency_dataset_to_data_frame(dev)).to_json(os.path.join(in_result_folder, 'devset.json'))
    (deep_disfluency_dataset_to_data_frame(test)).to_json(os.path.join(in_result_folder, 'testset.json'))


def configure_argument_parser():
    parser = ArgumentParser(description='Make dataset from deep_disfluency (Hough, Schlangen 2015)')
    parser.add_argument('result_folder')
    parser.add_argument('format', help='[disfluency/timings]')

    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.result_folder, args.format)
