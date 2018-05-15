import os
import sys
from argparse import ArgumentParser

import pandas as pd

THIS_FILE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(THIS_FILE_DIR, 'deep_disfluency'))

from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


def configure_argument_parser():
    parser = ArgumentParser(description='Train LSTM dialogue filter')
    parser.add_argument('dataset_file')

    return parser


def main(in_dataset):
    # Hough and Schlangen 2015 config
    disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/experiments/experiment_configs.csv",
        config_number=21,
        saved_model_dir="deep_disfluency/experiments/021/epoch_40"
    )
    for idx, row in in_dataset.iterrows():
        print row['utterance']
        print disf.tag_utterance(row['utterance'])


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset_file))