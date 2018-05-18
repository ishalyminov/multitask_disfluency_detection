import os
import sys
from argparse import ArgumentParser

import pandas as pd
import matplotlib
matplotlib.use('agg')

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
        config_file="deep_disfluency/deep_disfluency/experiments/experiment_configs.csv",
        config_number=21,
        saved_model_dir="deep_disfluency/deep_disfluency/experiments/021/epoch_40"
    )
    for idx, row in in_dataset.iterrows():
        print ' '.join(row['utterance'])
        tagger_input = [(word, pos, None) for word, pos in zip(row['utterance'], row['pos'])]
        import pdb; pdb.set_trace()
        print ' '.join(disf.tag_utterance(tagger_input))


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(pd.read_json(args.dataset_file))
