import os
from argparse import ArgumentParser
from operator import itemgetter

import pandas as pd
from nltk import CRFTagger

THIS_FILE_DIR = os.path.dirname(__file__)
DEEP_DISFLUENCY_FOLDER = os.path.join(THIS_FILE_DIR, 'deep_disfluency')
TAGGER_PATH = os.path.join(DEEP_DISFLUENCY_FOLDER, 'deep_disfluency/feature_extraction/crfpostagger')

POS_TAGGER = CRFTagger()
POS_TAGGER.set_model_file(TAGGER_PATH)


def configure_argument_parser():
    parser = ArgumentParser(description='POS tag dataset')
    parser.add_argument('dataset')
    parser.add_argument('result_file')

    return parser


def main(in_src_file, in_result_file):
    dataset = pd.read_json(in_src_file)
    pos = []
    for utterance in dataset['utterance']:
        pos_i = POS_TAGGER.tag(utterance)
        pos.append(map(itemgetter(1), pos_i))
    dataset['pos'] = pos
    dataset.reset_index(drop=True).to_json(in_result_file)


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    main(args.dataset, args.result_file)
