import argparse

import pandas as pd


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('tagged_dataset')
    result.add_argument('result_file')
    return result


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    dataset = pd.read_json(args.tagged_dataset)
    fluent_turns = dataset[dataset.apply(lambda x: set(x['tags']) == {'f'}, axis=1)]
    disfluent_turns = dataset[dataset.apply(lambda x: set(x['tags']) != {'f'}, axis=1)]
    fluent_turns_sample = fluent_turns.sample(n=int(disfluent_turns.shape[0] / 9), replace=False)
    result = pd.concat([disfluent_turns, fluent_turns_sample], axis=0)
    result.to_json(args.result_file)


if __name__ == '__main__':
    main()