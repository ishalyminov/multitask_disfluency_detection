from argparse import ArgumentParser

from dialogue_denoiser_lstm import load, denoise_line


def configure_argument_parser():
    parser = ArgumentParser(description='Interactive LSTM dialogue filter')
    parser.add_argument('model_folder')

    return parser


def run(in_model_folder):
    model, vocab = load(in_model_folder)
    try:
        line = raw_input().strip()
        while line:
            print denoise_line(line, model, vocab)
            line = raw_input().strip()
    except EOFError as e:
        pass


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()

    run(args.model_folder)