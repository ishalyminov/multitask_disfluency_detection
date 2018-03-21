import argparse
import re
from operator import itemgetter
import sys

import nltk
import pandas as pd

from swda import swda
from swda_utils import filter_nonspeech, filter_edits


class DisfluencyTagger(object):
    """
        Tagging convention:
            'f' - fluent token (KEEP)
            'e' - edit token (DROP)
            'rm-[n]' - fluent token, replaces a token n tokens back (KEEP)
            'rp' - fluent token, part of repair, replaces a token sequentially after rm-[n]
    """
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.states_stack = ['FLUENT']

    def tag_utterance(self, in_utterance):
        self.reset_state()
        tags = []
        prev_token = None
        rm_deltas = []
        open_bracket_stack = []
        for token_index, token in enumerate(in_utterance.split()):
            # handling disfluency tags
            if token in ['{F', '{E', '{D', '{A', '{C']:
                open_bracket_stack.append('{')
                if self.states_stack[-1] == 'BEGIN_RR':
                    self.states_stack.append('INSIDE_RR_NONSENT')
                else:
                    self.states_stack.append('INSIDE_NONSENT')
                tags.append(None)
            elif token == '[':
                open_bracket_stack.append('[')
                self.states_stack.append('INSIDE_RM')
                tags.append(None)
                rm_deltas.append(0)
            elif token == '+':
                if self.states_stack[-1] in ['INSIDE_RM', 'INSIDE_RR', 'BEGIN_RR']:
                    self.states_stack[-1] = 'BEGIN_RR'
                    assert len(rm_deltas), \
                           in_utterance + ' ({} {} {})'.format(token_index, token, self.states_stack[-1])
                tags.append(None)
            elif token == '}':
                assert open_bracket_stack[-1].startswith('{'), in_utterance + ' ({} {})'.format(token_index, token)
                self.states_stack.pop()
                open_bracket_stack.pop()
                tags.append(None)
            elif token == ']':
                assert self.states_stack[-1] in ['BEGIN_RR', 'INSIDE_RR', 'INSIDE_RM'] and open_bracket_stack[-1] == '[', \
                       in_utterance + ' ({} {} {})'.format(token_index, token, self.states_stack[-1])
                open_bracket_stack.pop()
                if prev_token == '+':
                    self.states_stack[-1] = 'AFTER_EMPTY_RESTART'
                self.states_stack.pop()
                tags.append(None)
                rm_deltas.pop()

            # handling actual words
            else:
                rm_deltas = [delta + 1 for delta in rm_deltas]
                if token.endswith('-'):
                    tags.append('e')
                elif self.states_stack[-1] == 'FLUENT':
                    tags.append('f')
                elif self.states_stack[-1] == 'INSIDE_NONSENT':
                    tags.append('e')
                elif self.states_stack[-1] == 'INSIDE_RM':
                    tags.append('f')
                elif self.states_stack[-1] == 'INSIDE_RR_NONSENT':
                    tags.append('e')
                elif self.states_stack[-1] == 'BEGIN_RR':
                    assert len(rm_deltas), in_utterance + ' ({} {})'.format(token_index, token)
                    tags.append('rm-' + str(rm_deltas[-1] - 1))
                    self.states_stack[-1] = 'INSIDE_RR'
                elif self.states_stack[-1] == 'INSIDE_RR':
                    tags.append('rp')
                elif self.states_stack[-1] == 'AFTER_EMPTY_RESTART':
                    assert len(rm_deltas), in_utterance + ' ({} {})'.format(token_index, token)
                    tags.append('rm-' + str(rm_deltas[-1] - 1))
                    self.states_stack[-1] = 'FLUENT'
            prev_token = token
        return tags

    def is_bracket_structure_complete(self, in_utterance):
        bracket_stack = []
        for token in in_utterance.split():
            if token in ['{F', '{E', '{D', '{C', '{A']:
                bracket_stack.append('{')
            if token == '[':
                bracket_stack.append(token)
            if token == '}':
                if not len(bracket_stack) or bracket_stack[-1] != '{':
                    return False
                bracket_stack.pop()
            if token == ']':
                if not len(bracket_stack) or bracket_stack[-1] != '[':
                    return False
                bracket_stack.pop()
        return len(bracket_stack) == 0


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('swda_data_folder')
    result.add_argument('result_file')
    return result


def filter_utterance(in_utterance, in_tags):
    tokens = in_utterance.split()
    assert len(tokens) == len(in_tags), in_tags + ' | ' + ' '.join(tokens)
    tokens_tags_filtered = filter(lambda (x, y): y is not None, zip(tokens, in_tags))

    tokens_filtered = map(itemgetter(0), tokens_tags_filtered)
    tags_filtered = map(itemgetter(1), tokens_tags_filtered)

    real_tokens, real_tags = [], []
    tokenizer = nltk.tokenize.TweetTokenizer()
    for token, tag in zip(tokens_filtered, tags_filtered):
        tokenized = filter(lambda x: re.findall('\w+', x), tokenizer.tokenize(token))
        if 1 < len(tokenized):
            print >> sys.stderr, 'WARN:', token
            continue
        if not len(tokenized):
            continue
        real_tokens.append(tokenized[-1].lower())
        real_tags.append(tag)

    return real_tokens, real_tags


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    reader = swda.CorpusReader(args.swda_data_folder)
    tagger = DisfluencyTagger()

    utterances, tags = [], []

    for utterance in reader.iter_utterances():
        if not tagger.is_bracket_structure_complete(utterance.text):
            print 'Incomplete utterance - skipping: "{}"'.format(utterance.text)
            continue
        utterance_text = filter_nonspeech(utterance.text)
        utterance_text = filter_edits(utterance_text)
        if '/' in utterance_text:
            utterance_text = utterance_text[:utterance_text.index('/') + 1]
        tagging = tagger.tag_utterance(utterance_text)
        utterance_filtered, tags_filtered = filter_utterance(utterance_text, tagging)
        utterances.append(utterance_filtered)
        tags.append(tags_filtered)

    result = pd.DataFrame({'utterance': utterances, 'tags': tags})
    result.to_json(args.result_file)


if __name__ == '__main__':
    main()
