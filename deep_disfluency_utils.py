from deep_disfluency.feature_extraction.feature_utils import load_data_from_disfluency_corpus_file
from deep_disfluency.evaluation.disf_evaluation import get_tag_data_from_corpus_file
from deep_disfluency.utils.tools import convert_from_eval_tags_to_inc_disfluency_tags


def get_tag_mapping(in_tag_map, mode='deep_disfluency'):
    if mode == 'deep_disfluency':
        grouped_tag_map = {'e': filter(lambda x: x.startswith('<e'), in_tag_map.keys()),
                           'rm': filter(lambda x: x.startswith('<rm'), in_tag_map.keys())}
        result = {name: map(in_tag_map.get, tags) for name, tags in grouped_tag_map.iteritems()}
    else:
        result = {name: [idx] for name, idx in in_tag_map.iteritems()}
    return result


def load_dataset(in_filename, convert_to_dnn_format=True):
    if 'timings' in in_filename:
        dialogues = []
        IDs, timings, words, pos_tags, labels = get_tag_data_from_corpus_file(in_filename)
        for dialogue, a, b, c, d in zip(IDs,
                                        timings,
                                        words,
                                        pos_tags,
                                        labels):
            dialogues.append((dialogue, (a, b, c, d)))
        ids, timings, utterances, tags, pos = [], [], [], [], []
        for speaker, speaker_data in dialogues:
            timing_data, lex_data, pos_data, labels = speaker_data

            for i in range(0, len(timing_data)):
                _, end = timing_data[i]
                if "<t" in labels[i]:
                    timings.append([])
                    utterances.append([])
                    tags.append([])
                    pos.append([])
                ids.append('')
                timings[-1].append(end)
                utterances[-1].append(lex_data[i])
                tags[-1].append(labels[i])
                pos[-1].append(pos_data[i])
        if convert_to_dnn_format:
            tags = [convert_from_eval_tags_to_inc_disfluency_tags(tags_i,
                                                                  words_i,
                                                                  representation="disf1")
                    for tags_i, words_i in zip(tags, utterances)]
        return ids, timings, utterances, pos, tags
    else:
        return load_data_from_disfluency_corpus_file(in_filename,
                                                     representation='disf1',
                                                     limit=8,
                                                     convert_to_dnn_format=convert_to_dnn_format)