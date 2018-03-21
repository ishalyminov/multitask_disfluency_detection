import re

NONSENTENCE_RE = '({(\w) ([^{]+?)})'


def extract_nonspeech(in_utterance):
    return map(lambda x: ('nonspeech', x), re.findall('<+[^<]+?>+', in_utterance))


def filter_nonspeech(in_utterance):
    return re.sub('<+[^<]+?>+', '', in_utterance).strip()

def filter_edits(in_utterance):
    return re.sub('\[+[^\[]+?\]+', '', in_utterance).strip()


def extract_nonsentence(in_utterance):
    return map(lambda (body, disfluency_type, text): ('{{{}...}}'.format(disfluency_type), (body, text)),
               re.findall(NONSENTENCE_RE, in_utterance))


def filter_nonsentence(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for disfluency_type, (body, text) in extract_nonsentence(in_utterance):
        disfluent = disfluent.replace(body, text)
        clean = clean.replace(body, '')
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


def extract_restarts_with_repair_and_nonsentence(in_utterance):
    return map(lambda x: ('[RM + {} RR]', x),
               re.findall('(\[([^[]+?)\s+\+\s+{}\s+([^[]+?)\])'.format(NONSENTENCE_RE), in_utterance))


def filter_restarts_with_repair_and_nonsentence(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm, nonsent_body, nonsent_type, nonsent_text, rr) in extract_restarts_with_repair_and_nonsentence(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm, nonsent_text, rr]))
        clean = clean.replace(body, rr)
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


def extract_restarts_with_repair(in_utterance):
    return map(lambda x: ('[RM + RR]', x),
               re.findall('(\[([^[]+?)\s+\+\s+([^[{]+?)\])', in_utterance))


def filter_restarts_with_repair(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm, rr) in extract_restarts_with_repair(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm, rr]))
        clean = clean.replace(body, rr)
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)


def extract_restarts_without_repair(in_utterance):
    return map(lambda x: ('[RM +]', x),
               re.findall('(\[([^[]+?)\s+\+\s+\])', in_utterance))


def filter_restarts_without_repair(in_utterance):
    disfluent, clean = in_utterance, in_utterance
    for repair_type, (body, rm) in extract_restarts_without_repair(in_utterance):
        disfluent = disfluent.replace(body, ' '.join([rm]))
        clean = clean.replace(body, '')
    return re.sub('\s+', ' ', disfluent), re.sub('\s+', ' ', clean)
