def get_tag_mapping(in_tag_map, mode='deep_disfluency'):
    if mode == 'deep_disfluency':
        grouped_tag_map = {'e': filter(lambda x: x.startswith('<e'), in_tag_map.keys()),
                           'rm': filter(lambda x: x.startswith('<rm'), in_tag_map.keys())}
        result = {name: map(in_tag_map.get, tags) for name, tags in grouped_tag_map.iteritems()}
    else:
        result = {name: [idx] for name, idx in in_tag_map.iteritems()}
    return result
