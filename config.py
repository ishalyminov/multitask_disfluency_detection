import json
import os

THIS_FILE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_FILE = os.path.join(THIS_FILE_DIR, 'config.json')


def read_config(in_filename=DEFAULT_CONFIG_FILE):
    with open(in_filename) as config_in:
        config = json.load(config_in)
    return config
