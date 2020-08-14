import string

import numpy as np
import pytest

from LambdaZero.examples.env3d.wandb_logger import recursively_remove_classes_and_functions_from_dictionary


def bogus_function(x):
    return x


class BogusClass(object):

    def __init__(self):
        self.string = 'this is just a test class'


def random_string():
    return ''.join(np.random.choice(list(string.ascii_letters), 8))


def get_random_dictionaries(width):

    keys = [random_string() for _ in range(width)]

    original = {}
    clean = {}

    for key in keys:
        value = np.random.choice([random_string(), bogus_function, BogusClass])
        original[key] = value
        if type(value) == str:
            clean[key] = value

    return original, clean


@pytest.fixture
def original_and_clean_dicts():

    np.random.seed(1231)

    original_dict = {}
    clean_dict = {}

    maximum_with = 10

    width1 = np.random.randint(1,  maximum_with)
    width2 = np.random.randint(1,  maximum_with)
    width3 = np.random.randint(1,  maximum_with)

    for i1 in range(width1):
        original_lvl2 = {}
        clean_lvl2 = {}
        for i2 in range(width2):
            key = random_string()
            original_lv3, clean_lvl3 = get_random_dictionaries(width3)
            original_lvl2[key] = original_lv3
            clean_lvl2[key] = clean_lvl3

        key = random_string()
        original_dict[key] = original_lvl2
        clean_dict[key] = clean_lvl2

    return original_dict, clean_dict


def test_recursively_remove_classes_and_functions_from_dictionary(original_and_clean_dicts):
    original_dict, expected_clean_dict = original_and_clean_dicts
    computed_clean_dict = recursively_remove_classes_and_functions_from_dictionary(original_dict)

    assert computed_clean_dict == expected_clean_dict


