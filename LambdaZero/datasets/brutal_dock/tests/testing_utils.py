import string

import numpy as np


def generate_random_string(string_length):
    letters = [letter for letter in string.ascii_uppercase]
    s = np.random.choice(letters, string_length)
    return ''.join(s)
