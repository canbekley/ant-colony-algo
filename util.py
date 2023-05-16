import numpy as np


def random_choice(arr):
    return np.random.choice(sorted(list(arr)), 1)[0]
