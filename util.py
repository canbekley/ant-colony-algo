import numpy as np


def random_selection(arr, *args, **kwargs):
    """Returns a random element of a sorted array."""
    return np.random.choice(sorted(list(arr)), 1)[0]


def fake_random_selection(arr, *args, **kwargs):
    """Always returns the first element of a sorted array."""
    return sorted(list(arr))[0]
