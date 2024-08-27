import numpy as np
from twowave import angular_frequency


def test_angular_frequency():
    assert angular_frequency(9.8, 1) == np.sqrt(9.8)
    assert angular_frequency(9.8, 1, 1) == np.sqrt(9.8) + 1
