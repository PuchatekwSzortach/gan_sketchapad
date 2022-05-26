"""
Tests for net.processing module
"""

import numpy as np

import net.processing


def test_get_batched_data_generator():
    """
    Test get batched data generator
    """

    x_data = np.arange(12)
    y_data = x_data % 3
    batch_size = 4

    expected = [
        ([0, 1, 2, 3], [0, 1, 2, 0]),
        ([4, 5, 6, 7], [1, 2, 0, 1]),
        ([8, 9, 10, 11], [2, 0, 1, 2])
    ]

    actual = net.processing.get_batched_data_generator(
        x_data=x_data,
        y_data=y_data,
        batch_size=batch_size
    )

    assert np.all(expected == list(actual))
    # import icecream
    # # print()
    # # icecream.ic(expected)
    # # icecream.ic(list(actual))
