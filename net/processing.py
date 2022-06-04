"""
Module with processing code
"""

import typing

import more_itertools
import numpy as np


def get_batched_data_generator(x_data: np.ndarray, y_data: np.ndarray, batch_size: int) -> typing.Generator:
    """
    Given x and y data, return a generator that yields batchs (x_data_batch), (y_data_batch)

    Args:
        x_data (np.ndarray): input data (e.g. images)
        y_data (np.ndarray): output data (e.g. labels)
        batch_size (int): batch size

    Returns:
        typing.Generator: generator that yields batches of data
    """

    x_data_batches_generator = more_itertools.chunked(x_data, n=batch_size)
    y_data_batches_generator = more_itertools.chunked(y_data, n=batch_size)

    return (np.array(x) for x in zip(x_data_batches_generator, y_data_batches_generator))
