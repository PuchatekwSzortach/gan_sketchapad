"""
Module with data utilities
"""

import random
import typing

import numpy as np


class MnistDataLoader:
    """
    Simple data loader for mnist dataset, yields batches of images and labels
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> None:
        """
        Constructor

        Args:
            images (np.ndarray): mnist images
            labels (np.ndarray): mnist labels
            batch_size (int): batch size loader should use
            shuffle (bool): if True, data is shuffled before every epoch
        """

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples_indices = list(range(len(self.labels)))

    def __len__(self):

        return len(self.labels) // self.batch_size

    def __getitem__(self, index) -> typing.Tuple[np.ndarray, np.ndarray]:

        batch_start_index = index * self.batch_size
        batch_end_index = batch_start_index + self.batch_size

        samples_batch_indices = self.samples_indices[batch_start_index:batch_end_index]
        return self.images[samples_batch_indices], self.labels[samples_batch_indices]

    def __iter__(self) -> typing.Iterator[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Iterator, yields tuples (images, labels)
        """

        while True:

            if self.shuffle is True:
                random.shuffle(self.samples_indices)

            for batch_index in range(len(self)):

                yield self[batch_index]
