"""
Commands with training code
"""

import invoke


@invoke.task
def train_mnist_gan(_context, config_path):
    """
    Train a simple GAN on MNIST dataset

    Args:
        _context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import net.utilities

    net.utilities.read_yaml(config_path)
