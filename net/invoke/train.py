"""
Commands with training code
"""

import invoke


@invoke.task
def train_mnist_gan(_context, config_path):
    """
    Train a simple GAN on MNIST dataset.
    Based on https://github.com/Zackory/Keras-MNIST-GAN/

    Args:
        _context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import numpy as np
    import tensorflow as tf

    import net.data
    import net.ml
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    gan_container = net.ml.GANContainer(
        noise_input_shape=10,
        generated_output_shape=784
    )

    data_loader = net.data.MnistDataLoader(
        images=x_train.astype(np.float32).reshape(len(x_train), -1) / 256,
        labels=y_train.astype(np.float32),
        batch_size=1024,
        shuffle=True
    )

    net.ml.GanTrainingManager(
        gan_container=gan_container,
        data_loader=data_loader,
        epochs=config["epochs"],
        logger=net.utilities.get_logger(config["logger_path"])
    ).train()
