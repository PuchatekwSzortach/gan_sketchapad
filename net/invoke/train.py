"""
Commands with training code
"""

import invoke


@invoke.task
def train_mnist_gan(_context, config_path):
    """
    Train a simple GAN on MNIST dataset.

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

    data_loader = net.data.MnistDataLoader(
        images=np.expand_dims(x_train.astype(np.float32) / 256, axis=-1),
        labels=y_train.astype(np.float32),
        batch_size=1024,
        shuffle=True
    )

    net.ml.GanTrainingManager(
        gan_container=net.ml.MNISTGANContainer(noise_input_size=100),
        data_loader=data_loader,
        epochs=config["epochs"],
        logger=net.utilities.get_logger(config["logger_path"])
    ).train()


@invoke.task
def train_mnist_conditional_gan(_context, config_path):
    """
    Train a simple conditional GAN on MNIST dataset.

    Args:
        _context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import numpy as np
    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_combined = np.concatenate([x_train, x_test])
    y_combined = np.concatenate([y_train, y_test])

    batch_size = 512
    categories_count = 10

    data_loader = net.data.MnistDataLoader(
        images=np.expand_dims(x_combined.astype(np.float32) / 255, axis=-1),
        labels=y_combined.astype(np.float32),
        batch_size=batch_size,
        shuffle=True
    )

    config = net.utilities.read_yaml(config_path)

    net.ml.ConditinalGanTrainingManager(
        gan_container=net.ml.MINSTConditionalGanContainer(
            noise_input_size=100,
            categories_count=categories_count),
        data_loader=data_loader,
        epochs=config["epochs"],
        logger=net.utilities.get_logger(config["logger_path"])
    ).train()
