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
    # import net.ml
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

    print(config)
    print(data_loader)

    # net.ml.GanTrainingManager(
    #     gan_container=net.ml.MNISTGANContainer(noise_input_size=100),
    #     data_loader=data_loader,
    #     epochs=config["epochs"],
    #     logger=net.utilities.get_logger(config["logger_path"])
    # ).train()
