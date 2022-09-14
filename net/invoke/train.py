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


@invoke.task
def train_keras_mnist_conditional_gan(_context, config_path):
    """
    Train a simple conditional GAN on MNIST dataset.
    Model is implemented using keras interafce

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

    data_loader = net.data.MnistDataLoader(
        images=np.expand_dims(x_combined.astype(np.float32) / 255, axis=-1),
        labels=y_combined.astype(np.float32),
        batch_size=batch_size,
        shuffle=True
    )

    dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(data_loader),
        output_types=(
            tf.float32,
            tf.float32),
        output_shapes=(
            tf.TensorShape([None, 28, 28, 1]),
            tf.TensorShape([None])
        )
    ).prefetch(32)

    config = net.utilities.read_yaml(config_path)

    model = net.ml.KerasBasedMINSTConditionalGanModel(
        noise_input_size=100,
        categories_count=10,
        batch_size=batch_size
    )

    model.compile(
        generator_optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
        discriminator_optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
        loss_function=tf.keras.losses.BinaryCrossentropy()
    )

    model.fit(
        x=dataset,
        steps_per_epoch=len(data_loader),
        epochs=config["epochs"],
        callbacks=[net.ml.ConditionalGanCallback(
            logger=net.utilities.get_logger(config["logger_path"])
        )]
    )
