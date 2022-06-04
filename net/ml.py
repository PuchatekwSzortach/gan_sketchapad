"""
Module with machine learning related code
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
import vlogging

import net.data


class MNISTGANContainer:
    """
    Class encapsulating a simple GAN model.
    Intended for use with MNIST dataset.
    """

    def __init__(self, noise_input_size) -> None:

        self.noise_input_shape = noise_input_size

        optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        self.generator_model = self._build_generator_model(
            noise_input_size=noise_input_size
        )

        self.discriminator_model = self._build_discriminator_model(
            output_shape=1,
            optimizer=optimizer
        )

        gan_input = tf.keras.layers.Input(shape=noise_input_size)

        self.gan_model = tf.keras.models.Model(
            inputs=gan_input,
            outputs=self.discriminator_model(self.generator_model(gan_input))
        )

        self.gan_model.compile(loss="binary_crossentropy", optimizer=optimizer)

    def _build_generator_model(self, noise_input_size):

        generator_input = tf.keras.layers.Input(shape=noise_input_size)

        # Foundation for a 7x7 image
        base_nodes_count = 128 * 7 * 7
        x = tf.keras.layers.Dense(base_nodes_count, activation="swish")(generator_input)

        # Reshape dense layer to convolutional layer
        x = tf.keras.layers.Reshape((7, 7, 128))(x)

        # upsample to 14x14, then 28x28
        x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation="swish")(x)
        x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation="swish")(x)

        # And a small convolution for good measure
        x = tf.keras.layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

        generator_model = tf.keras.models.Model(inputs=generator_input, outputs=x)

        return generator_model

    def _build_discriminator_model(self, output_shape, optimizer):

        discriminator_input = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Flatten()(discriminator_input)

        x = tf.keras.layers.Dense(1024, activation="swish")(x)
        x = tf.keras.layers.Dense(512, activation="swish")(x)
        x = tf.keras.layers.Dense(128, activation="swish")(x)
        x = tf.keras.layers.Dense(output_shape, activation="sigmoid")(x)

        discriminator_model = tf.keras.models.Model(inputs=discriminator_input, outputs=x)
        discriminator_model.compile(loss="binary_crossentropy", optimizer=optimizer)

        return discriminator_model


class GanTrainingManager:
    """
    Class for encapsulaing GAN trainig
    """

    def __init__(
            self, gan_container: MNISTGANContainer, data_loader: net.data.MnistDataLoader,
            epochs: int, logger: logging.Logger):
        """
        Constructor

        Args:
            gan_container (GANContainer): container for gan model to be trained
            data_loader (net.data.MnistDataLoader): data to train model on
            epochs (int): number of epochs to train
            logger (logging.Logger): logger instance
        """

        self.gan_container = gan_container
        self.data_loader = data_loader
        self.epochs = epochs
        self.logger = logger

    def log_results(self, epoch: int, generator_losses: list, discriminator_losses: list):
        """
        Log results
        """

        if epoch % 10 == 0:

            self.logger.info(f"<h2>Epoch {epoch}</h2><br>")

            noise = np.random.normal(0, 1, size=[10, self.gan_container.noise_input_shape])
            generated_images_vectors = self.gan_container.generator_model.predict(noise)

            generated_images = (255 * generated_images_vectors.reshape((-1, 28, 28))).astype(np.int32)

            self.logger.info(vlogging.VisualRecord(
                title="generated images",
                imgs=list(generated_images)
            ))

        if epoch > 0 and epoch % 10 == 0:

            figure = plt.figure()

            x_range = list(range(len(generator_losses)))

            plt.plot(x_range, discriminator_losses, label="discriminator")
            plt.plot(x_range, generator_losses, label="generator")

            plt.legend()

            self.logger.info(vlogging.VisualRecord(
                title="losses",
                imgs=figure
            ))

    def train(self):
        """
        Train GAN
        """

        average_epochs_losses = {
            "generator_losses": [],
            "discriminator_losses": []
        }

        for epoch in range(self.epochs):

            epoch_losses = {
                "generator_losses": [],
                "discriminator_losses": []
            }

            print(f"Epoch {epoch}")

            for index in tqdm.tqdm(range(len(self.data_loader))):

                ground_truth_images, _ = self.data_loader[index]

                noise = np.random.normal(0, 1, size=[len(ground_truth_images), self.gan_container.noise_input_shape])

                generated_images = self.gan_container.generator_model.predict(noise)

                # Combine real and generated images
                combined_images = np.concatenate([ground_truth_images, generated_images])

                # Create labels for discriminator
                # Make truth labels for real images "smooth" by setting them to 0.9 instead of 1
                discriminator_labels = np.zeros(len(combined_images), dtype=np.float32)
                discriminator_labels[:len(ground_truth_images)] = 0.9

                # Train discriminator
                self.gan_container.discriminator_model.trainable = True

                discriminator_loss = self.gan_container.discriminator_model.train_on_batch(
                    combined_images, discriminator_labels)

                self.gan_container.discriminator_model.trainable = False

                # Train generator
                noise = np.random.normal(0, 1, size=[len(ground_truth_images), self.gan_container.noise_input_shape])

                # Set labels to one, as we want to train generator to fool discriminator into predicting that
                # generated images are real
                generator_labels = np.ones(len(ground_truth_images), dtype=np.float32)

                generator_loss = self.gan_container.gan_model.train_on_batch(noise, generator_labels)

                epoch_losses["generator_losses"].append(generator_loss)
                epoch_losses["discriminator_losses"].append(discriminator_loss)

            average_epochs_losses["generator_losses"].append(np.mean(epoch_losses["generator_losses"]))
            average_epochs_losses["discriminator_losses"].append(np.mean(epoch_losses["discriminator_losses"]))

            self.log_results(
                epoch=epoch,
                generator_losses=average_epochs_losses["generator_losses"],
                discriminator_losses=average_epochs_losses["discriminator_losses"]
            )
