"""
Module with machine learning related code
"""

import logging

import numpy as np
import tensorflow as tf
import tqdm
import vlogging

import net.data


class GANContainer:
    """
    Class encapsulating a simple GAN model.
    Intended for use with MNIST dataset.
    """

    def __init__(self, noise_input_shape, generated_output_shape) -> None:

        self.noise_input_shape = noise_input_shape

        self.generator_model = self._build_generator_model(
            noise_input_shape=noise_input_shape,
            generated_output_shape=generated_output_shape
        )

        self.discriminator_model = self._build_discriminator_model(
            input_shape=generated_output_shape,
            output_shape=1
        )

        gan_input = tf.keras.layers.Input(shape=noise_input_shape)

        self.gan_model = tf.keras.models.Model(
            inputs=gan_input,
            outputs=self.discriminator_model(self.generator_model(gan_input))
        )

        self.gan_model.compile(loss="binary_crossentropy", optimizer="adam")

    def _build_generator_model(self, noise_input_shape, generated_output_shape):

        generator_input = tf.keras.layers.Input(shape=noise_input_shape)

        x = tf.keras.layers.Dense(256, activation="swish")(generator_input)
        x = tf.keras.layers.Dense(512, activation="swish")(x)
        x = tf.keras.layers.Dense(1024, activation="swish")(x)
        x = tf.keras.layers.Dense(generated_output_shape, activation="sigmoid")(x)

        generator_model = tf.keras.models.Model(inputs=generator_input, outputs=x)
        generator_model.compile(loss="binary_crossentropy", optimizer="adam")

        return generator_model

    def _build_discriminator_model(self, input_shape, output_shape):

        discriminator_input = tf.keras.layers.Input(shape=input_shape)

        x = tf.keras.layers.Dense(1024, activation="swish")(discriminator_input)
        x = tf.keras.layers.Dense(512, activation="swish")(x)
        x = tf.keras.layers.Dense(128, activation="swish")(x)
        x = tf.keras.layers.Dense(output_shape, activation="sigmoid")(x)

        discriminator_model = tf.keras.models.Model(inputs=discriminator_input, outputs=x)
        discriminator_model.compile(loss="binary_crossentropy", optimizer="adam")

        return discriminator_model


class GanTrainingManager:
    """
    Class for encapsulaing GAN trainig
    """

    def __init__(
            self, gan_container: GANContainer, data_loader: net.data.MnistDataLoader,
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

    def train(self):
        """
        Train GAN
        """

        for epoch in range(self.epochs):

            generator_losses = []
            discriminator_losses = []

            print(f"Epoch {epoch}")

            for index in tqdm.tqdm(range(len(self.data_loader))):

                ground_truth_images, _ = self.data_loader[index]

                noise = np.random.normal(0, 1, size=[len(ground_truth_images), self.gan_container.noise_input_shape])

                generated_images = self.gan_container.generator_model.predict(noise)

                # Combine real and generated images
                combined_images = np.concatenate(
                    [ground_truth_images.reshape(len(ground_truth_images), -1), generated_images]
                )

                # Create labels for discriminator
                # Make truth labels for real images "smooth" by setting them to 0.9 instead of 1
                discriminator_labels = np.zeros(len(combined_images), dtype=np.float32)
                discriminator_labels[:len(ground_truth_images)] = 0.9

                # Train discriminator
                self.gan_container.discriminator_model.trainable = True

                discriminator_loss = self.gan_container.discriminator_model.train_on_batch(
                    combined_images, discriminator_labels)

                self.gan_container.discriminator_model.trainable = False

                noise = np.random.normal(0, 1, size=[len(ground_truth_images), self.gan_container.noise_input_shape])
                generator_labels = np.ones(len(ground_truth_images), dtype=np.float32)

                # Train generator
                generator_loss = self.gan_container.gan_model.train_on_batch(noise, generator_labels)

                discriminator_losses.append(discriminator_loss)
                generator_losses.append(generator_loss)

            if epoch % 10 == 0:

                self.logger.info(f"<h2>Epoch {epoch}</h2><br>")
                self.logger.info(f"generator loss: {np.mean(generator_losses):.3f} <br>")
                self.logger.info(f"discriminator loss: {np.mean(discriminator_losses):.3f} <br>")

                noise = np.random.normal(0, 1, size=[10, self.gan_container.noise_input_shape])
                generated_images_vectors = self.gan_container.generator_model.predict(noise)

                generated_images = (255 * generated_images_vectors.reshape((-1, 28, 28))).astype(np.int32)

                self.logger.info(vlogging.VisualRecord(
                    title="generated images",
                    imgs=list(generated_images)
                ))
