import cv2
import numpy as np
import tensorflow as tf
import tqdm
import vlogging

import net.data
import net.utilities


# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

data_loader = net.data.MnistDataLoader(
    images=x_train.astype(np.float32).reshape(len(x_train), -1) / 256,
    labels=y_train.astype(np.float32),
    batch_size=1024,
    shuffle=True
)

# Optimizer
adam = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

generator = tf.keras.models.Sequential()

generator.add(tf.keras.layers.Dense(
    256, input_dim=randomDim,
    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))

generator.add(tf.keras.layers.LeakyReLU(0.2))
generator.add(tf.keras.layers.Dense(512))
generator.add(tf.keras.layers.LeakyReLU(0.2))
generator.add(tf.keras.layers.Dense(1024))
generator.add(tf.keras.layers.LeakyReLU(0.2))
generator.add(tf.keras.layers.Dense(784, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = tf.keras.models.Sequential()

discriminator.add(tf.keras.layers.Dense(
    1024, input_dim=784, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))

discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.3))
discriminator.add(tf.keras.layers.Dense(512))
discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.3))
discriminator.add(tf.keras.layers.Dense(256))
discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.3))
discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = tf.keras.layers.Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = tf.keras.models.Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)


def train(epochs=1, batchSize=128):
    """
    Train model
    """

    logger = net.utilities.get_logger("/tmp/reference_gan.html")

    for epoch in range(1, epochs+1):

        print(f'Epoch: {epoch}')

        dLosses = []
        gLosses = []

        for index in tqdm.tqdm(range(len(data_loader))):

            ground_truth_images, _ = data_loader[index]
            imageBatch = ground_truth_images

            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[len(ground_truth_images), randomDim])

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

            dLosses.append(dloss)
            gLosses.append(gloss)

        if epoch == 1 or epoch % 10 == 0:

            logger.info(f"<h2>Epoch {epoch}</h2><br>")
            logger.info(f"generator loss: {np.mean(gLosses):.3f} <br>")
            logger.info(f"discriminator loss: {np.mean(dLosses):.3f} <br>")

            noise = np.random.normal(0, 1, size=[10, randomDim])
            generatedImages = (255 * generator.predict(noise)).reshape((-1, 28, 28)).astype(np.int32)

            logger.info(vlogging.VisualRecord(
                "generated images",
                imgs=list(generatedImages)
            ))

if __name__ == '__main__':
    train(200, 1024)
