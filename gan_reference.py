import cv2
import numpy as np
import tensorflow as tf
import tqdm
import vlogging

import net.utilities


# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)

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

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)


def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize

    logger = net.utilities.get_logger("/tmp/reference_gan.html")

    for epoch in range(1, epochs+1):

        print(f'Epoch: {epoch}')

        dLosses = []
        gLosses = []

        for _ in tqdm.tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

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
