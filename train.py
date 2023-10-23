import tensorflow as tf
import numpy as np

from ae import Autoencoder

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 10


def load_mnist():
    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    # instantiate autoencoder object
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    #autoencoder.summary()
    # compile and train
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)

    return autoencoder


if __name__ == "__main__":

    # import mnist dataset
    x_train, _, _, _ = load_mnist()
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)

    autoencoder.save("model")
    autoencoder2 = Autoencoder.load("model")
    autoencoder2.summary()