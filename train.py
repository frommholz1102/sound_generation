import tensorflow as tf
import numpy as np
import os

from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150


def load_mnist():
    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize pixel values
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def fsdd(spectrograms_path):
    # spectrograms are stored as .npy in fsdd folder
    x_train = []
    for root, _, files in os.walk(spectrograms_path):
        for file in files:
            # load
            file_path = os.path.join(root, file)
            # append to x_train list
            spectrogram = np.load(file_path) # (n_f_bins, n_t_frames)
            x_train.append(spectrogram)
    # convert to numpy array
    x_train = np.array(x_train)        
    # model expects (n_t_frames, n_f_bins, 1) 3 dimensions (colored images)
    x_train = x_train[..., np.newaxis] 
    
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    # instantiate autoencoder object
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim = 128
    )
    autoencoder.summary()
    # compile and train
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)

    return autoencoder


if __name__ == "__main__":

    # import mnist dataset
    x_train = fsdd("fsdd/spectrograms/")
    print(x_train.shape)
    print('dataset loaded')
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)

    #autoencoder.save("models/vae_model")
