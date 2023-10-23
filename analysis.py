import numpy as np
import matplotlib.pyplot as plt

from ae import Autoencoder
from train import load_mnist


def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    autoencoder = Autoencoder.load("model")
    # for analysis, we only need the test data
    _, _, x_test, y_test = load_mnist()

    num_sample_images_to_show = 8
    # sample random number of images from test set
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    # get reconstructed images from decoder
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    # plot original and reconstructed images
    plot_reconstructed_images(sample_images, reconstructed_images)

    # visualize leaned representations in latent space in a scatter plot
    num_images = 6000
    # labels are needed for color coding
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    # only latent representations are needed
    _, latent_representations = autoencoder.reconstruct(sample_images)
    # plot images in latent space
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
