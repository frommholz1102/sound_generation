# this script rewrits the VAE model in pytorch

import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F

# make sure to use gpu
print('CUDA available: ', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    """
    Pytorch implementation of VAE
    """

    # Conv parameters are going to be lists (one value for each layer)
    def __init__(self, 
                input_shape, 
                conv_filters,
                conv_kernels,
                conv_strides,
                latent_space_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2

        # Define components of autoencoder
        self.encoder = None
        self.decoder = None
        # Build autoencoder model
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        # high-level build method
        self._build()
        self.summary()

    def summary(self):
        print("Encoder Summary:")
        torchsummary.summary(self.encoder, input_size=self.input_shape)

    
    def compile(self, learning_rate=0.0001):
        pass


    def train(self, x_train, batch_size, num_epochs):
        pass


    def save(self, save_folder="."):
        pass


    def _create_folder_if_it_doesnt_exist(self, folder):
        pass


    def _save_parameters(self, save_folder):
        pass
        

    def _save_weights(self, save_folder):
        pass


    def reconstruct(self, images):
        pass
    

    @classmethod
    def load(cls, save_folder="."):
        pass
    


    def _calculate_combined_loss(self, y_target, y_predicted):
        pass
    
    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        pass
    
    def _calculate_kl_loss(self, y_target, y_predicted):
        pass
        
    def get_vae_loss(self, y_target, y_predicted):
        pass


    def _build(self):
        self._build_encoder()


    # -------------- ENCODER ----------------
    def _build_encoder(self):
        """
        Build the encoder part of the VAE.
        """
        encoder_layers = []
        # add convolutional blocks
        encoder_layers = self._add_conv_blocks(
            encoder_layers,
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides
        )
        # save shape before bottleneck (for reconstruction in decoder)
        self._calculate_shape_before_bottleneck(encoder_layers)
        # flatten output and add bottleneck
        encoder_layers.append(nn.Flatten())
        # build encoder model
        self.encoder = nn.Sequential(*encoder_layers)
    

    def _conv_block(self, layer_index, in_channels, out_channels, kernel_size, stride):
        """
        Defines a convolutional block consisting of conv2d + ReLU + batch normalization.
        """
        # calculate padding since "same" padding is not supported for strided convolutions
        padding = (kernel_size - 1) // 2 
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            )
        relu = nn.ReLU()
        batch_norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            conv_layer,
            relu,
            batch_norm
        )
    

    def _add_conv_blocks(self, encoder_layers, input_shape, conv_filters, conv_kernels, conv_strides):
        """
        Returns a list of conv blocks (conv2d, relu and batchnorn).
        """
        # first in_channels is 1 (from input shape)
        in_channels = input_shape[0] 
        
        # append conv blocks to encoder architecture
        for layer_index in range(len(conv_filters)):
            print("layer_index: ", layer_index)
            encoder_layers.append(
                self._conv_block(
                    layer_index,
                    in_channels,
                    conv_filters[layer_index],
                    conv_kernels[layer_index],
                    conv_strides[layer_index]
                )
            )
            # next in_channels is the current conv_filter (out_channels)
            in_channels = conv_filters[layer_index]
        return encoder_layers
    

    def _calculate_shape_before_bottleneck(self, encoder_layers):
        dummy_input = torch.zeros(1, *self.input_shape)  # Create a dummy input with the desired shape
        encoder = nn.Sequential(*encoder_layers)  # Create encoder model
        output = encoder(dummy_input)  # Pass the dummy input through the layer
        self._shape_before_bottleneck = output.shape  # Get the shape of the output tensor


    def _add_bottleneck(self, x):
        pass
        

    # -------------- DECODER ----------------
    def build_decoder(self):
        pass
    

    def _add_dense_layer(self, decoder_input):  
       pass
    

    def _add_reshape_layer(self, dense_layer):
       pass


    def _add_conv_transpose_layers(self, x):
        pass
    

    def _add_conv_transpose_layer(self, layer_idx, x):
        pass


    def _add_decoder_output(self, x):
        pass

    
    # -------------- AUTOENCODER ----------------
    def _build_autoencoder(self):
       pass


if __name__ == "__main__":
    vae = VAE(
        input_shape=(1, 28, 28),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )

