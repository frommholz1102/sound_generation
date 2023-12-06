# this script rewrits the VAE model in pytorch

import torch
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

    def summary(self):
        pass

    
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
        encoder_layers = []
        input_shape = self.input_shape
        print(input_shape)

        pass
    

    def _add_conv_layers(self, encoder_input):
        pass
    

    def _add_conv_layer(self, layer_index, x):
        pass
    

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
        input_shape=(1, 256, 64),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )

