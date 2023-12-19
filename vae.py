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
        self.input_shape = input_shape # [1, 28, 28]
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
        self._shape_after_flatten = None
        self._model_input = None

        # high-level build method
        self._build()
        #self.summary()

    def summary(self):
        print("Encoder Summary:")
        torchsummary.summary(self.encoder, input_size=self.input_shape)

        print()
        print("Bottleneck Summary:")
        test_input = torch.zeros(1, self._shape_after_flatten)
        print('Shape before bottleneck: ', test_input.size())
        test_output = self.mu(test_input)
        print('Shape after bottleneck: ', test_output.size())

        print()
        print("Decoder Summary:")
        torchsummary.summary(self.decoder, input_size=(1,64,7,7))
        

    
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
        self.build_decoder()


    # -------------- ENCODER ----------------
    def _build_encoder(self):
        """
        Build the encoder part of the VAE.
        """
        # add convolutional blocks
        encoder_layers = self._add_conv_blocks()
        # save shape before bottleneck (for reconstruction in decoder)
        self._calculate_shape_before_bottleneck(encoder_layers)
        # flatten output and add bottleneck
        encoder_layers.append(nn.Flatten())
        # build encoder model
        self.encoder = nn.Sequential(*encoder_layers)
        # create mu and log_var layers
        self._add_bottleneck()

    

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
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
    

    def _add_conv_blocks(self):
        """
        Returns a list of conv blocks (conv2d, relu and batchnorn).
        """
        # first in_channels is 1 (from input shape)
        in_channels = self.input_shape[0] 
        # define layer list 
        encoder_layers = []
        
        # append conv blocks to encoder architecture
        for layer_index in range(len(self.conv_filters)):
            encoder_layers.append(
                self._conv_block(
                    in_channels,
                    self.conv_filters[layer_index],
                    self.conv_kernels[layer_index],
                    self.conv_strides[layer_index]
                )
            )
            # next in_channels is the current conv_filter (out_channels)
            in_channels = self.conv_filters[layer_index]
        return encoder_layers
    

    def _calculate_shape_before_bottleneck(self, encoder_layers):
        dummy_input = torch.zeros(1, *self.input_shape)  # Create a dummy input with the desired shape
        encoder = nn.Sequential(*encoder_layers)  # Create encoder model
        output = encoder(dummy_input)  # Pass the dummy input through the layer
        self._shape_before_bottleneck =output.size()  # Get the shape of the output tensor
        self._shape_after_flatten = torch.prod(torch.tensor(output.size()[1:]))


    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.log_var(x)
        return mu, logvar
        

    # -------------- BOTTLENECK ----------------
    def _add_bottleneck(self):
        # define dense layers for mu and log_var
        # same shape as self.latent_space_dim
        self.mu = nn.Linear(self._shape_after_flatten, self.latent_space_dim)
        self.log_var = nn.Linear(self._shape_after_flatten, self.latent_space_dim)

    
    def sample_from_normal_distribution(self, mu, log_var):
        # epsilon is drawn from distribution with same shapes as 
        epsilon = torch.randn_like(mu)  
        sampled_point = mu + torch.exp(log_var / 2) * epsilon
        return sampled_point


    # -------------- DECODER ----------------
    def build_decoder(self):
        """
        Build decoder part with cont transpose blocks (equivalent to encoder structure).
        """

        # get decoder layers 
        decoder_layers = self._add_conv_transpose_blocks()
        # build decoder model
        self.decoder = nn.Sequential(*decoder_layers)

    

    def reshape_before_decoder(self, z):
        # there is no layer that can resahpe a tensor so the reshape is done manually
        # fully connected layer from latent_space_dim to shape_after_flatten
        decoder_fc = nn.Linear(self.latent_space_dim, self._shape_after_flatten)
        x = decoder_fc(z)
        # reshape to shape_before_bottleneck
        x = torch.reshape(x, self._shape_before_bottleneck)
        return x


    def _conv_transpose_block(self, in_channels, out_channels, kernel_size, stride):
        """
        Equivalent to conv_block but with Conv2DTranspose.
        """

        print(in_channels, out_channels)

        padding = (kernel_size - 1) // 2 
        conv_transpose_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
        )
        relu = nn.ReLU()
        batch_norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            conv_transpose_layer,
            relu,
            batch_norm
        )
    

    def _add_conv_transpose_blocks(self):
        
        # first in_channel is the last shape before bottleneck
        in_channels = self._shape_before_bottleneck[0]
        print('decoder input shape: ', in_channels)
        # define layer list
        decoder_layers = []

        # go through indices of self.conv_filters in reverse order
        # only go to 1 because the last layer is only conv_transpose (not a block)
        for layer_index in range(len(self.conv_filters)-1, 0, -1):
            print(layer_index)
            # append conv_transpose block
            decoder_layers.append(
                self._conv_transpose_block(
                    in_channels,
                    self.conv_filters[layer_index],
                    self.conv_kernels[layer_index],
                    self.conv_strides[layer_index]
                )
            )
            # next in_channels is the current conv_filter (out_channels)
            in_channels = self.conv_filters[layer_index]
        print('for loop is done, in_channels = ', in_channels)
        # add last layer (transpose only)
        padding = (self.conv_kernels[0] - 1) // 2 
        decoder_layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=self.conv_kernels[0],
                stride=self.conv_strides[0],
                padding=padding
            )
        )
        return decoder_layers
    

    def decode(self, z):
        x = self.decoder(z)
        return x
    

    
    # -------------- AUTOENCODER ----------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_from_normal_distribution(mu, logvar)
        # reshape cannot be done in forward method because it is not a layer
        print('z shape: ', z.size())
        decoder_input = self.reshape_before_decoder(z)
        print('decoder input shape: ', decoder_input.size())
        model_output = self.decode(decoder_input)
        print('model output shape: ', model_output.size())
        return model_output


if __name__ == "__main__":
    
    vae = VAE(
        input_shape=(1, 28, 28),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )

    dummy_input = torch.zeros(1, 1, 28, 28)
    dummy_output = vae(dummy_input)
    print(dummy_output.shape)

