from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K


class Autoencoder:
    """
    Deep Convolutional Autoencoder architecture for sound generation.
    Components are encoder and decoder are mirrored.
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

        # high-level build method
        self._build()

    def summary(self):
        # use builtin tensorflow method
        self.encoder.summary()

    # private build method to create model
    def _build(self):
        self._build_encoder()
        #self._build_decoder()
        #self._build_autoencoder()

    # private build method to create encoder
    def _build_encoder(self):
        # get model input
        encoder_input = self._add_encoder_input()
        # pass input to convolutional layers
        conv_layers = self._add_conv_layers(encoder_input)
        # returns stack of conv layers as well as bottleneck layer
        bottleneck = self._add_bottleneck(conv_layers)
        # pass input and stack of layers with bottleneck architecture
        # set encoder as the final model attribute
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """
        Creates all convolutional blocks in encoder.
        """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    # this is applied multiple times in _add_conv_layers
    def _add_conv_layer(self, layer_index, x):
        """
        Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """
        Flatten data and add bottleneck (latent space).
        """
        # store information about shape of data (ignore batch size) before flattening
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [batch_size, width, height, channels]
        # flatten and add bottleneck
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x
        

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()