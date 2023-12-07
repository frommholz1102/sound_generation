import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow as tf


class VAE:
    """
    Deep Convolutional VAE (variational autoencoder) architecture for sound generation.
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
        print(self.input_shape)
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000

        # Define components of autoencoder
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        # high-level build method
        self._build()

    def summary(self):
        # use builtin tensorflow method
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    
    def compile(self, learning_rate=0.0001):
        # use custom loss here
        # metrics to observe during training
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss=self._calculate_combined_loss, 
                           metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss])


    def train(self, x_train, batch_size, num_epochs):
        # target data is input data for autoencoders
        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )


    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        # parameters of autoencoder object (needed to recreate object)
        self._save_parameters(save_folder)
        # save model weights
        self._save_weights(save_folder)


    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)


    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
        

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)


    def reconstruct(self, images):
        # images will be numpy array
        # get points on latent space for each image
        latent_representations = self.encoder.predict(images)
        # feed data in latent space to decoder
        reconstructed_images = self.decoder.predict(latent_representations)

        return reconstructed_images, latent_representations
    

    @classmethod
    def load(cls, save_folder="."):
        # load parameters
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        # instantiate class and pass list of params
        autoencoder = VAE(*parameters)
        # load weights
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)

        return autoencoder
    

    def _calculate_combined_loss(self, y_target, y_predicted):
        # call custom loss functions 
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)  
        # multiply reconstruction loss by weight factor (hyperparameter)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss

        return combined_loss
    

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        # error is difference between ground truth and predicted values
        error = y_target - y_predicted
        # mean squared error loss on all axes
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])

        return reconstruction_loss
    

    def _calculate_kl_loss(self, y_target, y_predicted):
        # custom keras loss function expects y_true and y_pred as arguments
        # KL divergence loss (closed form)
        # pull latent space distribution towards standard normal distribution
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)

        return kl_loss


    def _build(self):
        self._build_encoder()
        self.build_decoder()
        self._build_autoencoder()


    # -------------- ENCODER ----------------
    def _build_encoder(self):

        encoder = tf.keras.Sequential()
        # get input layer dimensions from input shape
        encoder.add(InputLayer(input_shape=self.input_shape, name="encoder_input"))
        # add convolutional layers, relu activation and batch normalization
        for layer_index in range(self._num_conv_layers):
            encoder.add(self._get_conv_layer(layer_index))
            encoder.add(ReLU(name=f"encoder_relu_{layer_index+1}"))
            encoder.add(BatchNormalization(name=f"encoder_bn_{layer_index+1}"))

        # get shape of last layer -> save as self._shape_before_bottleneck
        self._shape_before_bottleneck = K.int_shape(encoder.layers[-1].output)[1:] # [batch_size, width, height, channels]
        # flatten to prepare for bottleneck
        encoder.add(Flatten())

        self.encoder = encoder
    

    def _get_conv_layer(self, layer_index):
        """
        Returns a convolutional layer parametrized by the index of the layer.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        
        return conv_layer
    

    # -------------- BOTTLENECK ----------------
    def _sample_from_latent_space(self, x):
        """
        Flatten data and add bottleneck with gaussian sampling (latent space).
        The data point is sampled from a gaussian distribution in the
        latent space according to the function : z = mu + sigma * epsilon
        """

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            # sample epsilon from standard normal distribution
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            # sample point from normal distribution
            sampled_point = mu + K.exp(log_variance / 2) * epsilon

            return sampled_point
    

        # mu, log_variance to sample from gaussian distribution (define distribution)
        # no sequential graph since mu and log_variance are both applied to previous graph (split up) 
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        print("mu", self.mu, K.int_shape(self.mu))
        print("log_variance", self.log_variance, K.int_shape(self.log_variance))
        
        # sample data point from gaussian distribution
        # wrap functions within graph with lambda layer
        z = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])


        return z
        

    # -------------- DECODER ----------------
    def build_decoder(self):

        decoder = tf.keras.Sequential()
        # get input layer dimensions from laten space shape (enoconder output)
        decoder.add(InputLayer(input_shape=self.latent_space_dim, name="decoder_input"))
        # add dense layer with shape of last layer before flattening in encoder
        decoder.add(self._get_dense_decoder_layer())
        # reshape to shape before flattening in encoder
        decoder.add(Reshape(self._shape_before_bottleneck))

        for layer_idx in reversed(range(1, self._num_conv_layers)):
            decoder.add(self._get_conv_transpose_layer(layer_idx))
             # only add ReLU and batch normalization if this is not the last layer
            layer_num = self._num_conv_layers - layer_idx
            if layer_idx < self._num_conv_layers - 1:
                decoder.add(ReLU(name=f"decoder_relu_{layer_num}"))
                decoder.add(BatchNormalization(name=f"decoder_bn_{layer_num}"))

        # add final layer with sigmoid activation
        decoder.add(self._get_decoder_output())
        decoder.add(Activation("sigmoid"))
        # set decoder as the final model attribute
        self.decoder = decoder
    

    def _get_dense_decoder_layer(self):  
        # we want the same number as after the flattening in the encoder 
        # product of all dimensions except the first (batch size) gives length of dense layer
        num_neurons = K.prod(self._shape_before_bottleneck)
        dense_layer = Dense(K.get_value(num_neurons), name="decoder_dense")

        return dense_layer


    def _get_conv_transpose_layer(self, layer_idx):
        # layer number is asending as layer idx is descending
        layer_num = self._num_conv_layers - layer_idx
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_idx],
            kernel_size = self.conv_kernels[layer_idx],
            strides = self.conv_strides[layer_idx],
            padding = "same", 
            name=f"decoder_conv_transpose_layer_{layer_num}", 
        )
       
        return conv_transpose_layer


    def _get_decoder_output(self):
        # add final layer
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name="decoder_output"
        )

        return conv_transpose_layer

    
    def _decode(self, z):
        # pass data through decoder
        reconstruction = self.decoder(z)

        return reconstruction
    
    
    # -------------- AUTOENCODER ----------------
    def _build_autoencoder(self):

        # Define the input layer
        model_input = Input(shape=self.input_shape)
        # passing through encoder and sampling from latent space
        x = self.encoder(model_input)
        z = self._sample_from_latent_space(x)
        # passing through decoder
        model_output = self.decoder(z)

        self.model = Model(inputs=model_input, outputs=model_output, name="autoencoder")


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()