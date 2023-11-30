import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow as tf
print(tf.executing_eagerly())
"""
In TF.2.x the KL-loss leads to problems with eager execution.
This post explains how to solvew them by writing custim layers:
https://linux-blog.anracom.com/2022/05/23/variational-autoencoder-with-tensorflow-iii-problems-with-the-kl-loss-and-eager-execution/
"""
tf.compat.v1.disable_eager_execution()
#tf.python.framework.ops.disable_eager_execution()
print(tf.executing_eagerly())

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
        # use builtin tensorflow method
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    
    def compile(self, learning_rate=0.0001):
        # use custom loss here
        # metrics to observe during training
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss=None)
                           #metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss])


    def train(self, x_train, batch_size, num_epochs):
        # target data is input data for autoencoders
        self.model.fit(
            x_train,
            None,
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
    

    class CustomVAELayer(tf.keras.layers.Layer):
        """
        Custom variational autoencoder loss.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)
            self.mu, self.log_variance = args
            self.reconstruction_loss_weight = 1000000
            self.reconstruction_loss = None
            self.kl_loss = None
            self.combined_loss = None

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
            self.reconstruction_loss = reconstruction_loss
            return reconstruction_loss
        
        def _calculate_kl_loss(self, y_target, y_predicted):
            # custom keras loss function expects y_true and y_pred as arguments
            # KL divergence loss (closed form)
            # pull latent space distribution towards standard normal distribution
            kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
            self.kl_loss = kl_loss
            return kl_loss
            
        def get_vae_loss(self, y_target, y_predicted):
            input_img = y_target
            output_img = y_predicted
            # calculate and set internal losses
            self.combined_loss = self._calculate_combined_loss(input_img, output_img)

        def call(self, inputs):
            input_img = inputs[0]
            output_img = inputs[1]
            self.get_vae_loss(input_img, output_img)
            self.add_loss(self.combined_loss, inputs=inputs)
            self.add_metric(self.reconstruction_loss, name="reconstruction_loss", aggregation="mean")
            self.add_metric(self.kl_loss, name="kl_loss", aggregation="mean")
            self.add_metric(self.combined_loss, name="combined_loss", aggregation="mean")
            # We don't use this output.
            return output_img


    def _build(self):
        self._build_encoder()
        self.build_decoder()
        self._build_autoencoder()


    # -------------- ENCODER ----------------
    def _build_encoder(self):
        # get model input
        encoder_input = self._add_encoder_input()
        self._model_input = encoder_input
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
        Flatten data and add bottleneck with gaussian sampling (latent space).
        The data point is sampled from a gaussian distribution in the
        latent space according to the function : z = mu + sigma * epsilon
        """

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            # sample epsilon from standard normal distribution
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            # sample point from normal distribution
            sampled_point = mu + K.exp(log_variance / 2) * epsilon

            return sampled_point
    
        # store information about shape of data (ignore batch size) before flattening
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [batch_size, width, height, channels]
        # flatten  
        x = Flatten()(x)

        # mu, log_variance to sample from gaussian distribution (define distribution)
        # no sequential graph since mu and log_variance are both applied to previous graph (split up) 
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        #print(self.mu, self.log_variance)
        # sample data point from gaussian distribution
        # wrap functions within graph with lambda layer
        
        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])

        return x
        

    # -------------- DECODER ----------------
    def build_decoder(self):
        # bottleneck layer will be input to the decoder
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        # set decoder as the final model attribute
        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    def _add_decoder_input(self):

        return Input(shape=self.latent_space_dim, name="decoder_input")
    

    def _add_dense_layer(self, decoder_input):  
        # we want the same number as after the flattening in the encoder 
        # product of all dimensions except the first (batch size) gives length of dense layer
        num_neurons = K.prod(self._shape_before_bottleneck)
        dense_layer = Dense(K.get_value(num_neurons), name="decoder_dense")(decoder_input)

        return dense_layer
    

    def _add_reshape_layer(self, dense_layer):
        # result should be same shape as before flattening in the encoder
        # we can use the shape stored in self._shape_before_bottleneck
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)

        return reshape_layer


    def _add_conv_transpose_layers(self, x):
        """
        Add concolutional transpose blocks 
        (conv_transpose + ReLU + batch normalization).
        """
        # loop through all conv layers from encoder in reverse order 
        # (stop at first layer, special case for layer 1)

        for layer_idx in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_idx, x)
        
        return x
    

    def _add_conv_transpose_layer(self, layer_idx, x):
        # layer number is asending as layer idx is descending
        layer_num = self._num_conv_layers - layer_idx
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_idx],
            kernel_size = self.conv_kernels[layer_idx],
            strides = self.conv_strides[layer_idx],
            padding = "same", 
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        # only add ReLU and batch normalization if this is not the last layer
        if layer_idx < self._num_conv_layers - 1:
            x = ReLU(name=f"decoder_relu_{layer_num}")(x)
            x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        
        return x


    def _add_decoder_output(self, x):
        # add final layer with sigmoid activation
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name="decoder_output"
        )
        x = conv_transpose_layer(x)
        # apply sigmoid activation
        x = Activation("sigmoid")(x)

        return x

    
    # -------------- AUTOENCODER ----------------
    def _build_autoencoder(self):
        # model input
        model_input = self._model_input
        # model output is result of data through encoder and decoder
        decoder_output = self.decoder(self.encoder(model_input))
        # include custom loss in the model
        print(decoder_output.shape)
        model_output = self.CustomVAELayer(self.mu, self.log_variance)([model_input, decoder_output])
        print(model_output.shape)
        # define model
        self.model = Model(model_input, model_output, name="autoencoder")





if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()