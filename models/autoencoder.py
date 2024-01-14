import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

class Encoder(keras.Model):
    def __init__(self, latent_dim=2, data_dim=(3,3), filter_dim=(3,3)):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.filter_dim = filter_dim
        self.encoder = keras.Sequential([
            keras.Input(shape=(96, 96, 1)),
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(2000, activation="relu"),
        ])
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        # self.z = Sampling()

    def call(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return self.sampling([z_mean, z_log_var])
    
    def sampling(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Decoder(keras.Model):
    def __init__(self, latent_dim=2, data_dim=(3,3), filter_dim=(3,3)):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.filter_dim = filter_dim
        self.Decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(36864, activation="relu"),
            layers.Reshape((24, 24, 64)),
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")
        ])

    def call(self, x):
        x = self.encoder(x)
        return x


class VariationalAutoencoder(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

