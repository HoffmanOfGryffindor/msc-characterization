import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Encoder(keras.Model):
    def __init__(self, config: dict, shape: list):
        super(Encoder, self).__init__()
        self.latent_dim = config["model"]["latent_dim"]
        self.data_dim = tuple(shape)
        self.filter_dim = tuple(config["model"]["filter_dim"])
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=self.data_dim),
                layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
                layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
                layers.Flatten(),
                layers.Dense(2000, activation="relu"),
            ]
        )
        self.logits_mean = layers.Dense(self.latent_dim, name="z_mean")
        self.logits_log_var = layers.Dense(self.latent_dim, name="z_log_var")
        self.sample = Sampling()

    def call(self, x: np.ndarray) -> np.ndarray:
        x = self.encoder(x)
        logits_mean = self.logits_mean(x)
        logits_log_var = self.logits_log_var(x)
        return logits_mean, logits_log_var, self.sampling([logits_mean, logits_log_var])

    @tf.function
    def sampling(self, inputs):
        logits_mean, logits_log_var = inputs
        batch, dim = tf.shape(logits_mean)[0], tf.shape(logits_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return logits_mean + tf.exp(0.5 * logits_log_var) * epsilon


class Decoder(keras.Model):
    def __init__(self, config: dict):
        super(Decoder, self).__init__()
        self.latent_dim = config["model"]["latent_dim"]
        self.decoder = keras.Sequential(
            [
                keras.Input(shape=(self.latent_dim,)),
                layers.Dense(36864, activation="relu"),
                layers.Reshape((24, 24, 64)),
                layers.Conv2DTranspose(
                    64, 3, activation="relu", strides=2, padding="same"
                ),
                layers.Conv2DTranspose(
                    32, 3, activation="relu", strides=2, padding="same"
                ),
                layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        x = self.decoder(x)
        return x


class VariationalAutoencoder(keras.Model):
    def __init__(self, config: dict, shape: list):
        super().__init__()
        self.encoder = Encoder(config, shape)
        self.decoder = Decoder(config)
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
            logits_mean, logits_log_var, logits = self.encoder(data)
            reconstruction = self.decoder(logits)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + logits_log_var - tf.square(logits_mean) - tf.exp(logits_log_var))
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
