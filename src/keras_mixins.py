import numpy as np
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
  """
  https://keras.io/examples/generative/vae/
  """
  def __init__(self, fixed_epsilon=None):
    super(Sampling, self).__init__()
    self.fixed_epsilon = fixed_epsilon


  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    if self.fixed_epsilon is None:
      epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    else:
      return z_mean
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class AnnealingCallback(tf.keras.callbacks.Callback):
    """
    https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    """
    def __init__(self, n_iter, type='linear', start=0., stop=0.5, n_cycle=4, ratio=.5):
        if type == 'linear':
            self.kl_ratios = self.frange_cycle_linear(n_iter=n_iter, start=start, stop=stop, n_cycle=n_cycle, ratio=ratio)
        else:
            raise NotImplemented


    def on_epoch_begin(self, epoch, logs={}):
        tf.keras.backend.set_value(self.model.kl_ratio, self.kl_ratios[epoch])


    @classmethod
    def frange_cycle_linear(cls, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L



class VariationalAutoencoder(tf.keras.Model):
  """
  https://keras.io/examples/generative/vae/
  """
  def __init__(self, encoder, decoder, **kwargs):
    super(VariationalAutoencoder, self).__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
        name="reconstruction_loss"
    )
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    self.kl_ratio = tf.Variable(1.0, trainable=False, name='kl_ratio', dtype=tf.float32)


  @property
  def metrics(self):
    return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
    ]


  def train_step(self, data):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, reconstruction = self(data, training=True)
        reconstruction_loss = self.compiled_loss(data, reconstruction, regularization_losses=self.losses)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.kl_ratio * kl_loss
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


  def test_step(self, data):
    z_mean, z_log_var, reconstruction = self(data, training=True)
    reconstruction_loss = tf.keras.losses.mse(data, reconstruction)
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    total_loss = reconstruction_loss + self.kl_ratio * kl_loss
    reconstruction = self.decoder(z_mean)
    reconstruction_loss = self.compiled_loss(data, reconstruction, regularization_losses=self.losses)
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
    }


  def call(self, inputs, training=False, **kwargs):
    if training:
      z_mean, z_log_var, z = self.encoder(inputs)
      return z_mean, z_log_var, self.decoder(z)
    else:
      return self.decode(self.encode(inputs))

  def encode(self, instances, **kwargs):
    return self.encoder(instances)[0]

  def decode(self, instances, **kwargs):
    return self.decoder(instances)