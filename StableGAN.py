import os
import numpy as np
from numpy import random
from typing import Tuple
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Conv3DTranspose, Dropout, Input, Layer, InputSpec
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU, BatchNormalization, Reshape, LayerNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import Model
import tensorflow_addons as tfa
import math
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from m3gnet.models import M3GNet

m3gnet_model = M3GNet.load()
#Loading M3GNet model

def load_real_samples(data_path: str) -> np.ndarray:
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))

from google.colab import drive
drive.mount('/content/drive')

my_drive_path = '/content/drive/My Drive/Code'
os.chdir(my_drive_path)

load_real_samples('test.npy')

def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes

def conv_norm(x: tf.Tensor, units: int, filter: Tuple[int, int], stride: Tuple[int, int], discriminator: bool = True) -> tf.Tensor:
    if discriminator:
        conv = Conv3D(units, filter, strides = stride, padding = 'valid')
    else:
        conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
        x = tfa.layers.SpectralNormalization(conv)(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        return x

def dense_norm(x: tf.Tensor, units: int) -> tf.Tensor:
  x = tfa.layers.SpectralNormalization(Dense(units))(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  return x

class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=512, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size, shape)

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)

class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on exactly 2 inputs')

        (x,y) = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1],0,1)
        for i in range(len(K.int_shape(x))-1):
            weights = K.expand_dims(weights,-1)
        rw = x*weights + y*(1-weights)
        return rw

class Nontrainable(object):

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.trainable_status = self.model.trainable
        self.model.trainable = False
        return self.model

    def __exit__(self, type, value, traceback):
        self.model.trainable = self.trainable_status

class GradientPenalty(Layer):
    def call(self, inputs):
        real_image, generated_image, disc = inputs
        avg_image = RandomWeightedAverage()(
        [real_image, generated_image]
        )
        with tf.GradientTape() as tape:
          tape.watch(avg_image)
          disc_avg = disc(avg_image)

        grad = tape.gradient(disc_avg,[avg_image])[0]
        GP = K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))-1
        return GP

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)

def generate_real_samples(dataset: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    ix = random.randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]
    y = np.ones((n_samples,1))
    return X,y


def generate_latent_points(latent_dim: int, n_samples:int) -> np.ndarray:
    x_input = random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input

def generate_fake_samples(generator: tf.Tensor, latent_dim: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

def calculate_energy_above_hull(structure, m3gnet_model, phase_entries):
    energy = m3gnet_model.predict_structure(structure)
    pd_entry = PDEntry(composition=structure.composition, energy=energy)
    phase_entries.append(pd_entry)
    phase_diagram = PhaseDiagram(phase_entries)
    energy_above_hull = phase_diagram.get_e_above_hull(pd_entry)
    return energy_above_hull

def calculate_energy_above_hull(structure, m3gnet_model, phase_entries):
    energy = m3gnet_model.predict_structure(structure)
    pd_entry = PDEntry(composition=structure.composition, energy=energy)
    phase_entries.append(pd_entry)
    phase_diagram = PhaseDiagram(phase_entries)
    energy_above_hull = phase_diagram.get_e_above_hull(pd_entry) 
    return energy_above_hull

def filter_materials(npy_files, m3gnet_model, threshold=0.3):
    phase_entries = []
    filtered_files = []
    for npy_file in npy_files:
        structure = load_structure_from_npy(npy_file)
        energy_above_hull = calculate_energy_above_hull(structure, m3gnet_model, phase_entries)
        
        if energy_above_hull <= threshold:
            filtered_files.append(npy_file)

    return filtered_files

# List of .npy files
npy_files = ['material1.npy', 'material2.npy', 'material3.npy']  # Replace with your actual file names

# Filter the files
filtered_files = filter_materials(npy_files, m3gnet_model, threshold=0.3)

print(f"Filtered files: {filtered_files}")

    print(f'Epoch {epoch+1}/{num_epochs}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')


def define_critic(in_shape = (64, 64, 4, 1)) -> tf.Tensor:
    tens_in = Input(shape=in_shape, name="input")
    x = conv_norm(tens_in, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 16, (3,3,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 32, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 128, (2,2,2), (2,2,2), True)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    disc_out = tfa.layers.SpectralNormalization(Dense(1, activation = "linear"))(x)
    model = Model(inputs=tens_in, outputs=disc_out)
    return model

def define_generator(latent_dim):
    n_nodes = 16 * 16 * 4
    noise_in = Input(shape=(latent_dim,), name="noise_input")
    x = dense_norm(noise_in, n_nodes)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)
    outMat = tfa.layers.SpectralNormalization(Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid'))(x)
    model = Model(inputs=noise_in, outputs=outMat)
    return model


        def fit_generator(self, noise_gen, dataset, latent_dim, n_epochs=10, n_batch=20, n_critic=5, model_name=None):
          bat_per_epoch = int(1000 / n_batch)
          n_steps = bat_per_epoch * n_epochs
          half_batch = int(n_batch / 2)
          disc_out_shape = (n_batch, self.disc.output_shape[1])
          real_target = -np.ones(disc_out_shape, dtype=np.float32)
          fake_target = -real_target
          gp_target = np.zeros_like(real_target)
          lastEpoch = 0
          genLossArr = []
          disc0LossArr = []
          disc1LossArr = []
          disc2LossArr = []

          for epoch in range(n_epochs):
              print("Epoch {}/{}".format(epoch + 1, n_epochs))

              # Initialize tqdm progress bar
              progbar = tqdm(total=bat_per_epoch, desc=f'Epoch {epoch + 1}/{n_epochs}')

              for step in range(bat_per_epoch):
                  # Train discriminator
                  with Nontrainable(self.gen):
                      for repeat in range(n_critic):
                          tens_batch, _ = generate_real_samples(dataset, n_batch)
                          noise_batch = next(noise_gen)
                          disc_loss = self.disc_trainer.train_on_batch(
                              [tens_batch] + noise_batch,
                              [real_target, fake_target, gp_target]
                          )

                  # Train generator
                  with Nontrainable(self.disc):
                      noise_batch = next(noise_gen)
                      gen_loss = self.gen_trainer.train_on_batch(
                          noise_batch, real_target
                      )

                  losses = []
                  for i, dl in enumerate(disc_loss):
                      losses.append(("D{}".format(i), dl))
                      if i == 0:
                          disc0LossArr.append(dl)
                      elif i == 1:
                          disc1LossArr.append(dl)
                      elif i == 2:
                          disc2LossArr.append(dl)
                  losses.append(("G0", gen_loss))
                  genLossArr.append(gen_loss)
                  progbar.set_postfix(losses=dict(losses))
                  progbar.update(1)  # Update progress bar

              progbar.close()  # Close the progress bar after epoch

              # Save model and losses
              if model_name:
                  self.gen.save(model_name + "gen")
                  self.disc.save(model_name + "critic")
                  np.save(model_name + "real_loss", np.array(disc0LossArr))
                  np.save(model_name + "fake_loss", np.array(disc1LossArr))
                  np.save(model_name + "gp_loss", np.array(disc2LossArr))
                  np.save(model_name + "generator_loss", np.array(genLossArr))
                  print("Training complete!")


from tqdm import tqdm
class WGANGP(object):

        def __init__(self, gen, disc, lr_gen=0.0001, lr_disc=0.0001):

          self.gen = gen
          self.disc = disc
          self.lr_gen = lr_gen
          self.lr_disc = lr_disc
          self.build()

        def build(self):
            # ...
            try:
                #tens_shape = input_shapes(self.disc, "input")[0]
                tens_shape = (64, 64, 4, 1)
            except:
                tens_shape = (64, 64, 4, 1)
            try:
                noise_shapes = input_shapes(self.gen, "noise_input")
            except:
                noise_shapes =(128,)

            self.opt_disc = Adam(self.lr_disc, beta_1=0.0, beta_2=0.9)
            self.opt_gen = Adam(self.lr_gen, beta_1=0.0, beta_2=0.9)

            with Nontrainable(self.gen):
                real_image = Input(shape=tens_shape)
                noise = [Input(shape=s) for s in noise_shapes]

                disc_real = self.disc(real_image)
                generated_image = self.gen(noise)
                disc_fake = self.disc(generated_image)

                gp = GradientPenalty()([real_image, generated_image, self.disc])
                self.disc_trainer = Model(
                    inputs=[real_image, noise],
                    outputs=[disc_real, disc_fake, gp]
                )
                self.disc_trainer.compile(optimizer=self.opt_disc,
                    loss=[wasserstein_loss, wasserstein_loss, 'mse'],
                    loss_weights=[1.0, 1.0, 10.0]
                )

            with Nontrainable(self.disc):
                noise = [Input(shape=s) for s in noise_shapes]

                generated_image = self.gen(noise)
                disc_fake = self.disc(generated_image)

                self.gen_trainer = Model(
                    inputs=noise,
                    outputs=disc_fake
                )
                self.gen_trainer.compile(optimizer=self.opt_gen,
                    loss=wasserstein_loss)
                
        def fit_generator(self, noise_gen, dataset, latent_dim, n_epochs=10, n_batch=20, n_critic=5, model_name=None):
          bat_per_epoch = int(1000 / n_batch)
          n_steps = bat_per_epoch * n_epochs
          half_batch = int(n_batch / 2)
          disc_out_shape = (n_batch, self.disc.output_shape[1])
          real_target = -np.ones(disc_out_shape, dtype=np.float32)
          fake_target = -real_target
          gp_target = np.zeros_like(real_target)
          lastEpoch = 0
          genLossArr = []
          disc0LossArr = []
          disc1LossArr = []
          disc2LossArr = []

          for epoch in range(n_epochs):
              print("Epoch {}/{}".format(epoch + 1, n_epochs))

              # Initialize tqdm progress bar
              progbar = tqdm(total=bat_per_epoch, desc=f'Epoch {epoch + 1}/{n_epochs}')

              for step in range(bat_per_epoch):
                  # Train discriminator
                  with Nontrainable(self.gen):
                      for repeat in range(n_critic):
                          tens_batch, _ = generate_real_samples(dataset, n_batch)
                          noise_batch = next(noise_gen)
                          disc_loss = self.disc_trainer.train_on_batch(
                              [tens_batch] + noise_batch,
                              [real_target, fake_target, gp_target]
                          )

                  # Train generator
                  with Nontrainable(self.disc):
                      noise_batch = next(noise_gen)
                      gen_loss = self.gen_trainer.train_on_batch(
                          noise_batch, real_target
                      )

                  losses = []
                  for i, dl in enumerate(disc_loss):
                      losses.append(("D{}".format(i), dl))
                      if i == 0:
                          disc0LossArr.append(dl)
                      elif i == 1:
                          disc1LossArr.append(dl)
                      elif i == 2:
                          disc2LossArr.append(dl)
                  losses.append(("G0", gen_loss))
                  genLossArr.append(gen_loss)
                  progbar.set_postfix(losses=dict(losses))
                  progbar.update(1)  # Update progress bar

              progbar.close()  # Close the progress bar after epoch

              # Save model and losses
              if model_name:
                  self.gen.save(model_name + "gen")
                  self.disc.save(model_name + "critic")
                  np.save(model_name + "real_loss", np.array(disc0LossArr))
                  np.save(model_name + "fake_loss", np.array(disc1LossArr))
                  np.save(model_name + "gp_loss", np.array(disc2LossArr))
                  np.save(model_name + "generator_loss", np.array(genLossArr))
                  print("Training complete!")


batch_size = 1

data_path = '/content/drive/My Drive/Code/'
os.chdir(data_path)

file_path = '/content/drive/My Drive/Code/test.npy'
data = np.load(file_path)

n_epochs = 10
n_critic = 5
model_path = './model'

def main():
 #   args = parser.parse_args()
    noise_dim = 128
    critic = define_critic()
    generator = define_generator(noise_dim)
    gan_model = WGANGP(generator, critic)
    noise_gen = NoiseGenerator([noise_dim,], batch_size = batch_size)
    dataset = load_real_samples(file_path)
    gan_model.fit_generator(noise_gen, dataset, noise_dim, n_epochs, batch_size,n_critic, model_path)

if __name__ == "__main__":
    main()

