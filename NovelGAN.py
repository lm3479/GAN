import os
import argparse
import numpy as np
from typing import Tuple
import tensorflow as tf
from numpy import random
from typing import Tuple
import keras
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, Dropout, Input, Layer
from tensorflow.keras.layers import Flatten, LeakyReLU, ReLU, BatchNormalization, Reshape, LayerNormalization
from tensorflow.keras import backend as K
from keras.utils import generic_utils
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import InputSpec
import math

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to where the dataset is stored.")
parser.add_argument("--save_path", type=str, 
                    help="Path to where the models and loss history is stored.")
parser.add_argument("--n_epochs", type=int, help="The number of training epochs.")
parser.add_argument("--batch_size", type=int, help="The training batch size.")
parser.add_argument("--n_critic", type=int, help="Number of critic training cycles per epoch.")

def load_real_samples(data_path: str) -> np.ndarray:
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))

def input_shapes(model, prefix):
    shapes = [il.shape[1:] for il in 
        model.inputs if il.name.startswith(prefix)]
    shapes = [tuple([d for d in dims]) for dims in shapes]
    return shapes

def conv_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, filter: Tuple[int, int, int], stride : Tuple[int, int, int], discriminator: bool = True) -> tf.Tensor:
  if discriminator:
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = tfa.layers.SpectralNormalization(conv)(x)
  x = LayerNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x) 
  return x

def dense_norm(x: keras.engine.keras_tensor.KerasTensor, units: int) -> tf.Tensor:
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
 
def generate_real_samples(dataset: np.ndarray, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
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
 
class WGANGP(object):
    def __init__(self, gen, disc, lr_gen=0.0001, lr_disc=0.0001):
      self.gen = gen
      self.disc = disc
      self.lr_gen = lr_gen
      self.lr_disc = lr_disc
      self.build()
 
    def build(self):
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

def predict_ehull(dir, model_path, output_path, api_key):
  m3gnet_e_form = M3GNet.from_dir(model_path)
  ehull_list = []
  for file_name in os.listdir(dir):
      crystal = Structure.from_file(dir + file_name, sort = True, merge_tol=0.01)
      try:
          e_form_predict = m3gnet_e_form.predict_structure(crystal)
        
      except:
          print("Could not predict formation energy of ", crystal)
          ehull_list.append((file_name, "N/A"))
          break
      elements = ''.join([i for i in crystal.formula if not i.isdigit()]).split(" ")
      mat_api_key = api_key
      mpr = MPRester(mat_api_key)
  #Extracts information about crystals, and additional compounds
      all_compounds = mpr.summary.search(elements = elements)
      insert_list = []
    for compound in all_compounds:
          for element in ''.join([i for i in str(compound.composition) if not i.isdigit()]).split(" "):
              if element not in elements and element not in insert_list:
                insert_list.append(element)
          for element in elements + insert_list:
              all_compounds += mpr.summary.search(elements = [element], num_elements = (1,1))
          pde_list = []
          for i in range(len(all_compounds)):
              comp = str(all_compounds[i].composition.reduced_composition).replace(" ", "")
              pde_list.append(ComputedEntry(comp, all_compounds[i].formation_energy_per_atom))
          try:
              diagram = PhaseDiagram(pde_list)
              _, pmg_ehull = diagram.get_decomp_and_e_above_hull(ComputedEntry(Composition(crystal.formula.replace(" ", "")), e_form_predict[0][0].numpy()))
              ehull_list.append((file_name, pmg_ehull))
          except:
              print("Could not create phase diagram")
              ehull_list.append((file_name, "N/A"))
              continue
      np.save(output_path, np.array(ehull_list))
  #Predicting distance above convex hull: anything on convex hull is considered stable
      stable_ehulls = []
      m3gnet_model = M3GNET.from_dir(args.m3gnet_model_path)
      model = m3gnet_model.Sequential
      model.add(m3gnet.Dense(units = 64, activation=LeakyReLU, input_shape=input_dim))
      model.compile(optimizer='adam', loss='categorical_crossentropy')
      model.fit(X_train, y_train, epochs=100, batch_size=70)
      with m3gnet.GradientTape() as tape:
        predictions = model(X_batch)
        loss = m3gnet.losses.categorical_crossentropy(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
def filter_unrealistic_structures(m3gnet_model, pmg_ehull, ehull_threshold=0.1):
  if pmg_ehull <= ehull_threshold:  
      stable_ehulls.append(pmg_ehull)
  else:
      print("Unrealistic structure discarded: high energy above hull.")
    return False
  
    def fit_generator(self,noise_gen, dataset, latent_dim, n_epochs = 10, n_batch = 256,n_critic = 5, model_name=None):
        bat_per_epoch = int(53856/n_batch)
        n_steps = bat_per_epoch*n_epochs
        half_batch = int(n_batch/2)
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
            print("Epoch {}/{}".format(epoch+1, n_epochs))
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                bat_per_epoch*n_batch)
            for step in range(bat_per_epoch):
 
                # Train discriminator
                with Nontrainable(self.gen):
                    for repeat in range(n_critic):
                        tens_batch, _ = generate_real_samples(dataset, n_batch)
                        noise_batch = next(noise_gen)
                        #print(tens_batch.shape)
                        disc_loss = self.disc_trainer.train_on_batch(
                            [tens_batch]+noise_batch,
                            [real_target, fake_target, gp_target]
                        )
 
                # Train generator
                with Nontrainable(self.disc):
                    noise_batch = next(noise_gen)
                    gen_loss = self.gen_trainer.train_on_batch(
                        noise_batch, real_target)
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                    if i == 0:
                        disc0LossArr.append(dl)
                    elif i == 1:
                        disc1LossArr.append(dl)
                    elif i == 2:
                        disc2LossArr.append(dl)
                losses.append(("G0", gen_loss))
                genLossArr.append(gen_loss)
                progbar.add(n_batch, 
                    values=losses)
            self.gen.save(model_name + "gen")
            self.disc.save(model_name + "critic")
            np.save(model_name + "real_loss", np.array(disc0LossArr))
            np.save(model_name + "fake_loss", np.array(disc1LossArr))
            np.save(model_name + "gp_loss", np.array(disc2LossArr))
            np.save(model_name + "generator_loss", np.array(genLossArr))

def main():
    args = parser.parse_args()
    noise_dim = 128
    critic = define_critic()
    generator = define_generator(noise_dim)
    gan_model = WGANGP(generator, critic)
    noise_gen = NoiseGenerator([noise_dim,], batch_size = args.batch_size)
    dataset = load_real_samples(args.data_path)
    gan_model.fit_generator(noise_gen, dataset, noise_dim, args.n_epochs, args.batch_size, args.n_critic, args.model_path)

if __name__ == "__main__":
    main()
