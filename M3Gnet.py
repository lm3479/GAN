import os
import argparse
import numpy as np
from typing import Tuple
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Flatten, ReLU, LeakyReLU, BatchNormalization, Reshape
from sklearn.cluster import KMeans
from pymatgen.io.cif import CifWriter
from pymatgen.core import Lattice, Structure
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to where the dataset is stored.")
parser.add_argument("--save_path", type=str, help="Path to where the models and loss history is stored.")
parser.add_argument("--dir", type=str, help="Path to where crystals are.")
parser.add_argument("--m3gnet_model_path", type=str, help="Path to where M3GNet model is.")
parser.add_argument("--ehull_path", type=str, help="Path to where energy above calculations will be stored.")
parser.add_argument("--mp_api_key", type=str, help="API key for materials project.")

def conv_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
              filter: Tuple[int, int, int], stride : Tuple[int, int, int], 
              discriminator: bool = True
              ) -> keras.engine.keras_tensor.KerasTensor:
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    activation_function = ReLU()
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = conv(x)
  x = BatchNormalization()(x)
  x = activation_function(x) 
  return x
#Convolutional layer
#If convolution, LeakyReLU
#If transposed convolution, ReLU
#Returns keras tensor following convolution, normalization, and activation function

def dense_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
               discriminator: bool) -> keras.engine.keras_tensor.KerasTensor:
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
  else:
    activation_function = ReLU()
  x = Dense(units)(x)
  x = BatchNormalization()(x)
  x = activation_function(x)
  return x
#Dense layer
#Applying a dense layer, normalization, and an activation function
#If dense_norm is present in discriminator, LeakyReLU
#If dense_norm is not present, ReLU

def define_discriminator(in_shape: Tuple[int, int, int, int] = (64, 64, 4, 1)
) -> keras.engine.functional.Functional:
 
    tens_in = Input(shape=in_shape, name="input")
 
    y = Flatten()(tens_in)
    y = dense_norm(y, 1024, True) 
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)

    x = conv_norm(tens_in, 32, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 64, (3,3,1), (1,1,1), True) 
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)

    z = Reshape((32, 32, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 9, True) 
 
    x = conv_norm(x, 128, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 256, (2,2,2), (2,2,2), True)  
 
    z = Reshape((3, 3, 1, 1))(y)
    x = z + x
 
    x = Flatten()(x)
    x = Dropout(0.25)(x)
 
    disc_out = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs=tens_in, outputs=disc_out)
    opt = Adam(learning_rate = 1e-5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt,metrics = ['accuracy'])
    return model
#Forming discriminator using dense and convolutional layers 

def define_generator(latent_dim: int) -> keras.engine.functional.Functional:
    n_nodes = 16 * 16 * 4
 
    noise_in = Input(shape=(latent_dim, ), name="noise_input")

    y = dense_norm(noise_in, 484, False)
    y = dense_norm(y, 484, False)
    
    x = dense_norm(noise_in, n_nodes, False)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 256, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
 
    z = Reshape((22, 22, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
 
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
 
    z = Reshape((28, 28, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 1024, False)
    y = dense_norm(y, 1024, False)
 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
 
    z = Reshape((32, 32, 1, 1))(y)
    x = z + x

    y = dense_norm(y, 4096, False)
 
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)   
 
    z = Reshape((64, 64, 1, 1))(y)
    x = z + x
    outMat = Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid')(x)
    model = Model(inputs=noise_in, outputs=outMat)
    m3gnet_model = M3GNET.from_dir(args.m3gnet_model_path)
    def filter_unrealistic_structures(m3gnet_model, ehull_threshold=0.1):
      if e_above_hull <= ehull_threshold:  
         return True
      else:
          print("Unrealistic structure discarded: High energy above hull.")
        return False 
      return model
  
def define_gan(generator: keras.engine.functional.Functional, 
               discriminator: keras.engine.functional.Functional
               ) -> keras.engine.functional.Functional:
    discriminator.trainable = False
#Freezing discriminator weights
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)
    
    opt = Adam(learning_rate = 1e-5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt)
    return model
#Creating finished model
#Compiling optimizer (Adam, SGD variant) and loss function (binary cross-entropy)

def load_real_samples(data_path: str) -> np.ndarray:
    data_tensor = np.load(data_path)
    return np.reshape(data_tensor, (data_tensor.shape[0], 64, 64, 4))
#Loads in the tensor of real samples, which have the shape (x, 64, 64, 4)

                          ) -> Tuple[np.ndarray, np.ndarray]:
    ix = random.randint(0,dataset.shape[0],n_samples)
    X = dataset[ix]

y = np.ones((n_samples,1))
    return X,y
#Selects random values and indicates that they're true (they're from the dataset as opposed to the generator)

def generate_latent_points(latent_dim: int, n_samples:int) -> np.ndarray:
    x_input = random.randn(latent_dim*n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input
#Random array to be used

def generate_fake_samples(generator: keras.engine.functional.Functional, 
                          latent_dim: int, n_samples: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,0))
    return X,y
#Generates fake examples for the generator

def train(g_model: keras.engine.functional.Functional,
          d_model: keras.engine.functional.Functional,
          gan_model: keras.engine.functional.Functional,
          dataset: np.ndarray, latent_dim: int, save_path: str,
          n_epochs: int = 100, n_batch: int = 64) -> None:
#Trains the GAN over 100 epochs, each containing 64 examples

    bat_per_epoch = int(dataset.shape[0]/n_batch)
    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []
    for i in range(n_epochs):
        for j in range(bat_per_epoch//2):
            X_real,y_real = generate_real_samples(dataset, n_batch)
            d_loss_real,_ = d_model.train_on_batch(X_real, y_real)
            X_fake,y_fake = generate_fake_samples(g_model, latent_dim, n_batch)
            d_loss_fake,_ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
        
        d_loss_real_list.append(d_loss_real)
        d_loss_fake_list.append(d_loss_fake)
        g_loss_list.append(g_loss)

        g_model.save(os.path.join(save_path, 'generator'))
        d_model.save(os.path.join(save_path, 'discriminator'))
        np.savetxt(os.path.join(save_path, 'd_loss_real_list'),d_loss_real_list)
        np.savetxt(os.path.join(save_path, 'd_loss_fake_list'),d_loss_fake_list)
        np.savetxt(os.path.join(save_path, 'g_loss_list'),g_loss_list)
#Compiling GAN model, specifying losses so model can perform backprop accordingly

