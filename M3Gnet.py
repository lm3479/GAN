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
#Applies either a convolution or transposed convolution
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
    
    def filter_unrealistic_structures(generated_structure, m3gnet_model, ehull_threshold=0.1):

    if e_above_hull <= ehull_threshold:  
        # Pass the structure to the diffuser 
        diffuser_input = prepare_for_diffuser(generated_structure)  # Adapt if needed
        diffused_output = diffuser(diffuser_input)  # Assuming you have a 'diffuser' function
        # ... potentially more processing of the diffused output ... 

    else:
        # Example options for handling unrealistic structures:

        # Option 1: Discard (simplest)
        print("Unrealistic structure discarded: High energy above hull.")

        # Option 2: Modify (more complex, requires careful modification logic)
        modified_structure = attempt_modification(generated_structure)  # You'd need to define this function
        # ... then potentially send 'modified_structure' back through the filter


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
    return model
#Forming the discriminator using dense and transposed convolutional layers
