#The original GAN (Goodfellow et al.) proposed that the discriminator's parameters should be updated several times before each single step of generator
#The cycle of the generator moving and discriminator following, with alternating gradient descent and a fixed learning rate is known to suffer mode collapse
#In an unrolled GAN, this is avoided
#The generator's feedback for backprop take into account HOW the discriminator will respond
#The generator will try to make steps that the discriminator will have a hard time responding to
#This helps the generator spread its mass to make the next discriminator step less effective, as opposed to collapsing
#Simply put, unrolled GANs reduce the probability of mode collapse by playing 'k' number of steps for how the generator can be optimized

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
unrolled_d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

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
              #Units = number of kernels
              #Filter: tuple of three to define kernel size
              #Stride: tuple of how kernel will move
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
#Discriminator -> convolution -> LeakyReLU
#Generator -> transposed convolution -> ReLU
#Returns keras tensor following convolution, batchnorm, and activation function

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
#Discriminator -> LeakyReLU
#Generator -> ReLU

def define_discriminator(in_shape: Tuple[int, int, int, int] = (64, 64, 4, 1),
ehull_input_shape: Tuple[int, int, int, int] = (64, 64, 4, 1)) -> keras.engine.functional.Functional:
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
    hull_in = Input(shape=(e_hull_input_shape, ), name="ehull_input")
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

#Upcoming: Wasserstein loss
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
      m3gnet_model = M3GNET.from_dir(args.m3gnet_model_path)
      stable_ehulls = []
def filter_unrealistic_structures(m3gnet_model, pmg_ehull, ehull_threshold=0.1):
  if pmg_ehull <= ehull_threshold:  
      stable_ehulls.append(pmg_ehull)
  else:
      print("Unrealistic structure discarded: high energy above hull.")
    return False
  
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
    x_input = generate_latent_points(latent_dim,n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y

def define_critic(in_shape = (64, 64, 4, 1)
) -> keras.engine.functional.Functional:
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
                   for k in range(unroll_steps):
                        # Generate fake samples
                        X_gan = generate_latent_points(latent_dim, n_batch)
                        y_gan = np.ones((n_batch, 1))
                        # Calculate generator loss including e_hull
                        g_loss = gan_model.train_on_batch(X_gan, y_gan)
                        X_fake_filtered, _ = generate_fake_samples(g_model, latent_dim, n_batch)
                        pmg_ehull = [sample[1] for sample in X_fake_filtered]  # Extracting e_hulls
                        pmg_ehull = np.array(pmg_ehull)
                        e_hull_loss = np.mean(pmg_ehull)  # Using mean e_hull as the loss
                        g_loss += e_hull_loss  # Add e_hull loss to generator's loss
                        total_g_loss = 0.5 * g_loss + 0.5 * e_hull_loss  # Combining the losses
                        g_model.save(os.path.join(save_path, 'generator'))
                        d_model.save(os.path.join(save_path, 'discriminator'))
                        np.savetxt(os.path.join(save_path, 'd_loss_real_list'),d_loss_real_list)
                        np.savetxt(os.path.join(save_path, 'd_loss_fake_list'),d_loss_fake_list)
                        np.savetxt(os.path.join(save_path, 'g_loss_list'),g_loss_list)
#Compiling GAN model, specifying losses so model can perform backprop accordingly

def main(m3gnet_model: M3GNET):
    args = parser.parse_args()
    predict_ehull(args.dir, args.m3gnet_model_path, args.ehull_path, args.mp_api_key)
    latent_dim = 128
    discriminator = define_discriminator()
    generator = define_generator(latent_dim)
    gan_model = define_gan(generator,discriminator)
    dataset = load_real_samples(args.data_path)
    train(generator, discriminator, gan_model,dataset, latent_dim, args.save_path)
#Putting the layers together, constructing final GAN

if __name__ == "__main__":
  m3gnet_model = M3GNET.from_dir(args.m3gnet_model_path)
    main(m3gnet_model)
def train(g_model: keras.engine.functional.Functional,
          d_model: keras.engine.functional.Functional,
          gan_model: keras.engine.functional.Functional,
          dataset: np.ndarray, latent_dim: int, save_path: str,
          n_epochs: int = 100, n_batch: int = 64) -> None:
    # Trains the GAN over 100 epochs, each containing 64 examples
    bat_per_epoch = int(dataset.shape[0]/n_batch)
    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []

for epoch in range(epochs):
    for image_batch in dataset:
        images = image_batch
        noise = tf.random.normal([batch_size, noise_dim])
        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        unrolled_steps = 5
        for i in range(unrolled_steps):
            with tf.GradientTape() as tape:
                fake_images = generator(tf.random.normal([batch_size, noise_dim]), training=True)
                real_output = discriminator(images, training=True)
                fake_output = discriminator(fake_images, training=True)
                d_loss = discriminator_loss(real_output, fake_output)
            gradients = tape.gradient(d_loss, discriminator.trainable_variables)
            unrolled_d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
            noise = tf.random.normal([batch_size, noise_dim])
      
    for i in range(n_epochs):
        for j in range(bat_per_epoch//2):
            #Real samples from the dataset
            X_real,y_real = generate_real_samples(dataset, n_batch)
            d_loss_real,_ = d_model.train_on_batch(X_real, y_real)
            #Fake samples from generator
            X_fake,y_fake = generate_fake_samples(g_model, latent_dim, n_batch)
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)
            #Train generator to fool discriminator
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            g_loss = gan_model.train_on_batch(X_gan,y_gan)
            #Calculating e_hull loss for generator, which will make up 50% of the total feedback
            X_fake_filtered, _ = generate_fake_samples(g_model, latent_dim, n_batch)
            pmg_ehull = [sample[1] for sample in X_fake_filtered]  # Extracting e_hulls
            pmg_ehull = np.array(pmg_ehull)
            e_hull_loss = np.mean(pmg_ehull)  # Using mean e_hull as the loss
            g_loss += e_hull_loss  # Add e_hull loss to generator's loss
            total_g_loss = 0.5 * g_loss + 0.5 * e_hull_loss #Partially from regular discriminator feedback & partially from e_hull
            d_loss_real_list.append(d_loss_real)
            d_loss_fake_list.append(d_loss_fake)
            g_loss_list.append(g_loss)
            g_model.save(os.path.join(save_path, 'generator'))
            d_model.save(os.path.join(save_path, 'discriminator'))
            np.savetxt(os.path.join(save_path, 'd_loss_real_list'),d_loss_real_list)
            np.savetxt(os.path.join(save_path, 'd_loss_fake_list'),d_loss_fake_list)
            np.savetxt(os.path.join(save_path, 'g_loss_list'),g_loss_list)
