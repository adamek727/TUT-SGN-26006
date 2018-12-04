
# coding: utf-8

# # Keras based GNA mnist generating

# https://github.com/soumith/ganhacks
# https://arxiv.org/pdf/1406.2661.pdf
# https://arxiv.org/pdf/1511.06434.pdf

# ## Setup environment



import sys
import glob
print(sys.version)
print(sys.path)




sys.path.remove('/home/adam/Developer/kalibr_workspace/devel/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# ## Include



from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Reshape, UpSampling2D, Conv2DTranspose, LeakyReLU
from keras import optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras import initializers
from keras.layers import Input
from keras.models import Model

import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import math
import random
from tqdm import tqdm


# ## Parameters setup



img_rows, img_cols = 28, 28
channels = 1
input_shape= (img_rows, img_cols, channels)


# ## Load  Data



mnist_path = '/home/adam/Data/mnist/'

x_train, y_train = loadlocal_mnist(
                images_path=mnist_path+'/train-images.idx3-ubyte', 
                labels_path=mnist_path+'/train-labels.idx1-ubyte'
            )

x_train = x_train.reshape(x_train.shape[0], int(math.sqrt(x_train.shape[1])), int(math.sqrt(x_train.shape[1])))
x_train = (x_train.astype(np.float32) / 128) - 1




print(x_train.shape, y_train.shape)




plt.imshow(x_train[0,:,:])
plt.show()


# ## Model Definition

# ### Generator



leaky_alpha = 0.2
noise_size = 100

generator = Sequential()
generator.add(Dense(128*7*7, input_dim=noise_size, kernel_initializer=initializers.RandomNormal(stddev=0.01)))
generator.add(LeakyReLU(leaky_alpha))

generator.add(Reshape((7, 7, 128)))

generator.add(UpSampling2D(size=(2, 2)))

generator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
generator.add(LeakyReLU(leaky_alpha))

generator.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
generator.add(LeakyReLU(leaky_alpha))

generator.add(UpSampling2D(size=(2, 2)))

generator.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
generator.add(LeakyReLU(leaky_alpha))

generator.add(Conv2D(1, kernel_size=(3, 3), padding='same', activation='tanh'))

print(generator.summary())


# ### Discriminator



leaky_alpha = 0.2
dropout = 0.3

discriminator = Sequential()
discriminator.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=(28, 28,1), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(leaky_alpha))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
discriminator.add(LeakyReLU(leaky_alpha))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
discriminator.add(LeakyReLU(leaky_alpha))
discriminator.add(Dropout(dropout))

discriminator.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(leaky_alpha))
discriminator.add(Dropout(dropout))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=5e-4, beta_1=0.5))

print(discriminator.summary())


# ### GAN



# Combined network
discriminator.trainable = False
ganInput = Input(shape=(noise_size,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5))


# ## Learning Phase



epochs=50
batchSize=128
g_loss_hist = []
d_loss_hist = []
soft_up = 0.9
soft_down = 0.1

no_of_batches = int(x_train.shape[0] / batchSize)

for e in range(epochs):
    
    print(' * Epoch: ' + str(e+1))
    
    for b in tqdm(range(no_of_batches)):
        
        
        noise = np.random.normal(0, 1, size=[batchSize, noise_size]) # noise for generating fake images
        fake_images = generator.predict(noise) # generate fake images
        real_images = x_train[np.random.randint(0, x_train.shape[0], size=batchSize)] # pick random real images

        # Train Discriminator
        discriminator.trainable = True
        
        x = np.concatenate([real_images.reshape(batchSize,28,28,1), fake_images]) 
        y = np.ones(2*batchSize) * soft_down
        y[:batchSize] = soft_up
        
        dloss = discriminator.train_on_batch(x, y)
        
        # Train Ganerator
        discriminator.trainable = False
        y = np.ones(batchSize)
        noise = np.random.normal(0, 1, size=[batchSize, noise_size])
        gloss = gan.train_on_batch(noise, y)

    # epoch results
    print(' D_loss: ' + str(dloss) + ' G_loss: ' + str(gloss))
    d_loss_hist.append(dloss)
    g_loss_hist.append(gloss)

    noise = np.random.normal(0, 1, size=[1, noise_size])
    plt.imshow(generator.predict(noise).reshape(28,28))
    plt.show()
    
    if e % 5 == 10:
        generator.save('models/generator_epoch_' + str(epoch).zfill(5) + '.h5')
        discriminator.save('models/discriminator_epoch_' + str(epoch).zfill(5) + '.h5')


# ## Evaluation

# ### Generator Results



def plot_results(real_imgs, fake_imgs):
    
    for real, fake in zip(real_imgs, fake_imgs):
        plt.imshow(real)
        plt.show()
        print(' REAL ')
        print(10*'*')
        
        plt.imshow(fake)
        plt.show()
        print(' FAKE ')
        print(10*'*')




real_imgs = []
fake_imgs = []
n = 5

for i in range(n):
    
    real_imgs.append(x_train[random.randint(0,x_train.shape[0]-1)])
    
    noise = np.random.uniform(-1,1,size=(1,100))
    fake_imgs.append(generator.predict(noise).reshape(28,28))

plot_results(real_imgs, fake_imgs)


# ### Discriminator results



confusion_matrix = np.zeros((2,2))

test_batch_size = 100

noise = np.random.normal(0, 1, size=[test_batch_size, noise_size]) 
fake_images = generator.predict(noise)
real_images = x_train[np.random.randint(0, x_train.shape[0], size=test_batch_size)] 

x_test = np.concatenate([real_images.reshape(test_batch_size,28,28,1), fake_images]) 
y_test = np.zeros(2*test_batch_size)
y_test[:test_batch_size] = np.ones(test_batch_size)

for x, y in zip(x_test, y_test):

    result = discriminator.predict(x.reshape(1,28,28,1))
    confusion_matrix[int(y), int(np.round(result[0]).item())] += 1
    
print('Confusion Matrix')
print(confusion_matrix)

