
# coding: utf-8

# # Keras based Simle/Non-Smile face recognition

# ## Setup environment


import sys
import glob
import os
print(sys.version)
print(sys.path)


# Justo for my computer


#sys.path.remove('/home/adam/Developer/kalibr_workspace/devel/lib/python2.7/dist-packages')
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


# ## Include 



from __future__ import print_function

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# ## Parameters setup


# Dataset params
num_classes = 2
img_rows, img_cols = 64, 64
channels = 3
input_shape= (img_rows, img_cols, channels)


# ## Load  Data


dataset_path = '/home/adam/Data/genki4k/'
labels_path = dataset_path + 'labels.txt'
images_path = dataset_path + 'files/'



images_files = glob.glob(images_path+'*.jpg')
images_files.sort()
print('Data len: ',len(images_files))



images = []
labels = []
val_split = 3000
test_split = 3500

# Load all images
with open(labels_path) as f:
    for line in f:
        labels.append(str(line[0]))
        

# Resize images to specific size
for file in images_files:
    img = cv2.imread(file)
    img = cv2.resize(img, (img_rows,img_cols))
    img = img[...,::-1]
    images.append(img)

# Randomly shuffle dataset
c = list(zip(images, labels))
random.shuffle(c)
images, labels = zip(*c)

images = (np.array(images))
labels = (np.array(labels))

# Split train and test datasets
x_train = images[0:val_split, :, :, :]
x_val = images[val_split:test_split, :, :, :]
x_test = images[test_split:,:,:,:]

y_train = labels[0:val_split]
y_val = labels[val_split:test_split]
y_test = labels[test_split:]

# Normalize images to interval <0.0, 1.0>
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_val /= 255.0
x_test /= 255.0


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('x_train shape: ', x_train.shape)
print('x_val shape: ', x_val.shape)
print('x_test shape: ', x_test.shape)
print('y_train shape: ', y_train.shape)
print('y_val shape: ', y_val.shape)
print('y_test shape: ', y_test.shape)


# ## Data Generator and Augmentations


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    #vertical_flip= True,
    brightness_range=(0.5, 1.2),
    fill_mode='nearest'
)

datagen.fit(x_train)


# ## Model Definition



model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


# ## Learning Phase


batch_size = 32
epochs = 50
lr = 1.0
rho = 0.95
epsilon = None
decay = 0

model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay),
             metrics=['accuracy'])



epoch_counter = 1
while True:
    
    print(' * Epoch ' + str(epoch_counter) + ' * ') 
    
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=x_train.shape[0]):
        x_batch = x_batch/255.0
       
        model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_split=0.15,
              shuffle=True )
        break
    
    val_acc = model.evaluate(x_val, y_val, verbose=0)
    print('Validation acc [loss, acc]: ' + str(val_acc))
    if val_acc[1] > 0.90 or epoch_counter >= epochs:
        break 
        
    epoch_counter += 1




model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


# ## Evaluation



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay),
             metrics=['accuracy'])
loaded_model.load_weights("model.h5")




score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




confusion_matrix = np.zeros((2,2))

for x, y in zip(x_test, y_test):

    x = x.reshape(-1,64,64,3)
    result = loaded_model.predict(x)
    confusion_matrix[np.argmax(y), np.argmax(result[0])] += 1
    
print('Confusion Matrix')
print(confusion_matrix)


# ## Results Visualization



for i, (x, y)in enumerate(zip(x_test, y_test)):
    plt.imshow(x)
    plt.show()
    print('prediction: ', model.predict(x.reshape(1,64,64,3)))
    print('ground true:', y)
    if i == 20:
        break


# ## Play Video



def logVideoMetadata(video):

    print('current pose: ' + str(video.get(cv2.CAP_PROP_POS_MSEC)))
    print('0-based index: ' + str(video.get(cv2.CAP_PROP_POS_FRAMES)))
    print('pose: ' + str(video.get(cv2.CAP_PROP_POS_AVI_RATIO)))
    print('width: ' + str(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('height: ' + str(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('fps: ' + str(video.get(cv2.CAP_PROP_FPS)))
    print('codec: ' + str(video.get(cv2.CAP_PROP_FOURCC)))
    print('frame count: ' + str(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('format: ' + str(video.get(cv2.CAP_PROP_FORMAT)))
    print('mode: ' + str(video.get(cv2.CAP_PROP_MODE)))
    print('brightness: ' + str(video.get(cv2.CAP_PROP_BRIGHTNESS)))
    print('contrast: ' + str(video.get(cv2.CAP_PROP_CONTRAST)))
    print('saturation: ' + str(video.get(cv2.CAP_PROP_SATURATION)))
    print('hue: ' + str(video.get(cv2.CAP_PROP_HUE)))
    print('gain: ' + str(video.get(cv2.CAP_PROP_GAIN)))
    print('exposure: ' + str(video.get(cv2.CAP_PROP_EXPOSURE)))
    print('convert_rgb: ' + str(video.get(cv2.CAP_PROP_CONVERT_RGB)))
    print('rect: ' + str(video.get(cv2.CAP_PROP_RECTIFICATION)))
    print('iso speed: ' + str(video.get(cv2.CAP_PROP_ISO_SPEED)))
    print('buffersize: ' + str(video.get(cv2.CAP_PROP_BUFFERSIZE)))




def hot_ent_to_text(prediction):
    print(prediction)
    if(prediction[0,0] > prediction[0,1]):
        return 'NON-SMILE'
    else:
        return 'SMILE'



video = cv2.VideoCapture()
video_path = './smile_movie.MOV'
video.open(video_path)
if not video.isOpened():
    print('Error: unable to open video: ' + video_path)

logVideoMetadata(video)



resize_ratio = 0.125
roi = [150,550,800,800]
blur_kernel_size = 5

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) )
for i in range(total_frames):
    
    ret, orig_img = video.read()
    
    if i%20 != 0:
        continue
    
    img = orig_img[roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]]
    img = cv2.blur(img, (blur_kernel_size,blur_kernel_size))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip( img, 0 )
    
    plt.imshow(img)
    plt.show()
    
    prediction = model.predict(img.reshape(1,64,64,3))
    print(hot_ent_to_text(prediction))
    print(30*'*')

