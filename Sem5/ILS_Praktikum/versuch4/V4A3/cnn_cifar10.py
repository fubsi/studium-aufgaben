# The full CNN code!
####################
# from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
import numpy as np
import tensorflow as tf
import datetime
#import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD, Adam

# ********************************************************************
# (i) load and preprocess data
# ********************************************************************

# load dataset
(trainX, trainY_labels), (testX, testY_labels) = cifar10.load_data()

# one hot encode target values
trainY = to_categorical(trainY_labels)
testY = to_categorical(testY_labels)

# convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')

# normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))

# ********************************************************************
# (ii) define (neural network) model: baseline model = 3 VGG blocks
# ********************************************************************
model = Sequential()
# VGG block 1
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3), name='layer_vgg1_conv1'))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='layer_vgg1_conv2'))
model.add(MaxPooling2D((2, 2), name='layer_vgg1_maxpool'))
# VGG block 1
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='layer_vgg2_conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='layer_vgg2_conv2'))
model.add(MaxPooling2D((2, 2), name='layer_vgg2_maxpool'))
# VGG block 1
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='layer_vgg3_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='layer_vgg3_conv2'))
model.add(MaxPooling2D((2, 2), name='layer_vgg3_maxpool'))
# plus finally two dense layers 
model.add(Flatten(name='layer_flatten'))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='layer_dense'))
model.add(Dense(10, activation='softmax', name='layer_output'))            # output-layer for 10 classes 

# ********************************************************************
# (iii) compile and train model 
# ********************************************************************
# compile model
opt = SGD(lr=0.00001, momentum=0.1)    # define optimizer
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])   

# train model
log_dir = "logs/fit/cnn_cifar10_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history=model.fit(
    trainX,
    trainY,
    batch_size=500,
    epochs=5,
    validation_data=(testX,testY), 
    callbacks=[tensorboard_callback],
)
#history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)

# Save the model to disk.
model.save_weights('cnn_cifar10.h5')

# Load the model from disk later using:
# model.load_weights('cnn_cifar10.h5')

# Predict on the first 5 test images.
predictions = model.predict(testX[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(testY_labels[:5]) # [7, 2, 1, 0, 4]

# view results using tensorboard:
# tensorboard --logdir logs/fit
# tensorboard --logdir=logs/fit --host localhost --port 8088
