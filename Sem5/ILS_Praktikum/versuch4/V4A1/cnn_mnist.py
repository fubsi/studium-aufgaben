# ISML, Veruch 4, Aufg.1 
#############################
# for additional information/tutorial see: https://victorzhou.com/blog/keras-cnn-tutorial/
# see also ~/hs-albsig/02-research/Buch_LernendeSysteme_Springer/manuscript/LernendeSystemeKnoblauch/python/keras_beispiele

import numpy as np
import tensorflow as tf
import datetime
import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tf_keras.utils import to_categorical
from tf_keras.optimizers import SGD, Adam
from tf_keras.datasets.mnist import load_data

# (i) Load image data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
(train_images, train_labels), (test_images, test_labels) = load_data()

# (ii) Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# (iii) Reshape the images.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# (iv) Define Network and Training Hyper-Parameters.
num_filters = 4
filter_size = 3
pool_size = 2
eta = 1e-4
opt_alg=SGD(learning_rate=eta)
batchsize=50
epochs=5

# (v) Build the network model.
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1), name='layers_conv'),
  MaxPooling2D(pool_size=pool_size,name='layers_maxpool'),
  Flatten(name='layers_flatten'),
  Dense(10, activation='softmax',name='layers_dense'),
])

# (vi) Compile the model.
model.compile(
  opt_alg,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# (vii) Train the model.
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(
    train_images,
    to_categorical(train_labels),
    batch_size=batchsize,
    epochs=epochs,
    validation_data=(test_images, to_categorical(test_labels)),
    callbacks=[tensorboard_callback]
)

# (viii) Save the model to disk.
model.save_weights('cnn.h5')

# Load the model from disk later using:
# model.load_weights('cnn.h5')

# (ix) Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# (x) Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# (xi) Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]

# (xii) view results using tensorboard: (type this into command shell, and then click the printed link)
# tensorboard --logdir logs/fit
# tensorboard --logdir=logs/fit --host localhost --port 8088
