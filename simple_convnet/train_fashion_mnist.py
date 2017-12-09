# From https://www.kaggle.com/bugraokcu/cnn-with-keras/notebook
#
# Differences:
# - Use Keras-provided data loading library instead of pandas
# - Work with Keras config (~/.keras/keras.json) of "image_data_format": "channels_first"
# - Minor changes of hyper-parameters (e.g. proportion of validation data in original training set)
# - Bug fix in the last loop to display feature maps

import keras
from keras.utils.np_utils import to_categorical
# from keras.utils.np_utils import to_categorical
from keras.datasets import fashion_mnist, mnist
import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split

# Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
# Label  Description
# ------------------
#   0    T-shirt/top
#   1    Trouser
#   2    Pullover
#   3    Dress
#   4    Coat
#   5    Sandal
#   6    Shirt
#   7    Sneaker
#   8    Bag
#   9    Ankle boot

# Program parameters

COMPLEX_MODE = False
# COMPLEX_MODE = True
SHOW_REPORT  = False
# EPOCH        = 50
# EPOCH        = 100
#EPOCH        = 150
EPOCH        = 200

# data_train = pd.read_csv('../input/fashion-mnist_train.csv')
# data_test = pd.read_csv('../input/fashion-mnist_test.csv')

from complexnn import ComplexConv2D

(X_train_org, y_train_org), (X_test_org, y_test_org) = fashion_mnist.load_data()
# (X_train_org, y_train_org), (X_test_org, y_test_org) = mnist.load_data()
X_train_org = np.expand_dims(X_train_org, axis=1)
X_test_org  = np.expand_dims(X_test_org,  axis=1)

img_rows, img_cols = 28, 28
input_shape = (1, img_rows, img_cols)

y_train_cat = to_categorical(y_train_org, 10)
y_test_cat  = to_categorical(y_test_org,  10)

# X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_cat, test_size=0.2, random_state=13)
X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_cat, test_size=0.1, random_state=13)
X_test = X_test_org
y_test = y_test_cat
# 	Y_train    = to_categorical(y_train_split, nb_classes)
# 	Y_val      = to_categorical(y_val_split,   nb_classes)
# 	Y_test     = to_categorical(y_test,        nb_classes)
    
#Here we split validation data to optimiza classifier during training

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

#Test data
# X_test = np.array(data_test.iloc[:, 1:])
# y_test = to_categorical(np.array(data_test.iloc[:, 0]))



# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
# X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255


# In[3]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
# from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from complexnn import ComplexConv2D, ComplexBN, ComplexDense

batch_size = 256
num_classes = 10
epochs = EPOCH
# epochs = 100
# epochs = 1  # just test

#input image dimensions
img_rows, img_cols = 28, 28

channelAxis = 1 # 'channels_first' mode only
filsize = (3, 3)
convArgs = {
    "padding": "same",
    "use_bias": False,
    "kernel_regularizer": keras.regularizers.l2(0.0001),
}
bnArgs = {
    "axis": channelAxis,
    "momentum": 0.9,
    "epsilon": 1e-04
}

model = Sequential()
# if COMPLEX_MODE:
#     model.add(ComplexConv2D(32, (3, 3),
#                             kernel_initializer='he_normal',
#                             input_shape=input_shape,
#                             **convArgs))
# else:
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  kernel_initializer='he_normal',
#                  input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(BatchNormalization(**bnArgs))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
if COMPLEX_MODE:
    # model.add(ComplexConv2D(32, (3, 3)))
    model.add(ComplexConv2D(32, (3, 3), **convArgs))
    model.add(ComplexBN(**bnArgs))
else:
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Conv2D(32*2, (3, 3)))
    model.add(Conv2D(32 * 2, (3, 3), **convArgs))
    model.add(BatchNormalization(**bnArgs))
model.add(Activation('relu'))
# O = ComplexConv2D(sf, filsize, name='conv1', **convArgs)(O)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
if COMPLEX_MODE:
    # model.add(ComplexConv2D(64, (3, 3)))
    model.add(ComplexConv2D(64, (3, 3), **convArgs))
    model.add(ComplexBN(**bnArgs))
else:
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Conv2D(64*2, (3, 3)))
    model.add(Conv2D(64 * 2, (3, 3), **convArgs))
    model.add(BatchNormalization(**bnArgs))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
if COMPLEX_MODE:
    # model.add(ComplexConv2D(64, (3, 3)))
    model.add(ComplexConv2D(128, (3, 3), **convArgs))
    model.add(ComplexBN(**bnArgs))
else:
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(Conv2D(64*2, (3, 3)))
    model.add(Conv2D(128 * 2, (3, 3), **convArgs))
    model.add(BatchNormalization(**bnArgs))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
if COMPLEX_MODE:
    model.add(ComplexDense(64))
else:
    model.add(Dense(64 * 2))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
plot_model(model, to_file='./model.png')

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


if SHOW_REPORT == False:
    exit()

# ### Classification Report
# We can summarize the performance of our classifier as follows

# In[8]:


#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

#get the indices to be plotted
y_true = y_test_org
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]


# In[9]:


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))


# It's apparent that our classifier is underperforming for class 6 in terms of both precision and recall. For class 2, classifier is slightly lacking precision whereas it is slightly lacking recall (i.e. missed) for class 4.
# 
# Perhaps we would gain more insight after visualizing the correct and incorrect predictions.

# Here is a subset of correctly predicted classes.

# In[10]:


for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()
plt.show()


# And here is a subset of incorrectly predicted classes.

# In[11]:


for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()

plt.show()


# It looks like diversity of the similar patterns present on multiple classes effect the performance of the classifier although CNN is a robust architechture. A jacket, a shirt, and a long-sleeve blouse has similar patterns: long sleeves (or not!), buttons (or not!), and so on.

# #### What did the network learn?
# 
# The snippets are taken from _Chollet, F (2017)_. The idea is the give an input data and visualize the activations of the conv layers.

# In[12]:


test_im = X_train[13]
plt.imshow(test_im.reshape(28,28), cmap='gray', interpolation='none')

plt.show()


# Let's see the activation of the 2nd channel of the first layer:

# In[13]:


from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(input=model.input, output=layer_outputs)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_im.reshape(1,1,28,28))

first_layer_activation = activations[0]
# plt.matshow(first_layer_activation[0, :, :, 4], cmap='gray')
plt.matshow(first_layer_activation[0, 4, :, :], cmap='gray')

plt.show()

# Let's plot the activations of the other conv layers as well.

# In[14]:


layer_names = []
for layer in model.layers[:-1]:
    if isinstance(layer, Conv2D):
        layer_names.append(layer.name)
images_per_row = 16
# for layer_name, layer_activation in zip(layer_names, activations):
for layer_name, layer_activation in zip(layer_names, activations[::2]): # Skip intermediate maxpool activations
    # n_features = layer_activation.shape[-1]
    n_features = layer_activation.shape[1]
    # size = layer_activation.shape[1]
    size = layer_activation.shape[2]
    n_cols = n_features / images_per_row
    n_cols = int(n_cols)
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            # channel_image = layer_activation[0,:, :, col * images_per_row + row]
            channel_image = layer_activation[0, col * images_per_row + row, :, :]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='bone')

plt.show()

