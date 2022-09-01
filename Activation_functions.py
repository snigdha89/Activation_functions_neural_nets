import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models,optimizers
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization,Dense, Activation
from keras.models import Model,Sequential
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
import math

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Split them into train & test
x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

print("x_train.shape:", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape", y_test.shape)

# number of classes
num_classes = len(set(y_train))
K = num_classes
print("number of classes:", num_classes)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

## Sample Training images 
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = x_train[i]
    axarr[i].imshow(img)
plt.show()

def mlp_model_withoutBatch(actfunction):
  model = Sequential()
  model.add(Flatten(input_shape = x_train.shape[1:]))
  model.add(Dense(1000, activation=actfunction))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation=actfunction))
  model.add(Dropout(0.2))
  model.add(Dense(num_classes, activation='softmax'))

  return model

"""**STEP DECAY Learning Rate with SGD**"""

# define MLP model
model1wob = mlp_model_withoutBatch('relu')

# define SGD optimizer
momentum = 0.5
sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False) 

# define step decay function
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# compile the model
model1wob.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=sgd, metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# fit the model
history1 = model1wob.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=10, 
                     batch_size=64, 
                     callbacks=callbacks_list, 
                     verbose=2)

loss, acc = model1wob.evaluate(x_test,y_test)
print('Accuracy of SGD Basemodel with Step Decay and without Batch Normalization is: %.3f' % (acc * 100.0))

"""**EXPONENTIAL DECAY Learning Rate with SGD**"""

# define MLP model
model2wob = mlp_model_withoutBatch('relu')

# define SGD optimizer
momentum = 0.5
sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False) 

# define step decay function
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# compile the model
model2wob.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=sgd, metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# fit the model
history2 = model2wob.fit(x_train, y_train, 
                     validation_data=(x_test, y_test), 
                     epochs=10, 
                     batch_size=64, 
                     callbacks=callbacks_list, 
                     verbose=2)

loss, acc = model2wob.evaluate(x_test,y_test)
print('Accuracy of SGD Basemodel with Exponential Decay and without Batch Normalization is: %.3f' % (acc * 100.0))

"""**ADAM WITH NO LEARNING RATE SCHEDULER and NO BATCH NORMALISATION**"""

# define MLP model
model3wob = mlp_model_withoutBatch('relu')
model3wob.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model3wob.fit(x_train, y_train, epochs=10)

loss, acc = model3wob.evaluate(x_test,y_test)
print('Accuracy of Adam Basemodel with Dropout and without Batch Normalization is: %.3f' % (acc * 100.0))

"""**MLP MODEL WITH BATCH NORMALISATION**"""

def mlp_model(actfunction):
  model = Sequential()
  model.add(Flatten(input_shape = x_train.shape[1:]))
  model.add(BatchNormalization())
  model.add(Dense(1000, activation=actfunction))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(Dense(512, activation=actfunction))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Dense(num_classes, activation='softmax'))
  return model

"""**ADAM  with Batch Normalisation AND RELU**"""

# define MLP model
model1wb = mlp_model('relu')

model1wb.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1wb.fit(x_train, y_train, epochs=10)

loss, acc = model1wb.evaluate(x_test,y_test)
print('Accuracy of ADAM Base Model with Dropout and Batch Normalization(RELU) is: %.3f' % (acc * 100.0))

"""**ADAM with Batch Normalisation and TanH**"""

# define MLP model
model2wb = mlp_model('tanh')

model2wb.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2wb.fit(x_train, y_train, epochs=10)

loss, acc = model2wb.evaluate(x_test,y_test)
print('Accuracy of ADAM Base Model with Dropout and Batch Normalization(TanH) is: %.3f' % (acc * 100.0))

"""**ADAM with Batch Normalisation and ELU**"""

# define MLP model
model3wb = mlp_model('elu')

model3wb.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model3wb.fit(x_train, y_train, epochs=10)

loss, acc = model3wb.evaluate(x_test,y_test)
print('Accuracy of ADAM Base Model with Dropout and Batch Normalization(ELU) is: %.3f' % (acc * 100.0))

"""**ADAM  with Batch Normalisation and LeakyRelu(Alpha = 0.3)**"""

# define MLP model
val = tf.keras.layers.LeakyReLU(alpha=0.3)
model4wb = mlp_model(val)

model4wb.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model4wb.fit(x_train, y_train, epochs=10)

loss, acc = model4wb.evaluate(x_test,y_test)
print('Accuracy of ADAM Base Model with Dropout and Batch Normalization(LeakyRelu- alpha = 0.3) is: %.3f' % (acc * 100.0))

"""**ADAM  with Batch Normalisation and LeakyRelu(Alpha = 0.25)**"""

# define MLP model
val = tf.keras.layers.LeakyReLU(alpha=0.25)
model5wb = mlp_model(val)

model5wb.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model5wb.fit(x_train, y_train, epochs=10)

loss, acc = model5wb.evaluate(x_test,y_test)
print('Accuracy of ADAM Base Model with Dropout and Batch Normalization(LeakyRelu -alpha = 0.25) is: %.3f' % (acc * 100.0))

"""**IMPROVING THE ACCURACY BY DATA AUGMENTATION**"""

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
r = model3wb.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=10)

loss, acc = model3wb.evaluate(x_test, y_test)

print('Accuracy of Basemodel with Data Augmentation is: %.3f' % (acc * 100.0))

# Plot loss per iteration
plt.rcParams['figure.figsize'] = [10,5]
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

from sklearn.metrics import confusion_matrix
p_test = model3wb.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)

cm

misclassified_idx = np.where(p_test == y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (classes[y_test[i]], classes[p_test[i]]));

misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (classes[y_test[i]], classes[p_test[i]]));