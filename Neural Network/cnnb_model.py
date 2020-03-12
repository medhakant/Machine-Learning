import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import sgd, rmsprop
from keras.callbacks import LearningRateScheduler
import sys

param = sys.argv[1:]

data = pd.read_csv(param[0],sep=' ',header=None).values
X_train = data[:,0:-1]
Y_train = data[:,-1]
X_train = X_train.reshape(50000,3,32,32)
X_train = np.transpose(X_train, (0, 2, 3,1))
mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
Y_train = to_categorical(Y_train)

batch_size = 64
num_epoch = 2

model = Sequential()

#model layers
model.add(Conv2D(32, kernel_size=3,strides=1,kernel_regularizer=regularizers.l2(1e-4), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Conv2D(32, kernel_size=3,strides=1,kernel_regularizer=regularizers.l2(1e-4), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=3,strides=1,kernel_regularizer=regularizers.l2(1e-4), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Conv2D(64, kernel_size=3,strides=1,kernel_regularizer=regularizers.l2(1e-4), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3,strides=1, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Conv2D(128, kernel_size=3,strides=1, padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile
model.compile(optimizer=rmsprop(lr=0.001,decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

#datagenrator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(X_train)


def get_lr(epoch):
    lrate = 0.001
    if epoch > 20:
        lrate = 0.0008
    elif epoch > 30:
        lrate = 0.0007        
    elif epoch > 40:
        lrate = 0.0006
    elif epoch > 50:
        lrate = 0.0005       
    return lrate

#fit model
history1 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch= X_train.shape[0] // batch_size,epochs=num_epoch,shuffle=True,use_multiprocessing=True,callbacks=[LearningRateScheduler(get_lr)])

def get_lr(epoch):
    lrate = 0.0005
    if epoch > 20:
        lrate = 0.0004
    elif epoch > 30:
        lrate = 0.0003        
    elif epoch > 40:
        lrate = 0.0002       
    return lrate

#fit model
history2 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch= X_train.shape[0] // batch_size,epochs=num_epoch,shuffle=True,use_multiprocessing=True,callbacks=[LearningRateScheduler(get_lr)])

def get_lr(epoch):
    lrate = 0.0002    
    return lrate

#fit model
history3 = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch= X_train.shape[0] // batch_size,epochs=num_epoch,shuffle=True,use_multiprocessing=True,callbacks=[LearningRateScheduler(get_lr)])

test = pd.read_csv(param[1],sep=' ',header=None).values
X_test = test[:,0:-1]
Y_test = test[:,-1]
X_test = X_test.reshape(10000,3,32,32)
X_test = np.transpose(X_test, (0, 2, 3,1))
X_test = (X_test-mean)/(std+1e-7)
pred = np.argmax(model.predict(X_test),axis=1).astype(int)
np.savetxt(param[2],pred,delimiter="\n")

import matplotlib.pyplot as plt
plt.plot(history1.history['acc'])
plt.xlabel('epoch')
plt.plot(history1.history['loss'])
plt.title('Model Loss/Accuracy')
plt.ylabel('loss/accuracy')
plt.legend(['accuracy','loss'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('cifar10.png', dpi=100)