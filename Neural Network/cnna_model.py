import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
import sys

param = sys.argv[1:]

data = pd.read_csv(param[0],sep=' ',header=None).values
X_train = data[:,0:-1]
Y_train = data[:,-1]
X_train = X_train.reshape(50000,3,32,32)
X_train = np.transpose(X_train, (0, 2, 3,1))/255
Y_train = to_categorical(Y_train)

model = Sequential()

#model layers
model.add(Conv2D(64, kernel_size=3,strides=1, padding='same', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=3,strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(Dense(10, activation='softmax'))

#compile
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#fit model
model.fit(X_train,Y_train,validation_split=0.2,batch_size=64,epochs=30)

test = pd.read_csv(param[1],sep=' ',header=None).values
X_test = test[:,0:-1]
Y_test = test[:,-1]
X_test = X_test.reshape(10000,3,32,32)
X_test = np.transpose(X_test, (0, 2, 3,1))/255
pred = np.argmax(model.predict(X_test),axis=1).astype(int)
np.savetxt(param[2],pred,delimiter="\n")
