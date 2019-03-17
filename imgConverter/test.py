import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from imgConverter import imgConverter

batch_size = 2
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 1024, 1024

# the data, split between train and test sets
x, ids ,pre_y = imgConverter("C:/Users/ajare/Dropbox/school/dataScienceProject/imgConverter")

yDict = {}
y = []

counter = 0
for i in pre_y:
    if i not in yDict.keys():
        yDict[i] = counter
        counter += 1
    y += [yDict[i]]

yDict = {value: key for key, value in yDict.items()}
y = np.array(y)
x = np.array(x)
x = x.reshape(2,1024,1024,3)

# convert class vectors to binary class matrices
y = keras.utils.to_categorical(y, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(1024,1024,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x, y))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
