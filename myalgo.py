import numpy as np
np.random.seed(1)
import os, glob
import cv2
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Load dataset
datapath = '.\dataset\\train_kana'
img_size = 28
num_classes = 48
rabel_dict = {}
x = []
y = []
x_train = []
y_train = []
x_test = []
y_test = []

starttime = time.time()
for number, name in enumerate(glob.glob(datapath + "/*")):
    print(name.split("\\")[-1] + ":開始")
    files = glob.glob(name + "/*.jpg")
    rabel = name.split("\\")
    rabel_dict[number] = name.split("\\")[-1]
    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        img = cv2.resize(img, dsize=(img_size, img_size))
        ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        img_thresh = cv2.bitwise_not(img_thresh)
        img_thresh = cv2.GaussianBlur(img_thresh, (9, 9), 0)
        x.append(img_thresh)
        y.append(number)
        if i > 100:
                break
print('Computation time {0:3f} minute'.format((time.time() - starttime) / 60))
x = np.asarray(x).reshape(np.asarray(x).shape + (1,))
x = np.array(x) / 255
y = to_categorical(np.asarray(y), 48)
print(x.shape, y.shape)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)

print(x_train.shape, y_train.shape)
print(len(np.unique(y_train)))
#make cloan
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(img_size, img_size , 1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#cloan Study
starttime = time.time()
history = model.fit(x_train, y_train, batch_size=1000, epochs=20, verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])
print('Computation time{0:3f} minute'.format((time.time() - starttime) / 60))
