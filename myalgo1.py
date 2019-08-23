import numpy as np
np.random.seed(1)
import os, glob
import cv2
import time
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

if __name__ == "__main__":

    # Load dataset
    trainpath = './dataset/train_kana'
    testpath = './dataset/train'
    img_size = 224
    num_classes = 48
    rabel_dict = {}
    revers_rabel_dict = {}
    x = []
    y = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    starttime = time.time()
    #traindata
    #---一文字から学習データを作成---
    for number, name in enumerate(glob.glob(trainpath + "/*")):
        print(name.split("\\")[-1] + ":開始")
        files = glob.glob(name + "/*.jpg")
        rabel = name.split("\\")
        rabel_dict[number] = name.split("\\")[-1]
        revers_rabel_dict[name.split("\\")[-1]] = number
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
    print('Computation time{0:3f} minute'.format((time.time() - starttime) / 60))

    # starttime = time.time()
    #testdata
    #---画像を３等分---
    # print('testdata作成開始')
    # for i, name in enumerate(glob.glob(testpath + "\imgs" + "/*.jpg")):
    #     img = cv2.imread(name, 0)
    #     threshold = cv2.threshold(name, 0, 255, cv2.THRESH_OTSU)

        # box = []
        # height, width = img.shape[: 2]
        # height = int(height / 3)
        # img1 = img[0 : int(height), 0 : int(width)]
        # img1 = cv2.resize(img1, dsize=(img_size, img_size))
        # box.append(img1)
        # img2 = img[int(height) : int(height * 2), 0 : int(width)]
        # img2 = cv2.resize(img2, dsize=(img_size, img_size))
        # box.append(img2)
        # img3 = img[int(height * 2) : int(height * 3), 0 : int(width)]
        # img3 = cv2.resize(img3, dsize=(img_size, img_size))
        # box.append(img3)
        # for number, name in enumerate(box):
        #     ret, box[number] = cv2.threshold(name, 0, 255, cv2.THRESH_OTSU)
        # for number, name in enumerate(box):
        #     box[number] = cv2.bitwise_not(name)
        # for number, name in enumerate(box):
        #     box[number] = cv2.GaussianBlur(name, (9, 9), 0)
        # for number, name in enumerate(box):
        #     x_test.append(box[number])
    #     if i > 100:
    #         break
    # print('testdata作成終了')
    # print('Computation time{0:3f} minute'.format((time.time() - starttime) / 60))

    #testrabel
    # starttime = time.time()
    # print('testrabel作成開始')
    # with open(testpath + '\\annotations.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     for number, name in enumerate(reader):
    #         y_test.append(revers_rabel_dict[name[1]])
    #     y_test.append(revers_rabel_dict[name[2]])
    #     y_test.append(revers_rabel_dict[name[3]])
    #         if number > 100:
    #             break
    # print('testrabel作成終了')
    # print('Computation time {0:3f} minute'.format((time.time() - starttime) / 60))

    #データ整形
    x = np.asarray(x).reshape(np.asarray(x).shape + (1,))
    # x_test = np.asarray(x_test).reshape(np.asarray(x_test).shape + (1,))
    x = np.array(x) / 255
    # x_test = np.asarray(x_test) / 255
    y = to_categorical(np.asarray(y), 48)
    # y_test = to_categorical(np.asarray(y_test), 48)
    print(x.shape, y.shape)

    # traindataからtestdata作成
    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3)
    print(x_train.shape, y_train.shape)
    print(len(np.unique(y_train)))

    #make cloan
    #Alex Net
    model = Sequential()

    model.add(Conv2D(48, (11, 11), input_shape=(img_size, img_size, 1), strides=(4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 5, strides=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=1, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #cloan Study
    starttime = time.time()
    history = model.fit(x, y, batch_size=1000, epochs=50, verbose=1,validation_data=(x_test, y_test))
    #history = model.fit(x_train, y_train, batch_size=1000, epochs=100, verbose=1,validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    print('Computation time{0:3f} minute'.format((time.time() - starttime) / 60))
