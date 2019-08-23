import numpy as np
np.random.seed(1)
import os, glob
import cv2
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

datapath = '.\dataset\\train\imgs'

# 画像を読み込む。
img = cv2.imread(".\dataset\\train\imgs\\1.jpg", 0)

for number, name in enumerate(glob.glob(datapath + "/*.jpg")):
    img = cv2.imread(name, 0)
    img = cv2.resize(img, dsize=(600,600))
    ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_thresh = cv2.bitwise_not(img_thresh)
    img_thresh = cv2.GaussianBlur(img_thresh, (9, 9), 0)
    x = img_thresh
    y = cv2.flip(x, -1)
    blended = cv2.addWeighted(src1=x, alpha=0.5, src2=y, beta=0.5,gamma=1)
    cv2.imshow("test", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if number > 10:
        break
