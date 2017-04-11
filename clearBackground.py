import cv2
import numpy as np

image = cv2.imread('./images/clear background.png')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def clear_vertical(img, target):
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j][i]:
                break
            else:
                target[j][i] = [0, 0, 0]


def clear_horizontal(img, target):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]:
                break
            else:
                target[i][j] = [0, 0, 0]


def turn_off(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 0


def turn_on(img, result):
    for i in result:
        img[i[0][1]][i[0][0]] = 255


def f(list):
    max = []
    for i in list:
        if len(i) > len(max):
            max = i
    return max


def rem(ls, thresh):
    new_c = []
    for i in ls:
        if len(i) > thresh:
            new_c.append(i)
    return new_c


def rn(ls, min, max):
    ret = []
    for i in ls:
        if len(i) < max and len(i) > min:
            print(len(i))
            ret.append(i)
    return ret


# ret,tresh = cv2.threshold(img,40,255,cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
new = cv2.Canny(img, 190, 1)
dilated = cv2.dilate(new, kernel)
tresh, c, hr = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
c = rn(c, 2600, 4000)
turn_off(new)
turn_on(new, c[0])

clear_horizontal(new, image)
clear_vertical(new, image)

# cv2.imwrite('result_image_end.png',image)

cv2.imshow('wnd', image)
cv2.waitKey(0)
