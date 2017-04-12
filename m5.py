from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib
import math
import test.kie_image as im


# import ntpath
# head, tail = ntpath.split('http://comicstore.cf/uploads/diamonds/STK368216.jpg')

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format

    # e1 = cv2.getTickCount()

    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()

    height, width = image.shape[:2]
    delta = 800 / height

    if height <= 800:
        delta = 1

    # print 't: %s d: %s h: %s w: %s' % (time, delta, height, width)
    image = cv2.resize(image, (math.trunc(width * delta), math.trunc(height * delta)), interpolation=cv2.INTER_CUBIC)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # return the image
    return image


e1 = cv2.getTickCount()

MIN_MATCH_COUNT = 100
PATH = './images/'

fn1 = 'z1.jpg'
fn2 = 'z2.jpg'

t_start = cv2.getTickCount()
img1 = cv2.imread(PATH + fn1, 0)  # queryImage
t_end = cv2.getTickCount()
print "Time loading 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# cv2.imshow("Image", img1)
# cv2.waitKey(0)


# img1 = url_to_image('http://comicstore.cf/uploads/diamonds/STK368216.jpg')  # queryImage
t_start = cv2.getTickCount()
img2 = cv2.imread(PATH + fn2, 0)  # trainImage
t_end = cv2.getTickCount()
print "Time loading 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())


# cv2.imshow("Image", img2)
# cv2.waitKey(0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
t_start = cv2.getTickCount()
kp1, des1 = sift.detectAndCompute(img1, None)

print "Time keypoint len: %s" % len(kp1)
im.saveKpDesToPath(kp1, des1, PATH + im.fileName(fn1) + '.kp.png')
# im.saveKeypointToPath(kp1, PATH + im.fileName(fn1) + '.kp')
# im.saveDesToPath(des1, PATH + im.fileName(fn1) + '.des.png')
t_end = cv2.getTickCount()
print "Time keypoint 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())
t_start = cv2.getTickCount()
kp2, des2 = sift.detectAndCompute(img2, None)
im.saveKpDesToPath(kp2, des2, PATH + im.fileName(fn2) + '.kp.png')
# im.saveKeypointToPath(kp2, PATH + im.fileName(fn2) + '.kp')
# im.saveDesToPath(des2, PATH + im.fileName(fn2) + '.des.png')
t_end = cv2.getTickCount()
print "Time keypoint 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

t_start = cv2.getTickCount()
matches = flann.knnMatch(des1, des2, k=2)
t_end = cv2.getTickCount()
print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# store all the good matches as per Lowe's ratio test.
t_start = cv2.getTickCount()
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    print 'GOOD: %d' % (len(good))

else:
    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
    matchesMask = None

t_end = cv2.getTickCount()
print "Time calc comparing: %s" % ((t_end - t_start) / cv2.getTickFrequency())

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print "Time: %s" % (time)

plt.imshow(img3, 'gray'), plt.show()
