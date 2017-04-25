from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib
import math
import test.kie_image as im
import sys

from PIL import Image
import imagehash

# [[(int(math.ceil(_w[0][1])) if len(_w) > 0 else 0) for _w in _h] for _h in _kp]
# [[(len(_w) for _w in _h] for _h in _kp]
# [[len(_w['points']) for _w in _h] for _h in _kp]

MIN_MATCH_COUNT = 100
PATH = './images/'
# PATH = 'http://192.168.0.164:3000/media/'
fn3 = 'http://comicstore.cf/uploads/diamonds/STK309612.jpg'
fn1 = 'j11.jpg'
fn2 = 'j1.jpg'

e1 = cv2.getTickCount()
t_start = cv2.getTickCount()

img1 = im.loadImageFromPath(PATH + fn1, resize=True, maxSize=200)
h1, w1 = img1.shape[:2]
t_end = cv2.getTickCount()
# print "Time loading 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# cv2.imshow("Image", img1)
# cv2.waitKey(0)


# img1 = url_to_image('http://comicstore.cf/uploads/diamonds/STK368216.jpg')  # queryImage


# t_start = cv2.getTickCount()
img2 = im.loadImageFromPath(PATH + fn2, resize=True, maxSize=200)
# img2 = im.loadImageFromUrl(fn3, resize=True, maxSize=800)
# h2, w2 = img2.shape[:2]
# t_end = cv2.getTickCount()
# print "Time loading 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# cv2.imshow("Image", img2)
# cv2.waitKey(0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
# t_start = cv2.getTickCount()
# kp1, des1 = sift.detectAndCompute(img1, None)
_kp1, _des1 = sift.detectAndCompute(img1, None)
kp1, des1 = im.sortKp(_kp1, _des1, 100)
# kp1, des1 = sortKp(_kp1, _des1, h1, w1)

# print "Time keypoint len: %s" % len(kp1)
# im.saveKpDesToPath(kp1, des1, PATH + im.fileName(fn1) + '.kp.png')
# im.saveKeypointToPath(kp1, PATH + im.fileName(fn1) + '.kp')

"""
print sys.getsizeof(kp1)
print sys.getsizeof(des1)
exit(0)
"""

"""
im.saveDesToPath(des1, PATH + '1.des')
im.saveKeypointToPath(kp1, PATH + '1.kp')
des11 = im.loadDesFromPath(PATH + '1.des')
kp11 = im.loadKeypointFromPath(PATH + '1.kp')

print np.array_equal(des1, des11)
print np.array_equal(kp1, kp11)

flag = True
for i in range(0, 100):
    if kp1[i].pt[0] != kp11[i].pt[0] or \
                    kp1[i].pt[1] != kp11[i].pt[1] or \
                    kp1[i].size != kp11[i].size or \
                    kp1[i].angle != kp11[i].angle or \
                    kp1[i].class_id != kp11[i].class_id or \
                    kp1[i].octave != kp11[i].octave or \
                    kp1[i].response != kp11[i].response:
        flag = False
        break

print flag
"""

# exit(0)
# cv2.imwrite(PATH + im.fileName(fn1) + '.des.png', des1)
# des11 = cv2.imread(PATH + im.fileName(fn1) + '.des.png', cv2.IMREAD_GRAYSCALE)
# print np.array_equal(des1, des11)

# exit(0)

# t_end = cv2.getTickCount()
# print "Time keypoint 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())
# t_start = cv2.getTickCount()
# kp2, des2 = sift.detectAndCompute(img2, None)
_kp2, _des2 = sift.detectAndCompute(img2, None)
kp2, des2 = im.sortKp(_kp2, _des2, 100)
# kp2, des2 = sortKp(_kp2, _des2, h2, w2)

# im.saveKpDesToPath(kp2, des2, PATH + im.fileName(fn2) + '.kp.png')
# im.saveKeypointToPath(kp2, PATH + im.fileName(fn2) + '.kp')
# im.saveDesToPath(des2, PATH + im.fileName(fn2) + '.des.jpg')
# t_end = cv2.getTickCount()
# print "Time keypoint 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = {}
# search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

t_start = cv2.getTickCount()
matches = flann.knnMatch(des1, des2, k=2, compactResult=True)
t_end = cv2.getTickCount()
print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# store all the good matches as per Lowe's ratio test.
t_start = cv2.getTickCount()
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > 10:
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
# print "Time calc comparing: %s" % ((t_end - t_start) / cv2.getTickFrequency())

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,  # draw only inliers
                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

# img4 = None
# img5 = None
# img4 = cv2.drawKeypoints(img1, kp1, img4, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img5 = cv2.drawKeypoints(img2, kp2, img5, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imwrite(PATH + 'j1__.jpg', img4)
# cv2.imwrite(PATH + 'j4__.jpg', img5)

e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
# print "Time: %s" % (time)

# plt.imshow(img4, 'gray')
# plt.show()
# plt.imshow(img5, 'gray')
# plt.show()
plt.imshow(img3, 'gray'), plt.show()
