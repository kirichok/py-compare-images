from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib
import math
import test.kie_image as im

from operator import itemgetter, attrgetter


# import ntpath
# head, tail = ntpath.split('http://comicstore.cf/uploads/diamonds/STK368216.jpg')

# [[(int(math.ceil(_w[0][1])) if len(_w) > 0 else 0) for _w in _h] for _h in _kp]
# [[(len(_w) for _w in _h] for _h in _kp]
# [[len(_w['points']) for _w in _h] for _h in _kp]

def sortKp_(kp, des, h, w):
    # _kp = []
    i = 0
    _kp = [[{'min': None, 'pos': 0, 'points': []} for x in range(10)] for y in range(10)]

    h01 = h * 0.1
    h09 = h * 0.9
    w01 = w * 0.1
    w09 = w * 0.9

    h0 = h * 0.8
    w0 = w * 0.8

    for point in kp:
        if h01 < point.pt[1] < h09:
            if w01 < point.pt[0] < w09:
                pass
            else:
                continue
        else:
            continue

        a = _kp[int(math.floor((point.pt[1] - h01) / h0 * 10))][int(math.floor((point.pt[0] - w01) / w0 * 10))]

        if len(a['points']) == 10:
            if a['min'] > point.size:
                continue
            else:
                a['min'] = point.size
                a['points'][a['pos']] = \
                    (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, des[i])
                _i = 0
                for e in a['points']:
                    if a['min'] > e[1]:
                        a['min'] = e[1]
                        a['pos'] = _i
                    _i += 1
        else:
            if a['min'] is None:
                a['min'] = point.size
                a['pos'] = len(a['points'])
            elif a['min'] > point.size:
                a['min'] = point.size
                a['pos'] = len(a['points'])

            a['points'].append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id, des[i]))

        i += 1

    r_kp = []
    r_des = []
    __kp = []
    for _h in _kp:
        for _w in _h:
            if len(_w['points']) > 0:
                _w['points'] = sorted(_w['points'], key=itemgetter(1), reverse=True)
                __kp.append(_w['points'][0])

    __kp = sorted(__kp, key=itemgetter(1), reverse=True)

    for point in __kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        r_des.append(point[6])
        r_kp.append(temp)
        if len(r_kp) == 100:
            break

    return r_kp, np.asarray(r_des, np.float32)


def sortKp(kp, des, count):
    cc = 0
    _kp = []
    i = 0
    for point in kp:
        _kp.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id, des[i]))
        i += 1

    _kp = sorted(_kp, key=itemgetter(1), reverse=True)

    r_kp = []
    r_des = []
    for point in _kp:
        flag = False
        for p in r_kp:
            if abs(point[0][0] - p.pt[0]) < 1 and abs(point[0][1] - p.pt[1]) < 1:
                flag = True
                break

        if flag:
            cc += 1
            continue

        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        r_des.append(point[6])
        r_kp.append(temp)
        if len(r_kp) == count:
            break

    print cc
    return r_kp, np.asarray(r_des, np.float32)


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

fn1 = 'j11.jpg'
fn2 = 'j1.jpg'

t_start = cv2.getTickCount()
img1 = cv2.imread(PATH + fn1, 0)  # queryImage
h1, w1 = img1.shape[:2]
t_end = cv2.getTickCount()
print "Time loading 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# cv2.imshow("Image", img1)
# cv2.waitKey(0)


# img1 = url_to_image('http://comicstore.cf/uploads/diamonds/STK368216.jpg')  # queryImage
t_start = cv2.getTickCount()
img2 = cv2.imread(PATH + fn2, 0)  # trainImage
h2, w2 = img2.shape[:2]
t_end = cv2.getTickCount()
print "Time loading 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())

# cv2.imshow("Image", img2)
# cv2.waitKey(0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
# t_start = cv2.getTickCount()
_kp1, _des1 = sift.detectAndCompute(img1, None)
kp1, des1 = sortKp(_kp1, _des1, 100)
# kp1, des1 = sortKp(_kp1, _des1, h1, w1)

# print "Time keypoint len: %s" % len(kp1)
# im.saveKpDesToPath(kp1, des1, PATH + im.fileName(fn1) + '.kp.png')
# im.saveKeypointToPath(kp1, PATH + im.fileName(fn1) + '.kp')
# im.saveDesToPath(des1, PATH + im.fileName(fn1) + '.des.jpg')
# t_end = cv2.getTickCount()
# print "Time keypoint 1: %s" % ((t_end - t_start) / cv2.getTickFrequency())
# t_start = cv2.getTickCount()
_kp2, _des2 = sift.detectAndCompute(img2, None)
kp2, des2 = sortKp(_kp2, _des2, 100)
# kp2, des2 = sortKp(_kp2, _des2, h2, w2)

# im.saveKpDesToPath(kp2, des2, PATH + im.fileName(fn2) + '.kp.png')
# im.saveKeypointToPath(kp2, PATH + im.fileName(fn2) + '.kp')
# im.saveDesToPath(des2, PATH + im.fileName(fn2) + '.des.jpg')
# t_end = cv2.getTickCount()
# print "Time keypoint 2: %s" % ((t_end - t_start) / cv2.getTickFrequency())


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
    if m.distance < 0.6 * n.distance:
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

img4 = None
img5 = None
img4 = cv2.drawKeypoints(img1, kp1, img4, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv2.drawKeypoints(img2, kp2, img5, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite(PATH + 'j1__.jpg', img4)
cv2.imwrite(PATH + 'j4__.jpg', img5)

e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print "Time: %s" % (time)

# plt.imshow(img4, 'gray')
# plt.show()
# plt.imshow(img5, 'gray')
# plt.show()
plt.imshow(img3, 'gray'), plt.show()
