from __future__ import division
import numpy as np
import cv2
# from matplotlib import pyplot as plt
import urllib
import math
import os.path
from os.path import basename, splitext
import cPickle
import zlib
from operator import itemgetter

KP_EXT = '.kp'
DES_EXT = '.des'


def loadImageFromUrl(url, color=cv2.IMREAD_GRAYSCALE, resize=True, maxSize=800):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, color)
    if resize:
        return __getDeltaTransformation(image, maxSize)
    else:
        return image


def loadImageFromPath(path, color=cv2.IMREAD_GRAYSCALE, resize=True, maxSize=800):
    image = cv2.imread(path, color)
    if resize:
        return __getDeltaTransformation(image, maxSize)
    else:
        return image


def __getDeltaTransformation(image, maxSize):
    height, width = image.shape[:2]
    maxValue = max(height, width)
    delta = 1
    if maxValue > maxSize:
        delta = maxSize / maxValue
    return cv2.resize(image, (math.trunc(width * delta), math.trunc(height * delta)), interpolation=cv2.INTER_CUBIC)


def keypointDesCalc(image, savePath='', count=0, writeLock=None):
    kp, des = sift.detectAndCompute(image, None)
    if count != 0:
        kp, des = sortKp(kp, des, count)
    if savePath:
        # saveKpDesToPath(kp, des, savePath + KP_EXT)
        saveKeypointToPath(kp, savePath + KP_EXT, writeLock)
        saveDesToPath(des, savePath + DES_EXT, writeLock)
    # return kp, des


def loadKeypointFromPath(path):
    index = cPickle.loads(open(path, 'rb').read())
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


def loadDesFromPath(path):
    des = cPickle.loads(open(path, 'rb').read())
    return np.asarray(des, np.float32)


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    if os.path.getsize(filename) <= 0:
        return np.array([])
    f = np.load(filename)
    if f.size == 0:
        return np.array([])
    f = np.atleast_2d(f)
    return f


def write_features_to_file(filename, data, lock):
    if lock is not None:
        lock.acquire()
    np.save(filename, data)
    if lock is not None:
        lock.release()
    del data


def pack_keypoint(keypoints):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                      kp.angle, kp.response, kp.octave,
                      kp.class_id]
                     for kp in keypoints])
    return kpts


def pack_descriptor(descriptors):
    desc = np.array(descriptors)
    return desc


def unpack_keypoint(kpts):
    try:
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                     for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints
    except(IndexError):
        return np.array([])


def saveKeypointToPath__(kp, path, lock):
    data = pack_keypoint(kp)
    write_features_to_file(path, data, lock)


def saveDesToPath__(des, path, lock):
    data = pack_descriptor(des)
    write_features_to_file(path, data, lock)


def saveKeypointToPath(kp, path, lock):
    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        index.append(temp)
    write_to_file(path, index, lock)


def saveDesToPath(des, path, lock):
    # des = des.toList()
    write_to_file(path, des, lock)


def write_to_file(filename, data, lock):
    if lock is not None:
        lock.acquire()
    f = open(filename, "wb")
    p = cPickle.Pickler(f, 2)
    p.dump(data)
    # f.write(zlib.compress(cPickle.dumps(data, 2)))
    f.close()
    if lock is not None:
        lock.release()
    p.clear_memo()


def loadKpDesFromPath(path, decompress=True):
    if decompress:
        index = cPickle.loads(zlib.decompress(open(path, 'rb').read()))
    else:
        index = cPickle.loads(open(path, 'rb').read())
    kp = []
    des = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
        des.append(point[6])
    return kp, np.array(des)


def saveKps(data, path):
    f = open(path, "wb")
    f.write(zlib.compress(cPickle.dumps(data)))


def saveKpDesToPath(kp, des, path, compress=True):
    index = []
    i = 0
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, des[i])
        index.append(temp)
        i += 1
    f = open(path, "wb")
    if compress:
        f.write(zlib.compress(cPickle.dumps(index)))
    else:
        f.write(cPickle.dumps(index))
    f.close()


def loadKps(path):
    return cPickle.loads(zlib.decompress(open(path, 'rb').read()))


def match(des1, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


def getKpDes(img):
    return sift.detectAndCompute(img, None)


def sortKp(kp, des, count):
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
            continue

        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        r_des.append(point[6])
        r_kp.append(temp)
        if len(r_kp) == count:
            break

    return r_kp, np.asarray(r_des, np.float32)


def keypointDesCalcDb(collection, image, name='', sort=0):
    kp, des = sift.detectAndCompute(image, None)
    if sort != 0:
        kp, des = sortKp(kp, des, sort)
    if name:
        index = []
        i = 0
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id)
            index.append(temp)
            i += 1

        collection.insert({'name': name, 'kp': index, 'des': des.tolist()})
    return kp, des


def pack(kp, des, name, compress=True):
    data = {'f': name, 'k': [], 'd': des}
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        data['k'].append(temp)

    data = cPickle.dumps(data)
    if compress:
        data = zlib.compress(data)
    return data


def unpack(data, decompress=True):
    if decompress:
        data = cPickle.loads(zlib.decompress(data))
    else:
        data = cPickle.loads(data)
    kp = []
    des = data['d']
    name = data['f']
    for point in data['k']:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return name, kp, np.array(des)


def compare(name1, name2, img1, img2):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
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

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        cv2.imwrite("./matched/%d %s_%s" % (len(good), name1, name2), img3)
        print "Matched (%s - %s) - %d/%d" % (name1, name2, len(good), MIN_MATCH_COUNT)
    else:
        print "Not enough matches are found (%s - %s) - %d/%d" % (name1, name2, len(good), MIN_MATCH_COUNT)


def fileName(str):
    return splitext(basename(str))[0]


MIN_MATCH_COUNT = 300

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# e1 = cv2.getTickCount()
# e2 = cv2.getTickCount()
# time = (e2 - e1) / cv2.getTickFrequency()
# print "Time: %s" % (time)
