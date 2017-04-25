import cv2

import glob
import os
import test.kie_image as im
import numpy as np

HASH_PATH = './images/hash/'
DES_EXT = '.des'


def load_descriptors(knn, hashPath=HASH_PATH, withSubFolders=True, ext=DES_EXT):
    nameList = []
    if withSubFolders:
        folder = 0
        path = "%s%s/" % (hashPath, folder)
        while os.path.exists(path):
            for imagePath in glob.glob("%s*%s" % (path, ext)):
                nameList.append(imagePath)
            folder += 1
            path = "%s%s/" % (hashPath, folder)
    else:
        for imagePath in glob.glob("%s*%s" % (hashPath, ext)):
            nameList.append(imagePath)

    if len(nameList) == 0:
        print("Hash files count is empty")
        return

    print("Files count: %d " % len(nameList))
    for i, n in enumerate(nameList[:50000]):
        if i % 10000 == 0:
            print '%d/50000' % i
        knn.add([im.loadDesFromPath(n)[:25]])
        # knn.train()
    knn.train()
    knn.write(HASH_PATH + 'hash.bat')
    print("Loaded")


sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
# search_params = {}
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

load_descriptors(flann)

# img = im.loadImageFromUrl('http://comicstore.cf/uploads/diamonds/STK360887.jpg', resize=True, maxSize=200)
img = im.loadImageFromPath('./images/z13.jpg', resize=True, maxSize=800)
_kp, _des = sift.detectAndCompute(img, None)
kp, des = im.sortKp(_kp, _des, 50)

t_start = cv2.getTickCount()
matches = flann.knnMatch(des, k=2)
t_end = cv2.getTickCount()
print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())
m = [(m.queryIdx, m.trainIdx, m.imgIdx) for m, n in matches if m.distance < n.distance * 0.75]
duples = [c for n, (a, b, c) in enumerate(m) if c in [cc for aa, bb, cc in m[:n if n>0 else n+1]]]
no_duples = [c for n, (a, b, c) in enumerate(m) if c not in duples and c not in [cc for aa, bb, cc in m[:n]]]
print "Points: %d,\n no duples(%d): %s,\n duples(%d): %s" % (len(m),
                                                             len(no_duples), ' '.join([str(i) for i in no_duples]),
                                                             len(duples), ' '.join([str(i) for i in duples]))