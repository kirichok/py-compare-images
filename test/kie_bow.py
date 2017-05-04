from memory_profiler import profile
import cv2

import glob
import os
import kie_image as im
import kie_mysql as sql
import numpy as np

import threading
import Queue
import time

import gc
from sys import getsizeof

IMG_PATH = '../images/'
HASH_PATH = '../images/hash/'
DES_EXT = '.des'

FILES_CHECK = 1000

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=10)
search_params = {}
sift = cv2.xfeatures2d.SIFT_create()

flanns = []
filenames = []


@profile
def loadFiles(filesInQueue=10000, hashPath=HASH_PATH, ext=DES_EXT):
    def load_descriptors(bow, hashPath=HASH_PATH, withSubFolders=True, ext=DES_EXT):
        des_list = []
        kp_list = []

        folder = 0
        path = "%s%s/" % (hashPath, folder)
        while os.path.exists(path):
            for imagePath in glob.glob("%s*%s" % (path, '.des')):
                des_list.append(imagePath)
                kp_list.append('%s%s.kp' % (path, im.fileName(imagePath)))
                if len(des_list) == FILES_CHECK:
                    folder = -2
                    break

            folder += 1
            path = "%s%s/" % (hashPath, folder)

        if len(des_list) == 0:
            print("Hash files count is empty")
            return []

        print("Files count: %d " % len(des_list))
        for n in des_list:
            bow.add(im.loadDesFromPath(n))
        # bow.write(HASH_PATH + 'hash.bat')
        print("Loaded")

        images = []
        for n in des_list:
            images.append('http://comicstore.cf/uploads/diamonds/%s.jpg' % im.fileName(n))

        return images, kp_list

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    bow_train = cv2.BOWKMeansTrainer(50)  # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, flann)

    images, kps = load_descriptors(bow_train)

    print 'Getting cluster ...'
    voc = bow_train.cluster()
    print 'Set vocabulary ...'
    bow_extract.setVocabulary(voc)

    traindata, trainlabels = [], []

    j = 0
    for i in images:
        img = im.loadImageFromUrl(i, resize=True, maxSize=800)
        kp = im.loadKeypointFromPath(kps[j])
        traindata.extend(bow_extract.compute(img, kp))
        trainlabels.append(j)
        j += 1

    print 'Training SVM'
    svm = cv2.ml.SVM_create()
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    img = im.loadImageFromPath('%s%s' % (IMG_PATH, 'M2.jpg'), resize=True, maxSize=800)
    kp, des = im.keypointDesCalc(img, count=100)
    fbow = bow_extract.compute(img, kp)

    t_start = cv2.getTickCount()
    p = svm.predict(fbow)
    t_end = cv2.getTickCount()
    print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())


def downloadImages():
    nameList = sql.allComics()[:1000]
    for i in nameList:
        img = im.loadImageFromUrl(i, color=cv2.IMREAD_COLOR, resize=False, maxSize=800)
        cv2.imwrite('/home/evgeniy/projects/python/open-cv/images/comics/full/%s.jpg' % im.fileName(i), img)


if __name__ == '__main__':
    # loadFiles(120000)
    downloadImages()
