import cv2
import numpy as np
from os.path import join
import test.kie_image as im

import glob
import os

def test1():

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = {}
    search_params = dict(check=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    detect = cv2.xfeatures2d.SIFT_create()
    extract = cv2.xfeatures2d.SIFT_create()

    bow_train = cv2.BOWKMeansTrainer(4)  # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor(extract, flann)


    ## 1.b add positives and negatives to the bowtrainer, use SIFT DescriptorExtractor
    def feature_sift(fn):
        img = im.loadImageFromPath(fn, resize=True, maxSize=800)
        kp, des = im.keypointDesCalc(img, count=100)
        return des


    basepath = "./images/"

    images = ["j1.jpg",
              "j5.jpg",
              "j7.jpg",
              "z1.jpg",
              "M4.jpg",
              "z2.jpg",
              "M1.jpg"]

    for i in images:
        bow_train.add(feature_sift(join(basepath, i)))

    ## 1.c kmeans cluster descriptors to vocabulary
    voc = bow_train.cluster()
    bow_extract.setVocabulary(voc)


    ## 2.a gather svm training data, use BOWImgDescriptorExtractor
    def feature_bow(fn):
        img = im.loadImageFromPath(fn, resize=True, maxSize=800)
        kp, des = img.keypointDesCalc(im, count=100)
        return bow_extract.compute(img, kp)
        # return bow_extract.compute(im, detect.detect(im))


    traindata, trainlabels = [], []

    j = 1
    for i in images:
        traindata.extend(feature_bow(join(basepath, i)))
        trainlabels.append(j)
        j += 1

    print "svm items", len(traindata), len(traindata[0])

    ## 2.b create & train the svm
    svm = cv2.ml.SVM_create()
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

    ## 2.c predict the remaining 2*2 images, use BOWImgDescriptorExtractor again
    def predict(fn):
        f = feature_bow(fn)
        p = svm.predict(f)
        print fn, "\t", p[1][0][0]

    sample = feature_bow(join(basepath, "M4.jpg"))

    t_start = cv2.getTickCount()
    p = svm.predict(sample)[1].ravel()
    t_end = cv2.getTickCount()
    print p, ((t_end - t_start) / cv2.getTickFrequency())


def test2():
    HASH_PATH = './images/hash/'
    DES_EXT = '.des'

    def load_descriptors(bow, hashPath=HASH_PATH, withSubFolders=True, ext=DES_EXT):
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
        for n in nameList[:10]:
            bow.add(im.loadDesFromPath(n))
            # knn.train()
        # bow.write(HASH_PATH + 'hash.bat')
        print("Loaded")

    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = {}
    # search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    bow_train = cv2.BOWKMeansTrainer(1000)  # toy world, you want more.
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, flann)

    load_descriptors(bow_train)

    voc = bow_train.cluster()
    bow_extract.setVocabulary(voc)

    img = im.loadImageFromUrl('http://comicstore.cf/uploads/diamonds/STK309612.jpg', resize=True, maxSize=800)
    _kp, _des = sift.detectAndCompute(img, None)
    kp, des = im.sortKp(_kp, _des, 100)

    t_start = cv2.getTickCount()
    bdes = bow_extract.compute(img, kp)
    # matches = flann.knnMatch(des)
    t_end = cv2.getTickCount()
    print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())

test1()