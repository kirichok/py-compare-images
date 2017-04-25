import cv2

import glob
import os
import kie_image as im
import numpy as np

import threading
import Queue

HASH_PATH = '../images/hash/'
DES_EXT = '.des'


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flanns = []
filenames = []


class LoadHashThread(threading.Thread):
    def __init__(self, threadID, name, task, flann, files, event):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.task = task
        self.files = files
        self.flann = flann
        self.event = event

    def run(self):
        while not self.event.is_set():
            if not self.task.empty():
                task = self.task.get()
                des = im.loadDesFromPath(task)
                if len(des) >= 2:
                    self.files.append(im.fileName(task))
                    self.flann.add([des[:25]])


class checkHashThread(threading.Thread):
    def __init__(self, threadID, name, task, flann, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.task = task
        self.files = files
        self.flann = flann

    def run(self):
        while not self.event.is_set():
            if not self.task.empty():
                task = self.task.get()
                matches = self.flann.knnMatch(task[1], k=2)

                m = [m.imgIdx for m, n in matches if m.distance < n.distance * 0.75]
                results = {'v': [], 'c': []}
                for i in m:
                    try:
                        index = results['v'].index(i)
                        results['c'][index] += 1
                    except ValueError:
                        results['v'].append(i)
                        results['c'].append(1)

                print '%d:\n %s' % (len(m), ' '.join(
                    ['- %s(%s): \n' % (results['v'][i], results['c'][i], self.files[results['v'][i]]) for i in
                     range(0, len(results['v']))]))


def loadFiles(filesInQueue=10000, hashPath=HASH_PATH, ext=DES_EXT):
    qFiles = []
    folder = 0
    path = "%s%s/" % (hashPath, folder)
    count = 0
    countAll = 0
    curQueue = Queue.Queue(filesInQueue)
    qFiles.append(curQueue)
    while os.path.exists(path):
        for imagePath in glob.glob("%s*%s" % (path, ext)):
            curQueue.put(imagePath)
            count += 1
            countAll += 1
            if count == filesInQueue:
                qFiles.append(curQueue)
                curQueue = Queue.Queue(filesInQueue)
                count = 0

            # if countAll == 2000:
            #     folder = -2
            #     break

        folder += 1
        path = "%s%s/" % (hashPath, folder)

    if countAll == 0:
        print("Hash files count is empty")
    else:
        print("Files count: %d " % countAll)

        threadList = []
        count = 0
        while count < len(qFiles):
            threadList.append("FilesThread-%d" % count)
            filenames.append([])
            count += 1

        event = threading.Event()
        threads = []
        threadID = 0

        # Create new threads
        for i, tName in enumerate(threadList):
            currflann = cv2.FlannBasedMatcher(index_params, search_params)
            flanns.append(currflann)
            thread = LoadHashThread(threadID, tName, qFiles[i], currflann, filenames[i], event)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            threadID += 1

        # Wait for queue to empty
        empty = False
        while not empty:
            empty = True
            for q in qFiles:
                if not (empty and q.empty()):
                    empty = False

        # Notify threads it's time to exit
        event.set()

        # Wait for all threads to complete
        for t in threads:
            t.join()


def testFlan():
    sift = cv2.xfeatures2d.SIFT_create()

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
    duples = [c for n, (a, b, c) in enumerate(m) if c in [cc for aa, bb, cc in m[:n if n > 0 else n + 1]]]
    no_duples = [c for n, (a, b, c) in enumerate(m) if c not in duples and c not in [cc for aa, bb, cc in m[:n]]]
    print "Points: %d,\n no duples(%d): %s,\n duples(%d): %s" % (len(m),
                                                                 len(no_duples), ' '.join([str(i) for i in no_duples]),
                                                                 len(duples), ' '.join([str(i) for i in duples]))


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


if __name__ == '__main__':
    loadFiles()

    inpt = -1
    while inpt != 0:
        try:
            inpt = int(raw_input(' 0 - exit \n 1 - check \n'))
        except ValueError:
            inpt = -1
            continue
        pass