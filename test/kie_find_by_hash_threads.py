import Queue
import threading
import cv2
import glob
import numpy as np
import time
# import mysql_conn

import kie_mysql as sql
import kie_image as image

HASH_PATH = '../images/hash/'
KP_EXT = '.kp'
DES_EXT = '.des.jpg'

WORK_THREAD_COUNT = 200
FILES_THREAD_COUNT = 50
exitFlag = 0

threading.stack_size(64*1024)


class FilesThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        # print "Starting " + self.name
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty() and not filesQueue.full():
                path = self.q.get()
                print "%d" % filesQueue.qsize()
                queueLock.release()

                # t1 = cv2.getTickCount()
                des2 = np.asarray(image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False), np.float32)
                name = image.fileName(path)
                # t2 = cv2.getTickCount()

                if len(des2) >= 2:
                    queueLock.acquire()
                    filesQueue.put({'n': name, 'd': des2})
                    if filesQueue.qsize() > WORK_THREAD_COUNT * 10:
                        e.set()
                    queueLock.release()



                # t3 = cv2.getTickCount()
                # time = (e2 - e1) / cv2.getTickFrequency()
                # print "R: %s s P: %s s" % ((t2 - t1) / cv2.getTickFrequency(), (t3 - t2) / cv2.getTickFrequency())
            else:
                queueLock.release()
                time.sleep(5)


        print "Exiting " + self.name


class WorkThread(threading.Thread):
    def __init__(self, threadID, name, q, e):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
        self.e = e

    def run(self):
        # print "Starting " + self.name
        queueLock.acquire()
        des1 = des.copy()
        queueLock.release()
        while not self.e.wait():
            time.sleep(5)
        print "Starting " + self.name
        while not exitFlag:
            queueLock.acquire()
            if not filesQueue.empty():
                data = self.q.get()
                queueLock.release()
                des2 = data['d']
                name = data['n']

                m = image.match(des1, des2)
                if len(m) >= 50:
                    print "Matched %s file %s" % (len(m), name)
            else:
                queueLock.release()
                time.sleep(1)

        print "Exiting " + self.name

img = image.loadImageFromPath('../images/M6.jpg', cv2.IMREAD_GRAYSCALE, True, 200)
kp, des = image.getKpDes(img)

e = threading.Event()

workThreadList = []
count = 0
while count < WORK_THREAD_COUNT:
    workThreadList.append("WorkThread-%d" % count)
    count += 1

filesThreadList = []
count = 0
while count < FILES_THREAD_COUNT:
    filesThreadList.append("FileThread-%d" % count)
    count += 1

nameList = []
for imagePath in glob.glob("../images/hash/*.des.jpg"):
    nameList.append(imagePath)


if len(nameList) == 0:
    exit(0)
else:
    print "Files count: %d " % len(nameList)

queueLock = threading.Lock()
workQueue = Queue.Queue(1000000)
filesQueue = Queue.Queue(10000)
threads = []
threadID = 1

for tName in filesThreadList:
    thread = FilesThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# Create new threads
for tName in workThreadList:
    thread = WorkThread(threadID, tName, filesQueue, e)
    thread.start()
    threads.append(thread)
    threadID += 1

# Fill the queue
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

e1 = cv2.getTickCount()
# Wait for queue to empty
while not workQueue.empty() or not filesQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()

e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
print "Time: %s s" % (time)

print "Exiting Main Thread"
