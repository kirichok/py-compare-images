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

THREAD_COUNT = 200
exitFlag = 0

class HashThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print "Starting " + self.name
        queueLock.acquire()
        des1 = des.copy()
        queueLock.release()
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                path = self.q.get()
                queueLock.release()

                des2 = image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False)
                name = image.fileName(path)

                m = image.match(des1, np.asarray(des2, np.float32))
                if len(m) >= 50:
                    print "Matched %s file %s" % (len(m), name)
            else:
                queueLock.release()

        print "Exiting " + self.name

img = image.loadImageFromPath('../images/M6.jpg', cv2.IMREAD_GRAYSCALE, True, 200)
kp, des = image.getKpDes(img)

threadList = []
count = 0
while count < THREAD_COUNT:
    threadList.append("Thread-%d" % count)
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
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = HashThread(threadID, tName, workQueue)
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
while not workQueue.empty():
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
