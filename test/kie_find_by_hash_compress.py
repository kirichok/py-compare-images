import Queue
import threading
import cv2
import glob
# import kie_const
import numpy as np
import kie_image as image
import argparse

HASH_PATH = '../images/hash3/'
KPC_EXT = '.kpc'

exitFlag = 0


class HashThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        queueLock.acquire()
        des1 = des.copy()
        queueLock.release()
        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                path = self.q.get()
                queueLock.release()
                o1 = cv2.getTickCount()
                kps = image.loadKps(path)
                o2 = cv2.getTickCount()
                for data in kps:
                    name, kp2, des2 = image.unpack(data)
                    if len(kp2) > 2:
                        m = image.match(des1, des2)  # np.asarray(des2, np.float32))
                        if len(m) >= 50:
                            print("Matched %s file %s" % (len(m), name))
                    else:
                        print("KP in file %s is less" % name)
                p1 = cv2.getTickCount()
                print("Load: %s s, Process: %s s" % ((o2 - o1) / cv2.getTickFrequency(), (p1 - o2) / cv2.getTickFrequency()))
            else:
                queueLock.release()

        print("Exiting " + self.name)


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threads", required=True, type=int, help="Threads count")
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-hs", "--hash", required=True, help="Path to the hash files")
args = vars(ap.parse_args())

img = image.loadImageFromPath(args['image'], cv2.IMREAD_GRAYSCALE, True, 200)
kp, des = image.getKpDes(img)

threadList = []
count = 0
while count < args['threads']:
    threadList.append("Thread-%d" % count)
    count += 1

nameList = []
for imagePath in glob.glob("%s*%s" % (args['hash'], KPC_EXT)):
    nameList.append(imagePath)

if len(nameList) == 0:
    exit(0)
else:
    print("Files count: %d " % len(nameList))

queueLock = threading.Lock()
workQueue = Queue.Queue(1000000)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = HashThread(threadID, tName, workQueue)
    # thread.daemon = True
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
print("Time: %s s" % (time))

print("Exiting Main Thread")