import Queue
import threading
import cv2
import glob
import numpy as np
import kie_image as image
import argparse
import os

HASH_PATH = '../images/hash/'
DES_EXT = '.des.jpg'

exitFlag = 0

class HashThread(threading.Thread):
    def __init__(self, threadID, name, lock, workQueue, kp, des, finds):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.workQueue = workQueue
        self.lock = lock
        self.kp = kp
        self.des = des
        self.finds = finds

    def run(self):
        global exitFlag
        # print("Starting " + self.name)
        while not exitFlag:
            self.lock.acquire()
            if not self.workQueue.empty():
                path = self.workQueue.get()
                self.lock.release()

                des2 = image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False)
                des2 = np.asarray(des2, np.float32)
                if isinstance(des2, list) and len(des2) >= 2:
                    name = image.fileName(path)
                    m = image.match(self.des, des2)
                    if len(m) >= 50:
                        self.lock.acquire()
                        self.finds.append({'m': len(m), 'n': name})
                        self.lock.release()
                        print("Matched %s file %s" % (len(m), name))
            else:
                self.lock.release()

        # print("Exiting " + self.name)


def check(imgPath, hashPath=HASH_PATH, withSubFolders=True, threadsCount=200):
    global exitFlag

    finds = []
    exitFlag = 0
    img = image.loadImageFromPath(imgPath, cv2.IMREAD_GRAYSCALE, True, 200)
    kp, des = image.getKpDes(img)

    threadList = []
    count = 0
    while count < threadsCount:
        threadList.append("Thread-%d" % count)
        count += 1

    nameList = []
    if withSubFolders:
        folder = 1
        path = "%s%s/" % (hashPath, folder)
        while os.path.exists(path):
            for imagePath in glob.glob("%s*%s" % (path, DES_EXT)):
                nameList.append(imagePath)
            folder += 1
            path = "%s%s/" % (hashPath, folder)
    else:
        for imagePath in glob.glob("%s*%s" % (hashPath, DES_EXT)):
            nameList.append(imagePath)

    if len(nameList) == 0:
        print("Hash files count is empty")
    else:
        print("Files count: %d " % len(nameList))

        queueLock = threading.Lock()
        workQueue = Queue.Queue(1000000)
        threads = []
        threadID = 1

        # Create new threads
        for tName in threadList:
            thread = HashThread(threadID, tName, queueLock, workQueue, kp, des, finds)
            thread.daemon=True
            thread.start()
            threads.append(thread)
            threadID += 1

        # Fill the queue
        queueLock.acquire()
        for word in nameList:
            workQueue.put(word)
        queueLock.release()

        # Wait for queue to empty
        while not workQueue.empty():
            pass

        # Notify threads it's time to exit
        exitFlag = 1

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return finds


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y" "1"):
            return True
        if v.lower() in ("no", "false", "f", "n" "0"):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threads", required=True, type=int, help="Threads count", default=50)
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-sf", "--subfolder", required=True, type=str2bool, help="Check sub folder", nargs='?', const=True)
    ap.add_argument("-hf", "--hash", required=True, help="Path to the hash folder", default=HASH_PATH)
    ap.add_argument("-ext", "--extention", help="Hash files extentions", default=DES_EXT)
    args = vars(ap.parse_args())

    e1 = cv2.getTickCount()
    f = check(args['image'], args['hash'], args['subfolder'], args['threads'])
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("Time: %s s" % (time))
    print("Finds: %s" % f)
    print("Exiting Main Thread")
