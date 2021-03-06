import Queue
import threading
import cv2
import os, os.path
import kie_mysql as sql
import kie_image as image

HASH_PATH = '../images/hashORB/'

THREAD_COUNT = 30
exitFlag = 0


class HashThread(threading.Thread):
    def __init__(self, threadID, name, queue, lock, wlock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.queue = queue
        self.lock = lock
        self.wlock = wlock

    def run(self):
        print("Starting " + self.name)

        while not exitFlag:
            self.lock.acquire()
            if not self.queue.empty():
                data = self.queue.get()
                self.lock.release()

                url = data[0]
                folder = '%s%s/' % (HASH_PATH, data[1])

                img = image.loadImageFromUrl(url, resize=True, maxSize=800)
                name = image.fileName(url)
                image.keypointDesCalc(img, folder + name, 100, self.wlock)
            else:
                self.lock.release()

        print("Exiting " + self.name)


def filesFromFolder(path, ext='.jpg'):
    import glob
    images = []
    if os.path.exists(path):
        for imagePath in glob.glob("%s*%s" % (path, ext)):
            images.append(imagePath)

    return images


threadList = []
count = 0
while count < THREAD_COUNT:
    threadList.append("Thread-%d" % count)
    count += 1

nameList = sql.allComics()
# nameList = filesFromFolder('../images/tatto/')

if len(nameList) == 0:
    exit(0)

writeLock = threading.Semaphore(15)
queueLock = threading.Lock()
workQueue = Queue.Queue(1000000)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = HashThread(threadID, tName, workQueue, queueLock, writeLock)
    thread.daemon = True
    thread.start()
    threads.append(thread)
    threadID += 1

queueLock.acquire()
_folder = 0
i = 0
for word in nameList:
    if i == 0:
        os.makedirs('%s%s/' % (HASH_PATH, _folder))
    workQueue.put((word, _folder))
    i += 1
    if i == 1000:
        i = 0
        _folder += 1

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
print("Time: %s" % (time))

print("Exiting Main Thread")
