import Queue
import threading
import cv2
import os, os.path
import kie_mysql as sql
import kie_image as image


HASH_PATH = '../images/hash/'

THREAD_COUNT = 1
exitFlag = 0


class HashThread(threading.Thread):
    def __init__(self, threadID, name, q):
        import kie_mongo as mongo
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
        self.collection = mongo.connect('comicstore', 'files')


    def run(self):
        print("Starting " + self.name)

        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                url = self.q.get()
                queueLock.release()

                img = image.loadImageFromUrl(url, cv2.IMREAD_GRAYSCALE, False)
                name = image.fileName(url)
                image.keypointDesCalcDb(self.collection, img, name, 100)
            else:
                queueLock.release()

        print("Exiting " + self.name)


threadList = []
count = 0
while count < THREAD_COUNT:
    threadList.append("Thread-%d" % count)
    count += 1

nameList = sql.allComics()

if len(nameList) == 0:
    exit(0)

queueLock = threading.Lock()
workQueue = Queue.Queue(1000000)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = HashThread(threadID, tName, workQueue)
    thread.daemon = True
    thread.start()
    threads.append(thread)
    threadID += 1

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
print("Time: %s" % (time))

print("Exiting Main Thread")
