import Queue
import threading
import cv2
import time
# import mysql_conn

import kie_mysql as sql
import kie_image as image

THREAD_COUNT = 50

exitFlag = 0


class myThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print "Starting " + self.name
        process_data(self.name, self.q)
        print "Exiting " + self.name


def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            url = q.get()
            queueLock.release()

            name2 = image.fileName(url)
            img2 = image.loadImageFromUrl(url)

            image.compare(name1, name2, img1.copy(), img2);
            # print "%s processing %s" % (threadName, data)
        else:
            queueLock.release()


url1 = '../images/M7.jpg'
img1 = image.loadImageFromPath(url1)

# url1 = 'http://comicstore.cf/uploads/diamonds/STK309612.jpg'
# img1 = image.loadImageFromUrl(url1)

name1 = image.fileName(url1)

threadList = []
count = 0
while count < THREAD_COUNT:
    threadList.append("Thread-%d" % count)
    count += 1

nameList = sql.executeQuery()

if len(nameList) == 0:
    exit(0)

queueLock = threading.Lock()
workQueue = Queue.Queue(1000000)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
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
print "Time: %s" % (time)

print "Exiting Main Thread"
