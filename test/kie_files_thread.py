import Queue
import threading
import cv2
import glob
# import kie_const
import numpy as np
import kie_image as image

class FilesThread(threading.Thread):
    def __init__(self, threadID, name, exitFlag, fileQueue, workQueue, handle):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.exitFlag = exitFlag
        self.fileQueue = fileQueue
        self.workQueue = workQueue
        self.handle = handle

    def run(self):
        # print "Starting " + self.name
        while not self.exitFlag:
            queueLock.acquire()
            if not self.workQueue.empty() and not self.filesQueue.full():
                task = self.workQueue.get()
                print "%d" % self.filesQueue.qsize()
                queueLock.release()

                self.handle(task, self.fileQueue)
            else:
                queueLock.release()
                time.sleep(5)
        print "Exiting " + self.name


def init(count, filesPath, handle):
    threadsName = []
    c = 0
    while c < count:
        threadsName.append("FileThread-%d" % c)
        c += 1

    files = []
    for f in glob.glob(filesPath):
        files.append(f)

    queue = Queue.Queue(len(files))

    threads = []
    threadID = 1
    for name in threadsName:
        thread = FilesThread(threadID, name, exitFlag, queue, workQueue, handle)
        thread.start()
        threads.append(thread)
        threadID += 1

    return files, threads




exitFlag = 0


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
