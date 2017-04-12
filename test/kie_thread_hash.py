import queue
import threading
import cv2
import time
# import mysql_conn

import os, os.path
import kie_mysql as sql
import kie_image as image

HASH_PATH = '../images/hash/'
KPC_EXT = '.kpc'

THREAD_COUNT = 200
exitFlag = 0


class HashThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)

        while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                data = self.q.get()
                url = data['url']
                folder = '%s%s/' % (HASH_PATH, data['f'])
                # if self.folder > gl['folder']:
                #     gl['folder'] = self.folder
                #     self.newpath = '%s%s/' % (HASH_PATH, self.folder)
                #
                # if os.path.exists(self.newpath):
                #     if len([name for name in os.listdir(self.newpath)]) >= 2000:
                #         self.folder += 1
                #         gl['folder'] = self.folder
                #         print('new folder: %s' % self.folder)
                #         self.newpath = '%s%s/' % (HASH_PATH, self.folder)
                #         os.makedirs(self.newpath)
                # else:
                #     os.makedirs(self.newpath)

                queueLock.release()
                # fn = currQueue['fn']
                # urls = currQueue['urls']
                # kps = []
                # for url in urls:
                #     img = image.loadImageFromUrl(url, cv2.IMREAD_GRAYSCALE, True, 200)
                #     name = image.fileName(url)
                #     kp, des = image.keypointDesCalc(img)
                #     kps.append(image.pack(kp, des, name))
                #     time.sleep(0)
                # image.saveKps(kps, HASH_PATH + fn + KPC_EXT)


                img = image.loadImageFromUrl(url, cv2.IMREAD_GRAYSCALE, True, 200)
                name = image.fileName(url)
                image.keypointDesCalc(img, folder + name)


                # image.saveKps(kps, HASH_PATH + fn + KPC_EXT)
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
workQueue = queue.Queue(1000000)
threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = HashThread(threadID, tName, workQueue)
    thread.daemon = True
    thread.start()
    threads.append(thread)
    threadID += 1

# Fill the queue
# queueLock.acquire()
# tasks = []
# i = 0
# hc = 0
# for word in nameList:
#     tasks.append(word)
#     i += 1
#     if i == 100:
#         queue = {'fn': 'HASH %d' % hc, 'urls': list(tasks)}
#         workQueue.put(queue)
#         hc += 1
#         i = 0
#         tasks = []
# queueLock.release()

queueLock.acquire()
_folder = 0
i = 0
for word in nameList:
    workQueue.put({'url': word, 'f': _folder})
    i += 1
    if i == 1000:
        i = 0
        _folder += 1
        os.makedirs('%s%s/' % (HASH_PATH, _folder))
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
