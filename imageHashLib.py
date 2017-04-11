from PIL import Image
import imagehash

# hash = imagehash.phash(Image.open('./images/1.jpg'), 8)
# print(hash)
# otherhash = imagehash.phash(Image.open('./images/2.jpg'), 8)
# print(otherhash)
# print(hash == otherhash)
# print(hash - otherhash)

hash = imagehash.whash(Image.open('./images/1.jpg'))
print(hash)
otherhash = imagehash.whash(Image.open('./images/2.jpg'))
print(otherhash)
print(hash == otherhash)
print(hash - otherhash)