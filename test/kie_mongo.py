from pymongo import MongoClient


def connect(_db, _collection):
    client = MongoClient()
    client = MongoClient('localhost', 27017)

    db = client[_db]
    collection = db[_collection]

    return collection


def insert(collection, data):
    collection.insert_one(data)


def getFile(collection, name):
    collection.find(name)
