from pymongo import MongoClient

client = MongoClient(maxPoolSize=None)
client = MongoClient('localhost', 27017)

db = client.test

table = db.table
post_id = table.insert_one({'u': 1, 't': 'test'}).inserted_id