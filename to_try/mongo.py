from pymongo import MongoClient
import re
import time

client = MongoClient("localhost", 27017)
db = client.local

def add_year():
    collection = db.movies
    ss = collection.find({})
    t1 = time.time()
    for s in ss:
        try:
            year = re.search(r"\d{4}", s["title"]).group()
            collection.update({"_id":s["_id"]},{"$set":{"year":int(year)}})
        except AttributeError as e:
            pass
    t2 = time.time()
    print("add_year:{:.3f}ms".format((t2-t1)*1000))
# add_year()

def change_to_hundred():
    collection = db.ratings
    ss = collection.find({})
    t1 = time.time()
    for s in ss:
        try:
            h = int(s["rating"])*20
            collection.update({"_id":s["_id"]},{"$set":{"hundred":h}})
        except AttributeError as e:
            pass
    t2 = time.time()
    print("change_to_hundred:{:.3f}ms".format((t2-t1)*1000))
change_to_hundred()



