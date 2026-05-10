from pymongo import MongoClient

from utils.ignore_list import DB_NAME, MONGO_URI

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

chunks_collection = db["chunks"]
