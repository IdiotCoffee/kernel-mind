
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client.kernelmind

def save_entities(entities, repo_name):
    """
    entities: list of dicts returned by your AST parser
    """

    for ent in entities:
        ent["repo"] = repo_name
        ent["created_at"] = datetime.utcnow()

    result = db.entities.insert_many(entities)
    return result.inserted_ids
