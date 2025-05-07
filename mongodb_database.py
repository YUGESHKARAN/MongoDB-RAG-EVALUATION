from pymongo import MongoClient
from datetime import datetime
class MongoDBDatabase:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection_schema(self, collection_name):
        # Example method to get a sample document to infer schema
        sample_document = self.db[collection_name].find_one()
        return sample_document if sample_document else "No schema available"

    def run(self, collection_name, query ):
        # Example method to execute a MongoDB query
        collection = self.db[collection_name]
        
        
        results =list(eval(query))
        return results
