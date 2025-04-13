from pymongo import MongoClient
from typing import List, Dict

class ClientMongo():
    def __init__(self, db="dev"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db]

    def write_to_db(self, collection_str:str, doc:List[Dict]):
        """
        Write the parsed log to MongoDB.
        """
        collection = self.db[collection_str]
        try:
            collection.insert_many(doc)
            print(f"Write succesfull :)")
        except Exception as e:
            print(f"Error writing to MongoDB: {e}")
            traceback.print_exc()
        return

    def get_data(self, collection_str:str, query:Dict):
        """
        Get data from MongoDB.
        """
        collection = self.db[collection_str]
        try:
            data = collection.find(query)
            return list(data)
        except Exception as e:
            print(f"Error getting data from MongoDB: {e}")
            return []
    
    def close_connection(self):
        self.client.close()
