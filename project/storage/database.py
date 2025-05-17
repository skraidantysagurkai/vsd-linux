from pymongo import MongoClient
from project.config.settings import settings
from typing import List
import numpy as np

class Database:
    def __init__(self):
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB_NAME]
        self.log_collection = db[settings.MONGODB_LOG_COLLECTION]
        self.embedded_collection = db[settings.MONGODB_EMBEDDED_COLLECTION]
        
        
    def get_embedded_features(self, pipeline:List) -> List[np.ndarray]:
        result = self.embedded_collection.aggregate(pipeline)
        
        if not result:
            return np.zeros((1, settings.EMBEDDING_DIM))[0]  
    
        array_of_embeds = [np.array(doc['features']) for doc in result]
        
        return array_of_embeds
    
    
    def get_regular_features(self, pipeline:List) -> List:
        result = self.log_collection.aggregate(pipeline)
        
        if not result:
            return 
        
        return list(result)
        
    def insert_into_db(self, data:dict, type:str):
        if type == 'regular':
            self.log_collection.insert_one(data)
            return
            
        if type == 'embedded':
            self.embedded_collection.insert_one(data)
            return
        
db = Database()