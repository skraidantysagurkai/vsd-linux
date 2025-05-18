from pymongo import MongoClient
from typing import List, Optional, Dict
import numpy as np
import logging

from project.config.settings import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB_NAME]
        self.log_collection = db[settings.MONGODB_LOG_COLLECTION]
        self.embedded_collection = db[settings.MONGODB_EMBEDDED_COLLECTION]
        
        
    def get_embedded_features(self, pipeline:List) -> List[np.ndarray]:
        result = self.embedded_collection.aggregate(pipeline)
        
        if not list(result):
            logger.warning('No embedded hisotry found in database')
            return np.zeros((1, settings.EMBEDDING_DIM))
    
        array_of_embeds = [np.array(doc['features']) for doc in result]
        
        return array_of_embeds
    
    
    def get_regular_features(self, pipeline:List[dict], time_window:int) -> List[Dict]:
        result = self.log_collection.aggregate(pipeline)
        
        if not list(result):
            logger.info('No log history found in the database')
            if time_window == 30:
                return [{'log_count':0, 'success_rate':0, 
                         'bash_ratio':0, 'unique_pid_count':0}]
            if time_window == 300:
                return [{'log_count':0, 'success_rate':0, 'bash_ratio':0}]   
        
        return list(result)
        
    def insert_into_db(self, data:dict, type:str):
        if type == 'regular':
            self.log_collection.insert_one(data)
            logger.info('Log inserted into regular database')
            return
            
        if type == 'embedded':
            self.embedded_collection.insert_one(data)
            logger.info('Log inserted into regular database')
            return
        