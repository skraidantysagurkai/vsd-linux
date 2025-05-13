from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
from datetime import datetime, timedelta

class Log(BaseModel):
    timestamp: str
    success: int
    uid: str
    euid: str 
    syscall: str
    ppid: str
    pid: str
    command: str
    arguments: Optional[List[str]]
    CWD: Optional[str]
    
    
class MlAPI:
    def __init__(self):
        self.app = FastAPI()
        self.client = MongoClient("mongodb://localhost:27017/")
        db = self.client["ml_logs"]
        self.log_collection = db["logs"]
        self.embedded_collection = db["embedded_logs"]
        
    def get_history(self, log: Log):
        log_dict = log.model_dump_json()
        # 30 seconds window
        pipeline_30 = self.contruct_pipeline(log_dict, 30)
        # 300 seconds window
        pipeline_300 = self.contruct_pipeline(log_dict, 300)

        result_30 = list(self.embedded_collection.aggregate(pipeline_30))
        result_300 = list(self.embedded_collection.aggregate(pipeline_300))
        
        
    
    def save_log_to_mongo(self, log: Log):
        log_dict = log.model_dump_json()
        self.collection.insert_one(log_dict)

    def accept_request(self):
        @self.app.post("/predict")
        def predict(log: Log):
            pass
        
    @staticmethod
    def contruct_pipeline(log: dict, time: int) -> List[dict]:
        pipeline = [
            {"$match": {
                "uid": log["uid"],
                "timestamp": {"$gte": log['timestamp'] - time, "$lte": log['timestamp']}
            }},
            {"$group": {
                "_id": None,
                "count": {"$sum": 1},
                "f1_sum": {"$sum": "$features.f1"},
                "f2_sum": {"$sum": "$features.f2"},
                "f3_sum": {"$sum": "$features.f3"},
            }},
            {"$project": {
                "_id": 0,
                "average_f1": {"$divide": ["$f1_sum", "$count"]},
                "average_f2": {"$divide": ["$f2_sum", "$count"]},
                "average_f3": {"$divide": ["$f3_sum", "$count"]},
            }}
        ]
        
        return pipeline

            
        
ml_api = MlAPI()
app = ml_api.app