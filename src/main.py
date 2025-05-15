from fastapi import FastAPI
from typing import List, Optional
from pymongo import MongoClient
from datetime import datetime, timedelta
import numpy as np

from src.shared.log_check import Log
from src.ml_pipeline import MlPipeline


class MlAPI:
    def __init__(self):
        self.app = FastAPI()
        self.client = MongoClient("mongodb://localhost:27017/")
        self.ml_pipeline = MlPipeline()
        db = self.client["ml_logs"]
        self.log_collection = db["logs"]
        self.embedded_collection = db["embedded_logs"]
        
    def accept_request(self):
        @self.app.post("/predict")
        def predict(log: Log):
            self.save_log_to_db(log)
                
    def get_history(self, log: Log):
        log_dict = log.model_dump_json()
        # 30 seconds window
        pipeline_30 = self.contruct_pipeline(log_dict, 30)
        # 300 seconds window
        pipeline_300 = self.contruct_pipeline(log_dict, 300)

        result_raw_30 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(pipeline_30)]
        result_raw_300 = [np.array(doc['features']) for doc in self.embedded_collection.aggregate(pipeline_300)]
        
        thirty_sec_features = self.log_collection.aggregate(self.thirthy_sec_pipeline(log))
        
        # compute averages
        average_features_30 = np.mean(result_raw_30, axis=0)
        average_features_300 = np.mean(result_raw_300, axis=0)
        
        # we need all thirty sec embeds
        thirty_sec_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_30])
        # we only need number 5 and 6
        five_min_avg_embeds = self.ml_pipeline.pca_model.transform([average_features_300])[4:6]
        
                
    def save_log_to_db(self, log: Log):
        log_dict = log.model_dump_json()
        # save not embeded log
        log_dict["arg_count"] = len(log_dict["arguments"]) if log_dict["arguments"] else 0
        self.log_collection.insert_one(log_dict)
        
        embedded_command = self.ml_pipeline.feature_extractor.embed_log([log['command']])
        
        self.embedded_collection.insert_one(
            {
                "uid": log["uid"],
                "timestamp": log["timestamp"],
                "features": embedded_command
            }
        )
    
    # regular log example
    # {"timestamp": 1234567890, "success": 1, "uid": "1234", pid: "5678", "command": "ls", "arguments": ["-l"], "CWD": "/home/user"}
    
    # embedded log example
    # {"uid": "1234", "timestamp": 1234567890, "features": [0.2, ... , 0.3]}}
    
    
    
                
ml_api = MlAPI()
app = ml_api.app