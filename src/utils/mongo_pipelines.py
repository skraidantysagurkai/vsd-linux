from typing import List
from src.shared.log_check import Log

def embeded_pipeline(log: Log, time: int) -> List[dict]:
    pipeline = [
        {"$match": {
            "uid": log["uid"],
            "timestamp": {"$gte": log['timestamp'] - time, "$lte": log['timestamp']}
        }},
        {"$project": {
            "_id": 0,
            "features": 1
        }}
    ]
    
    return pipeline

def thirthy_sec_pipeline(log: Log):
    thirty_sec_pipeline= [
        {"$match": {
            "uid": log["uid"],
            "timestamp": {"$gte": log['timestamp'] - 30, "$lte": log['timestamp']}
        }},
        {"$group": {
            "_id": None,
            "_id": "$pid",
            "log_count": {"$sum": 1},
            "success_rate": {"$avg": "$success"},
            "total": {"$sum": 1},
            "bash_commands": {"$sum": {"$cond": [{"$eq": ["$is_bash", 1]}, 1, 0]}}
        }},
        {"$project": {
            "pid": "$_id",
            "log_count": 1,
            "success_rate": 1,
            "bash_ratio": {
                "$cond": [
                    {"$gt": ["$log_count", 0]},
                    {"$divide": ["$bash_commands", "$log_count"]},
                    0
                ]
        }
        }}
    ]
    
    return thirty_sec_pipeline

def five_min_pipeline(log: Log):
    five_min_pipeline = [
        {"$match": {
            "uid": log["uid"],
            "timestamp": {"$gte": log['timestamp'] - 300, "$lte": log['timestamp']}
        }},
        {"$group": {
            "_id": None,
            "log_count": {"$sum": 1},
            "success_rate": {"$avg": "$success"},
            "bash_commands": {"$sum": {"$cond": [{"$eq": ["$is_bash", 1]}, 1, 0]}}
        }},
        {"$project": {
            "log_count": 1,
            "success_rate": 1,
            "bash_ratio": {
                "$cond": [
                    {"$gt": ["$log_count", 0]},
                    {"$divide": ["$bash_commands", "$log_count"]},
                    0
                ]
        }
        }}
    ]
    
    return five_min_pipeline