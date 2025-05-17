from typing import List

def embedded_pipeline(log: dict, time: int) -> List[dict]:
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

def thirthy_sec_pipeline(log: dict):
    thirty_sec_pipeline= [
        {"$match": {
            "uid": log["uid"],
            "timestamp": {"$gte": log['timestamp'] - 30, "$lte": log['timestamp']}
        }},
        {"$group": {
            "_id": None,
            "unique_pids": {"$addToSet": "$pid"},
            "log_count": {"$sum": 1},
            "success_rate": {"$avg": "$success"},
            "total": {"$sum": 1},
            "bash_commands": {"$sum": {"$cond": [{"$eq": ["$is_bash", 1]}, 1, 0]}}
        }},
        {"$project": {
            "_id": 0,
            "pid": "$_id",
            "log_count": 1,
            "success_rate": 1,
            "bash_ratio": {
                "$cond": [
                    {"$gt": ["$log_count", 0]},
                    {"$divide": ["$bash_commands", "$log_count"]},
                    0
                ]
            },
            "unique_pid_count": {"$size":"$unique_pids"}
        }}
    ]
    
    return thirty_sec_pipeline

def five_min_pipeline(log: dict):
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
            "_id": 0,
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