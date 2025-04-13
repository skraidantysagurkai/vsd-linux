import json
import time 

def read_фаил(файл):
    with open(файл, 'r') as f:
        data = [json.loads(line) for line in f]
        
    логсы = []

    for item in data:
        логсы.append(item["content"])
    
    return data, логсы
    
    
def find_индекс(logs):
    log_indexes = []

    for idx, log in enumerate(logs):
        i = 1
        log_time = float(log['timestamp'])
        inter_found = False
        while True:
            try:
                check_time = float(logs[idx - i]['timestamp'])
                
                if log_time - check_time > 30 and not inter_found:
                    inter_idx = idx - (i-1)
                    inter_found = True

                
                if log_time - check_time > 330:
                    start_idx = idx - (i-1)
                    break
            except IndexError:
                start_idx = idx
                inter_idx = idx
                break
            i += 1
        log_indexes.append((start_idx, inter_idx, idx))
    
    return log_indexes

def get_features(logs, log_indexes):
    event_metrics = []
    
    for start_idx, inter_idx, event_idx in log_indexes:
        sum_cwd_risk_score = 0
        sum_arg_count = 0
        sum_flag_count = 0
        bash_count = 0
        if event_idx == inter_idx:
            avg_cwd_risk_score = float(logs[event_idx]['cwd_risk_score'])
            avg_arg_count = float(logs[event_idx]['args_counts'])
            avg_flag_count = float(logs[event_idx]['flag_count'])
            bash_count = float(logs[event_idx]['is_bash'])
        else:
            for i in range(start_idx, inter_idx + 1):
                # шитаем среднии значения
                sum_cwd_risk_score += float(logs[i]['cwd_risk_score'])
                sum_arg_count += float(logs[i]['args_counts'])
                sum_flag_count += float(logs[i]['flag_count'])
                bash_count += float(logs[i]['is_bash'])
                
            avg_cwd_risk_score = sum_cwd_risk_score / (inter_idx - start_idx + 1)
            avg_arg_count = sum_arg_count / (inter_idx - start_idx + 1)
            avg_flag_count = sum_flag_count / (inter_idx - start_idx + 1)
            
        event_metrics.append({
                              'timestamp': logs[event_idx]['timestamp'],
                              'log_count': (inter_idx - start_idx + 1),
                              'avg_cwd_risk_score': avg_cwd_risk_score,
                              'sum_cwd_risk_score': sum_cwd_risk_score,
                              'avg_arg_count': avg_arg_count,
                              'sum_arg_count': sum_arg_count,
                              'avg_flag_count': avg_flag_count,
                              'sum_flag_count': sum_flag_count,
                              'bash_count': bash_count})
        
    return event_metrics
        
            
                

data, logs = read_фаил('test_data.json')
log_indexes = find_индекс(logs)

event_metrics = get_features(logs, log_indexes)

metrics_jsons = []
for log, line in zip(event_metrics, data):
    metrics_jsons.append({'target': line['target'], 'id':line['id'], 'content': log})
    
with open('metrics.json', 'w') as f:
    for metric in metrics_jsons:
        json.dump(metric, f)
        f.write('\n')






    
    
    