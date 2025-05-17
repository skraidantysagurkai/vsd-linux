'''
Functions for log operations
'''

from typing import List
from numpy.typing import NDArray
import numpy as np

def add_full_command_to_log(log:dict)-> dict:
    command = log['command']
    if log['arguments']:
        args = ' '.join(log['arguments'])
    else:
        args = ''
        
    log['full_command'] = command +' '+ args
    
    return log

def construct_features(thirty_sec_features, five_min_features,
                            thirty_sec_avg_embeds, five_min_avg_embeds) -> dict:
        
        features = {
            'thirty_sec_bash_count_rate':thirty_sec_features['bash_ratio'],
            'thirty_sec_avg_embedded_command_4': thirty_sec_avg_embeds[4],
            'thirty_sec_avg_embedded_command_0': thirty_sec_avg_embeds[0],
            'thirty_sec_avg_embedded_command_7': thirty_sec_avg_embeds[7],
            'thirty_sec_avg_embedded_command_8': thirty_sec_avg_embeds[8],
            'thirty_sec_avg_embedded_command_9': thirty_sec_avg_embeds[9],
            'thirty_sec_log_count': thirty_sec_features['log_count'],
            'thirty_sec_avg_embedded_command_5': thirty_sec_avg_embeds[5],
            'thirty_sec_avg_embedded_command_3': thirty_sec_avg_embeds[3],
            'thirty_sec_avg_embedded_command_2': thirty_sec_avg_embeds[2],
            'thirty_sec_avg_embedded_command_1': thirty_sec_avg_embeds[1],
            'thirty_sec_success_rate': thirty_sec_features['success_rate'],
            'thirty_sec_avg_embedded_command_6': thirty_sec_avg_embeds[6],
            'thirty_sec_unique_pids': thirty_sec_features['unique_pid_count'],
            'five_min_avg_embedded_command_4': five_min_avg_embeds[0],
            'five_min_bash_count_rate': five_min_features['bash_ratio'],
            'five_min_avg_embedded_command_5': five_min_avg_embeds[1],
            'five_min_success_rate': five_min_features['success_rate'],
            'five_min_log_count': five_min_features['log_count']
        }
    
        return features
    
def add_current_embeds_to_features(features:dict, current_embed:NDArray[np.float64]) -> dict:
        features['cur_event_avg_embedded_command_0'] = current_embed[0]
        features['cur_event_avg_embedded_command_6'] = current_embed[6]
        features['cur_event_avg_embedded_command_9'] = current_embed[9]
        features['cur_event_avg_embedded_command_7'] = current_embed[7]
        features['cur_event_avg_embedded_command_4'] = current_embed[4]
        features['cur_event_avg_embedded_command_2'] = current_embed[2]
        features['cur_event_avg_embedded_command_5'] = current_embed[5]
        features['cur_event_avg_embedded_command_3'] = current_embed[3]
        features['cur_event_avg_embedded_command_1'] = current_embed[1]
        features['cur_event_avg_embedded_command_8'] = current_embed[8]
        
        return features

    
def unpack_features_to_list(features: dict) -> List[float]:
        """
        Maximum LOL function
        """
        return [
            features['thirty_sec_bash_count_rate'],
            features['thirty_sec_avg_embedded_command_4'],
            features['thirty_sec_avg_embedded_command_0'],
            features['thirty_sec_avg_embedded_command_7'],
            features['thirty_sec_avg_embedded_command_8'],
            features['thirty_sec_avg_embedded_command_9'],
            features['thirty_sec_log_count'],
            features['thirty_sec_avg_embedded_command_5'],
            features['thirty_sec_avg_embedded_command_3'],
            features['thirty_sec_avg_embedded_command_2'],
            features['thirty_sec_avg_embedded_command_1'],
            features['thirty_sec_success_rate'],
            features['thirty_sec_avg_embedded_command_6'],
            features['cur_event_avg_embedded_command_0'],
            features['cur_event_avg_embedded_command_6'],
            features['thirty_sec_unique_pids'],
            features['five_min_avg_embedded_command_4'],
            features['five_min_bash_count_rate'],
            features['five_min_avg_embedded_command_5'],
            features['cur_event_avg_embedded_command_9'],
            features['cur_event_avg_embedded_command_7'],
            features['cur_event_avg_embedded_command_4'],
            features['cur_event_avg_embedded_command_2'],
            features['five_min_success_rate'],
            features['cur_event_avg_embedded_command_5'],
            features['five_min_log_count'],
            features['cur_event_avg_embedded_command_3'],
            features['cur_event_avg_embedded_command_1'],
            features['cur_event_avg_embedded_command_8']
        ]