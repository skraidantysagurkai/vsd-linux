'''
Functions for log operations
'''

from typing import List
from numpy.typing import NDArray
import os
import numpy as np
import logging
import json

from paths import DOCUMENTATION_DIR, BASH_COMMANDS_PATH, TESTING_RES_DIR

logger = logging.getLogger(__name__)

class FeatureManager:
    def __init__(self):
        self.bash_commands = self.get_bash_commands()
    
    def get_bash_commands(self):
        bash_commands = []
        try:
            with open(BASH_COMMANDS_PATH, 'r') as f:
                    for line in f:
                        bash_commands.append(json.loads(line)['command'])
            return bash_commands
        except Exception as e:
            logger.error(f'Error loading bash commands \n {e}')
    
    def add_bash_feature(self, log:dict) -> dict:
        log['is_bash'] = int(log['command'] in self.bash_commands)
        return log
    
    @staticmethod
    def add_full_command_to_log(log:dict)-> dict:
        command = log['command']
        if log['arguments']:
            args = ' '.join(log['arguments'])
        else:
            args = ''
            
        log['full_command'] = command +' '+ args
        
        return log

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                

def get_command_doc(command:str) -> str:
    try:
        with open(os.path.join(DOCUMENTATION_DIR, f'{command}.txt')) as f:
            documentation = f.read()
            
        return documentation
    except FileNotFoundError:
        logger.error(f"No {command} documentation file found")
        return 'No documentation'
           
            
def construct_prompt(log:dict, history:dict) -> str:
    prompt = f'''
    ## MAIN COMMAND
    {log}
    ## USER HISTORY
    {history}
    ## MAIN COMMAND DOCUMENTATION
    {get_command_doc(log['command'])}
   '''
    
    return prompt


def save_response_to_file(target: int, xgboost_prediction: int, llm_prediction: int) -> None:
    response_json = {'target': target, 'xgboost_prediction': xgboost_prediction, 'llm_prediction': llm_prediction}
    # change file name according to the json file
    filename = os.path.join(TESTING_RES_DIR, f'31-December.json')
    try:
        with open(filename, 'w') as f:
            json.dump(response_json, f)
    except Exception as e:
        logger.error(f"Error saving response to file {filename}: {e}")
