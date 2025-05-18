from openai import OpenAI
from typing import List
import logging

from project.config.settings import settings
from project.utils.handy_functions import construct_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant. Provide an answer if this command is malicious or not, based on the command and the user log history, timestamps are in unix time.
If the command is malicious answer with 1, if not answer with 0. There can only be two answers: 1 or 0.
The answer should be only a number, without any additional text or explanation.
"""

class LLM():
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        endpoint = settings.LLM_ENDPOINT
        token = settings.GITHUB_TOKEN
        self.client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )
        
    def classify_log(self, log:dict, user_history: List[dict]) -> dict:
        prompt = 
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0
        )
        
        print(response.choices[0].message.content)
        
        return response
        
