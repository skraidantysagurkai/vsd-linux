from fastapi import FastAPI
from lelemas import LLM 

from src.shared.log_check import Log, LogPrompt

class LogAPI:
    def __init__(self, model_name: str):
        self.app = FastAPI()
        self.llm = LLM(model_name=model_name)
        self.setup_routes()
        
        
    def setup_routes(self):
        @self.app.post("/generate/") 
        def classify(log: Log):
            response = self.llm.classify_log(log.model_dump_json())
            return {
                "response": response,
            }
            
            
model_name = "openai/gpt-4o"
log_api = LogAPI(model_name=model_name)
app = log_api.app
