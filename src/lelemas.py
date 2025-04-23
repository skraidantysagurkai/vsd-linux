from openai import OpenAI

class LLM():
    def __init__(self, model_name: str):
        self.model_name = model_name
        endpoint = "https://models.github.ai/inference"
        token = ""
        self.client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )
        
    def classify_log(self, log:str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Classify if this log is malicious or not, based on the log and the user command history. Return the classification if malicious as a 1 and if not mailicious as a 0"
                },
                {
                    "role": "user",
                    "content": log,
                }
            ],
            temperature=0.7,
            max_tokens=100,
        )
        return response
        