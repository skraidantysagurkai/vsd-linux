from openai import OpenAI
SYSTEM_PROMPT = """
You are a helpful assistant. Provide an answer if this command is malicious or not, based on the command and the user log history, timestamps are in unix time.
If the command is malicious answer with 1, if not answer with 0. There can only be two answers: 1 or 0.
The answer should be only a number, without any additional text or explanation.
"""
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
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": log,
                }
            ],
            temperature=0
        )
        return response
        
