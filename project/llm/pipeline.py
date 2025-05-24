import logging

from project.config.settings import settings

from datetime import datetime
from typing import List

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)

class LLM():
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.pipe = self._set_up()

    def _set_up(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def build_log_line(self, log: dict):
        # Convert timestamp to readable date (UTC)
        try:
            readable_time = datetime.fromtimestamp(float(log.get("timestamp", 0))).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            readable_time = "Invalid timestamp"

        # Format command and arguments
        command = log.get("command", "<no command>")
        args = log.get("arguments", [])
        if isinstance(args, list):
            full_command = ' '.join(args)
        elif args is None:
            full_command = command
        else:
            full_command = str(args)

        # Format other fields with labels
        fields = [
            f"Time: {readable_time}",
            f"Success: {'Yes' if log.get('success') == 1 else 'No'}",
            f"UID: {log.get('uid', '?')} | EUID: {log.get('euid', '?')}",
            f"Syscall: {log.get('syscall', '?')}",
            f"PID: {log.get('pid', '?')} | PPID: {log.get('ppid', '?')}",
            f"CWD: {log.get('CWD', '[unknown]')}",
            f"Command: {full_command}"
        ]

        return " ".join(fields).strip()

    def build_prompt(self, current_event: dict, data: list):
        log_history = "\n".join([self.build_log_line(log) for log in data]).strip()
        current_command = self.build_log_line(current_event).strip()
        return (
            f"""
            You are a security analyst reviewing system command logs. 
            Given the following log history and current flagged command, classify it as either 'malicious' or 'benign', and explain the reasoning behind your classification.

            Log history:
            {log_history}
            
            Current command:
            {current_command}

            Respond in the following format:

            ###Classification: <malicious or benign>  
            ###Reason: <brief explanation>
            """)

    def parse_answer(self, answer: str):
        """
        Parse the model's answer to extract classification and reasoning.
        """
        try:
            classification = answer.split("###Classification:")[-1].strip().split("\n")[0].lower()
            binary_class = 1 if "malicious" in classification else 0
            reasoning = answer.split("###Reason:")[-1].strip()
        except Exception as e:
            print(f"Error parsing answer: {e}")
            reasoning = "unknown"
            binary_class = None
        return binary_class, reasoning

    def classify_log(self, log: dict, user_history: List[dict]):
        prompt = self.build_prompt(log, user_history)

        response = self.pipe(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            top_p=1.0,
        )
        label, reasoning = self.parse_answer(response[0]['generated_text'])
        return label, reasoning
