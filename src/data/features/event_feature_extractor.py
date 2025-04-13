from typing import Dict, List

import numpy as np
import torch
from numpy.typing import ArrayLike
from transformers import AutoTokenizer, AutoModel

from paths import ROOT_DIR
from src.shared.batch import batch_iterable
from src.shared.json_tools import load_json_long

HIGH_RISK = [
	"/root", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/dev",
	"/snap", "/lost+found", "/initrd", "/var/lib", "/var/cache", "/"
]
MEDIUM_RISK = [
	"/var", "/tmp", "/run", "/srv", "/proc", "/sys", "/var/log", "/var/tmp"
]
LOW_RISK = [
	"/home", "/usr", "/opt", "/boot", "/media", "/mnt"
]


class EventFeatureExtractor:
	def __init__(self, batch_size=100):
		self.model_name = ''
		self.bash_commands = load_json_long(ROOT_DIR / "bash_commands.json")
		self.batch_size = batch_size

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")

		self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
		self.model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)

	def extract_features(self, dataset: ArrayLike, with_labels: bool = True) -> List[Dict]:
		if with_labels:
			targets = [i.get("target") for i in dataset]
		else:
			targets = None
		features = [i.get("content") for i in dataset]
		ids = [i.get("id") for i in dataset]

		for entry in features:
			entry["command"] = self.create_command_string(entry.get("command"), entry.get("args"))

			entry["cwd_risk"] = self.assess_cwd_risk(entry.get("cwd"))
			entry["is_bash_command"] = self.is_bash_command(entry.get("command"))
			entry["flag_count"] = self.flag_count(entry.get("command"))
			entry["args_count"] = self.args_count(entry.get("command")) - entry["flag_count"]

		features = self.get_embedded_command(features, self.batch_size)
		if with_labels:
			return [{'id': id_, 'target': target, 'content': content} for id_, target, content in
			        zip(ids, targets, features)]
		else:
			return [{'id': id_, 'content': content} for id_, content in zip(ids, features)]

	def get_embedded_command(self, dataset, batch_size=100):
		batches = batch_iterable(dataset, batch_size)
		results = []
		for batch in batches:
			embedded = self.embed_command([entry.get("command") for entry in batch])
			results.append(embedded)
		results = np.concatenate(results, axis=0)

		for entry, rez in zip(dataset, results):
			entry["embedded_command"] = list(rez.astype(float))
		return dataset

	@staticmethod
	def create_command_string(command: str | None, args: list) -> str | None:
		"""
		Create a command string from the command and arguments.
		"""
		if command is None:
			return None
		elif args:
			args = " ".join(args)
			return args
		else:
			return command

	@staticmethod
	def assess_cwd_risk(cwd: str) -> int:
		if not cwd:
			return 3  # Treat None as high risk

		cwd = cwd.strip().lower()
		for path in HIGH_RISK:
			if cwd.startswith(path):
				return 2
		for path in MEDIUM_RISK:
			if cwd.startswith(path):
				return 1
		for path in LOW_RISK:
			if cwd.startswith(path):
				return 0
		return 3  # Default conservative fallback

	def is_bash_command(self, command: str) -> int:
		if not command:
			return 0

		if "/" in command.split()[0]:
			if any(f in self.bash_commands + ["sh"] for f in command.split()[0].split("/")):
				return 1
		else:
			if command in self.bash_commands + ["sh"]:
				return 1
		return 0

	@staticmethod
	def flag_count(command: str):
		if not command:
			return 0  # Treat None as high risk

		command = command.split()[1:]
		return sum(1 for part in command if part.startswith('-'))

	@staticmethod
	def args_count(command: str):
		if not command:
			return 0  # Treat None as high risk

		command = command.split()[1:]
		return len(command)

	def embed_command(self, batch):
		inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(self.device)
		with torch.no_grad():  # Disable gradients for efficiency
			outputs = self.model(**inputs)
		return torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
