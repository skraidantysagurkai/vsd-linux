import numpy as np
import joblib
from joblib import Parallel, delayed


def get_labels():
	agg_labels = [
		'log_count',
		'cwd_avg_risk_score',
		'avg_arg_count',
		'avg_flag_count',
		'bash_count_rate',
		'success_rate',
		'unique_pids',
		'avg_embedded_command'
	]

	five_min_labels = [f'five_min_{label}' for label in agg_labels]
	thirty_sec_labels = [f'thirty_sec_{label}' for label in agg_labels]
	event_labels = [
		'success',
		'pid',
		'embedded_command'
	]

	return event_labels + thirty_sec_labels + five_min_labels

class DataAggregator:
	def __init__(self, pca_path: str, num_jobs: int, window_sizes_sec: tuple = (30, 300)):
		self.pca = joblib.load(pca_path)
		self.num_jobs = num_jobs

		self.window_one = window_sizes_sec[0]
		self.window_two = window_sizes_sec[1] + self.window_one

	def find_idx(self, logs):
		log_indexes = []

		for idx, log in enumerate(logs):

			log_time = float(log['timestamp'])
			inter_idx = idx
			start_idx = idx

			for i in range(idx - 1, -1, -1):
				check_time = float(logs[i]['timestamp'])
				delta = log_time - check_time

				if delta <= self.window_one and i <= inter_idx:
					inter_idx = i
				if delta <= self.window_two and i <= start_idx:
					start_idx = i
				else:
					break

			log_indexes.append((start_idx, inter_idx, idx))
		return log_indexes

	def compute_window_metrics(self, window, default_dim=10):
		if not window:
			return (
				0,  # log_count
				0,  # cwd_avg_risk_score
				0,  # avg_arg_count
				0,  # avg_flag_count
				0,  # bash_count_rate
				0,  # success_rate
				0,  # unique_pids
				[0] * default_dim  # avg_embedded_command
			)

		log_count = len(window)
		avg_embedded_command = np.mean([i['embedded_command'] for i in window], axis=0).reshape(1, -1)
		transformed_command = list(self.pca.transform(avg_embedded_command).flatten())

		return (
			log_count,
			float(np.mean([i['cwd_risk'] for i in window])),
			float(np.mean([i['args_count'] for i in window])),
			float(np.sum([i['flag_count'] for i in window])),
			float(np.sum([i['is_bash_command'] for i in window]) / log_count),
			float(np.sum([i['success'] for i in window]) / log_count),
			len(set(i['pid'] for i in window)),
			transformed_command
		)

	def process_single_event(self, logs, idx_tuple):
		start_idx, inter_idx, event_idx = idx_tuple

		if start_idx == inter_idx == event_idx:
			five_min_window = [logs[event_idx]]
			thirty_sec_window = [logs[event_idx]]
		elif start_idx == inter_idx:
			five_min_window = logs[start_idx:event_idx + 1]
			thirty_sec_window = five_min_window
		else:
			five_min_window = logs[start_idx:inter_idx]
			thirty_sec_window = logs[inter_idx:event_idx + 1]

		# Compute window-level metrics
		five_min_metrics = self.compute_window_metrics(five_min_window)
		thirty_sec_metrics = self.compute_window_metrics(thirty_sec_window)

		# Event-level metrics
		event = logs[event_idx]
		pid = event['pid'] if isinstance(event['pid'], int) else eval(event['pid'])
		embedded_command = list(self.pca.transform(np.array(event['embedded_command']).reshape(1, -1)).flatten())

		event_metrics = (
			event['success'],
			pid,
			embedded_command
		)

		return tuple(
			event_metrics + five_min_metrics + thirty_sec_metrics
		)

	def get_features(self, logs, log_indexes):
		return Parallel(n_jobs=self.num_jobs)(
			delayed(self.process_single_event)(logs, idx) for idx in log_indexes
		)