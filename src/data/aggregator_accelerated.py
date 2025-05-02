import numpy as np
from joblib import Parallel, delayed
from typing import Dict, List
from src.shared.batch import batch_iterable
import cupy as cp
import pickle

def get_labels():
    agg_feature_labels = [
        'log_count',
        'cwd_avg_risk_score',
        'avg_arg_count',
        'avg_flag_count',
        'bash_count_rate',
        'success_rate',
        'unique_pids',
    ]
    agg_command_labels = [f'avg_embedded_command_{i}' for i in range(10)]


    five_min_feature_labels = [f'five_min_{label}' for label in agg_feature_labels]
    thirty_sec_feature_labels = [f'thirty_sec_{label}' for label in agg_feature_labels]

    five_min_command_labels = [f'five_min_{label}' for label in agg_command_labels]
    thirty_sec_command_labels = [f'thirty_sec_{label}' for label in agg_command_labels]
    event_command_labels = [f'cur_event_{label}' for label in agg_command_labels]

    event_feature_labels = [
        'success',
    ]

    return event_command_labels + event_feature_labels + thirty_sec_command_labels + thirty_sec_feature_labels + five_min_command_labels + five_min_feature_labels

COLUMNS = ["timestamp", "success", "pid", "cwd_risk", "is_bash_command", "flag_count", "args_count", "embedded_command"]

class DataAggregatorAccelerated:
    def __init__(self, pca_path: str, num_jobs: int, window_sizes_sec: tuple = (30, 300), cpu_batch_size: int = 100, gpu_batch_size: int = 1000):
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        self.num_jobs = num_jobs

        self.window_one = window_sizes_sec[0]
        self.window_two = window_sizes_sec[1] + self.window_one

        self.cpu_batch_size = cpu_batch_size
        self.gpu_batch_size = gpu_batch_size

        self.keys = None

    @staticmethod
    def prep_data(logs: List[Dict]):
        assert all(i in list(set(logs[0].keys())) for i in COLUMNS), "Log data is not in the expected format"
        embedded_commands = []
        features = []
        if 'embedded_command' in COLUMNS:
            COLUMNS.remove('embedded_command')
        keys = COLUMNS
        for log in logs:
            if any(log[key] in [None, ''] for key in keys) or log['embedded_command'] is None:
                continue
            embedded_commands.append(log['embedded_command'])
            features.append([log[key] for key in keys])
        return np.array(features, dtype=np.float32), np.array(embedded_commands, dtype=np.float32), keys


    def find_idx(self, timestamps): # This is quite fast
        log_indexes = []

        for idx, timestamp in enumerate(timestamps):

            log_time = float(timestamp)
            inter_idx = idx
            start_idx = idx

            for i in range(idx - 1, -1, -1):
                check_time = float(timestamps[i])
                delta = log_time - check_time

                if delta <= self.window_one and i <= inter_idx:
                    inter_idx = i
                if delta <= self.window_two and i <= start_idx:
                    start_idx = i
                else:
                    break

            log_indexes.append((start_idx, inter_idx, idx))
        return log_indexes

    def compute_window_metrics(self, window):
        if len(window) == 0:
            return (
                0,  # log_count
                0,  # cwd_avg_risk_score
                0,  # avg_arg_count
                0,  # avg_flag_count
                0,  # bash_count_rate
                0,  # success_rate
                0,  # unique_pids
            )

        if isinstance(window, list):
            window = np.array(window)

        log_count = len(window)

        return (
            log_count,
            float(np.mean(window[:, self.keys['cwd_risk']])),
            float(np.mean(window[:, self.keys['args_count']])),
            int(np.sum(window[:, self.keys['flag_count']])),
            float(np.sum(window[:, self.keys['is_bash_command']]) / log_count),
            float(np.sum(window[:, self.keys['success']]) / log_count),
            int(np.unique(window[:, self.keys['pid']]).size)
        )

    def process_events(self, logs, window_idxs):
        def process_batch(batch):
            results = []
            for index_tuple in batch:
                five_min_window, thirty_sec_window = self.create_window(logs, index_tuple)
                five_min_metrics = self.compute_window_metrics(five_min_window)
                thirty_sec_metrics = self.compute_window_metrics(thirty_sec_window)

                event = self.get_current_event(logs, index_tuple)
                event_metrics = (float(event[self.keys['success']]),)

                results.append((event_metrics, thirty_sec_metrics, five_min_metrics))
            return results

        for batch in batch_iterable(window_idxs, self.cpu_batch_size):
            batch_results = Parallel(n_jobs=self.num_jobs)(
                delayed(process_batch)([idx]) for idx in batch
            )
            for res in batch_results:
                yield from res

    def process_commands(self, embedded_commands, window_idxs):
        for batch in batch_iterable(window_idxs, self.gpu_batch_size):
            five_min_means = []
            thirty_sec_means = []
            current_event_arr = []

            for index_tuple in batch:
                five_min_window, thirty_sec_window = self.create_window(embedded_commands, index_tuple)
                five_min_means.append(np.mean(five_min_window, axis=0))
                thirty_sec_means.append(np.mean(thirty_sec_window, axis=0))
                current_event_arr.append(self.get_current_event(embedded_commands, index_tuple))

            # Stack and transform as arrays (no slow appending)
            five_min_arr = np.vstack(five_min_means)
            thirty_sec_arr = np.vstack(thirty_sec_means)
            current_event_arr = np.vstack(current_event_arr)

            five_min_transformed = self.pca.transform(cp.asarray(five_min_arr.astype(np.float16)))
            thirty_sec_transformed = self.pca.transform(cp.asarray(thirty_sec_arr.astype(np.float16)))
            current_event_arr = self.pca.transform(cp.asarray(current_event_arr.astype(np.float16)))

            five_min_transformed = cp.asnumpy(five_min_transformed)
            thirty_sec_transformed = cp.asnumpy(thirty_sec_transformed)
            current_event_arr = cp.asnumpy(current_event_arr)

            for current_event, thirty_sec, five_min in zip(current_event_arr, thirty_sec_transformed, five_min_transformed,):
                yield current_event, thirty_sec, five_min

    @staticmethod
    def construct_feature_vector(feature_metrics_tuple, commands_tuple):
        event_commands, thirty_sec_commands, five_min_commands = commands_tuple
        event_metrics, thirty_sec_metrics, five_min_metrics = feature_metrics_tuple
        return np.concatenate((event_commands, event_metrics, thirty_sec_commands,
                               thirty_sec_metrics, five_min_commands, five_min_metrics)).reshape(1, -1)

    @staticmethod
    def create_window(logs, window_idx_tuple):
        start_idx, inter_idx, event_idx = window_idx_tuple
        if start_idx == inter_idx == event_idx:
            five_min_window = [logs[event_idx]]
            thirty_sec_window = five_min_window
        elif start_idx == inter_idx:
            five_min_window = logs[start_idx:event_idx + 1]
            thirty_sec_window = five_min_window
        else:
            five_min_window = logs[start_idx:inter_idx]
            thirty_sec_window = logs[inter_idx:event_idx + 1]
        return five_min_window, thirty_sec_window

    @staticmethod
    def get_current_event(logs, window_idx_tuple):
        _, _, event_idx = window_idx_tuple
        return logs[event_idx]

    def _set_keys(self, keys):
        self.keys = {j: i for i, j in enumerate(keys)}

    def get_features(self, logs):
        logs = sorted(logs, key=lambda x: float(x['timestamp']))
        features, embedded_commands, keys = self.prep_data(logs)
        self._set_keys(keys)
        if len(features) == 0:
            return []
        window_idxs = self.find_idx(features[:, self.keys.get('timestamp')])
        for commands, features in zip(
            self.process_commands(embedded_commands, window_idxs),
            self.process_events(features, window_idxs)
        ):
            yield self.construct_feature_vector(features, commands).tolist()
