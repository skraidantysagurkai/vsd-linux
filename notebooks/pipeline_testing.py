import requests
import json
import random as rd
from datetime import datetime
from pathlib import Path
from paths import DATA_DIR
from src.shared.json_tools import load_json_long
import time


def load_data(malicious_path: Path, benign_path: Path):
    """
    Load malicious and benign log data from JSON files.
    """
    malicious_data = [(load_json_long(file), file) for file in malicious_path.glob("*.json")]
    benign_data = [(load_json_long(file), file) for file in benign_path.glob("*.json")]
    return malicious_data, benign_data

if __name__ == "__main__":
    url = 'http://localhost:8000/predict'

    # Example list of data entries
    malicious_data, benign_data = load_data(DATA_DIR / "changed_uid/malicious", DATA_DIR / "changed_uid/benign")
    all_data = malicious_data + benign_data
    rd.shuffle(all_data)

    saved_path = DATA_DIR / 'results'
    saved_path.mkdir(parents=True, exist_ok=True)

    for file in all_data:
        print(f"Processing file: {file[1]}")
        file_name = file[1]
        file_stem = file_name.stem
        data = file[0]
        now = datetime.now()
        metrics = {}
        for idx, data in enumerate(data):
            try:
                response = requests.post(url, json=data)
                response.raise_for_status()
                result = response.json()
                if result == 1:
                    metrics = {'stopped': idx, 'time': str(datetime.now() - now)}
                    break
            except requests.exceptions.RequestException as e:
                print(f"[{idx+1}] Request failed: {e}")
        if metrics != {}:
            with open(str(saved_path / f"{file_stem}.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

