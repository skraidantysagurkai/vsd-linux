import argparse
from pathlib import Path

import pandas as pd

from src.shared.logger import setup_logger

logger = setup_logger("info")

import re
import json


def parse_audit_log(log):
	data = {}

	# Extract timestamp
	timestamp_match = re.search(r'audit\((\d+\.\d+):\d+\)', log)
	data['timestamp'] = timestamp_match.group(1) if timestamp_match else None

	# Extract success status
	success_match = re.search(r'success=(\w+)', log)
	data['success'] = (1 if success_match.group(1) == 'yes' else 0) if success_match else None

	# Extract UID, EUID, syscall
	uid_match = re.search(r'uid=(\d+)', log)
	euid_match = re.search(r'euid=(\d+)', log)
	syscall_match = re.search(r'syscall=(\d+)', log)

	data['uid'] = uid_match.group(1) if uid_match else None
	data['euid'] = euid_match.group(1) if euid_match else None
	data['syscall'] = euid_match.group(1) if syscall_match else None

	# Extract process IDs
	ppid_match = re.search(r'ppid=(\d+)', log)
	pid_match = re.search(r'pid=(\d+)', log)

	data['ppid'] = ppid_match.group(1) if ppid_match else None
	data['pid'] = pid_match.group(1) if pid_match else None

	# Extract command
	command_match = re.search(r'comm="([^"]+)"', log)
	data['command'] = command_match.group(1) if command_match else None

	# Extract arguments
	args_match = re.findall(r'a\d+="([^"]+)"', log)
	data['arguments'] = args_match if args_match else None

	# Extract CWD
	cwd_match = re.search(r'type=CWD.*?cwd="([^"]+)"', log)
	data['CWD'] = cwd_match.group(1) if cwd_match else None

	return data


def write_json_long(data: list, output_file):
	with open(output_file, 'w') as f:
		for item in data:
			f.write(json.dumps(item) + '\n')


def process_data(input_dir, output_dir):
	i = 0
	le = 0
	for file in input_dir.glob("*.csv"):
		logger.info(f"Processing file {file.name}")
		df = pd.read_csv(file)
		df = df[df['_source.decoder.name'] == "auditd"]
		target = [1 if i != " " else 0 for i in df['_source.rule.mitre.tactic'].to_list()]
		logs = [parse_audit_log(i) for i in df['_source.full_log'].to_list()]
		logs = [i for i in logs if i['command'] is not None]
		if len(logs) == 0:
			continue
		processed_data = [{"id": i + 1, "target": t, "content": l} for i, (t, l) in enumerate(zip(target, logs))]
		output_file = output_dir / f'{file.stem}.json'
		write_json_long(processed_data, output_file)
		logger.info(f"Processed {file.name}, contains {sum(target)} malicious logs")
		i += sum(target)
		le += len(target)
	logger.info(f"Processed and saved all files to {output_dir}, contains {i} malicious logs out of {le}")


def args_parser():
	parser = argparse.ArgumentParser(description='Process some data.')
	parser.add_argument('input_dir', type=str, help='The input directory containing data files')
	parser.add_argument('output_dir', type=str, help='The output directory to save processed data')
	return parser.parse_args()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some data.')
	parser.add_argument('-input-dir', '-i', type=Path, help='The input directory containing data files')
	parser.add_argument('-output-dir', '-o', type=Path, help='The output directory to save processed data')

	args = parser.parse_args()

	input_dir = args.input_dir
	output_dir = args.output_dir

	if not input_dir.exists():
		raise ValueError(f"Input directory {input_dir} does not exist")

	if not output_dir.exists():
		output_dir.mkdir(parents=True, exist_ok=True)

	process_data(input_dir, output_dir)
