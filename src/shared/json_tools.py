import json


def write_json_long(data: list, output_file):
	with open(output_file, 'a') as f:
		for item in data:
			f.write(json.dumps(item) + '\n')


def load_json_long(path):
	with open(path, "r", encoding="utf-8") as f:
		return [json.loads(line) for line in f.readlines()]
