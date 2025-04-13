import argparse
from pathlib import Path

import httpx

from src.shared.logger import setup_logger

logger = setup_logger("info")

from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from itertools import islice
import os

key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=key)


class Scores(BaseModel):
	event_id: int
	score: float


class AllEventScores(BaseModel):
	event_scores: list[Scores]


@retry(
	retry=retry_if_exception_type(httpx.HTTPStatusError),
	wait=wait_exponential(multiplier=1, min=1, max=60),
	stop=stop_after_attempt(5),
	reraise=True,
)
async def label_batch(batch):
	user_prompt = '''\n'''.join([f"{event["id"]}: {event["content"]}" for event in batch])
	completion = await client.chat.completions.create(
		model="gpt-3.5-turbo",  # replace with the model deployment name of your gpt-4o 2024-08-06 deployment
		messages=[
			{"role": "system",
			 "content": f'''Evaluate each Linux audit event for its potential risk of being a malicious command, based on the provided details. Assign a risk score to each event in the range 0, 1, where 1.0 indicates highly suspicious activity and 0.0 indicates benign behavior. For calibration: 'pwd' should receive a score around 0.1, 'ss' run by a non-root user should score around 0.5. You will be given {len(batch)} events. Return a JSON array of floats representing the risk scores, in the same order as the input events.'''},
			{"role": "user", "content": user_prompt},
		],
		temperature=0
		# response_format=AllEventScores,
	)
	try:
		results = json.loads(completion.choices[0].message.content)
		for i, (_, result) in enumerate(zip(batch, results)):
			if batch[i]['target'] == 0:
				batch[i]['target'] = result
		results = batch
	except json.JSONDecodeError as e:
		logger.error(f"JSON decode error: {e}")
		results = []
	return results


def write_json_long(data: list, output_file):
	with open(output_file, 'a') as f:
		for item in data:
			f.write(json.dumps(item) + '\n')


def load_json_long(path):
	with open(path, "r", encoding="utf-8") as f:
		return [json.loads(line) for line in f.readlines()]


def batch_iterable(iterable, batch_size=10):
	iterable = iter(iterable)
	while batch := list(islice(iterable, batch_size)):
		yield batch


async def main(input_dir, output_dir, batch_size):
	for file in input_dir.glob("*.json"):
		output_file = output_dir / f'{file.stem}.json'
		data = load_json_long(file)
		batches = batch_iterable(data, batch_size)
		tasks = [label_batch(batch) for batch in batches]
		results = []
		for task in tasks:
			results += (await asyncio.gather(task))[0]
			if len(results) >= 100:
				write_json_long(results[:100], output_file)
				results = results[100:]
			await asyncio.sleep(0.5)

		if len(results) > 0:
			write_json_long(results, output_file)
		logger.info(f"Processed {file.name}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some data.')
	parser.add_argument('-input-dir', '-i', type=Path, help='The input directory containing data files')
	parser.add_argument('-output-dir', '-o', type=Path, help='The output directory to save processed data')
	parser.add_argument('-batch-size', '-b', type=int, help='Size of batch', default=100)

	args = parser.parse_args()

	input_dir = args.input_dir
	output_dir = args.output_dir
	batch_size = args.batch_size

	if not input_dir.exists():
		raise ValueError(f"Input directory {input_dir} does not exist")

	if not output_dir.exists():
		output_dir.mkdir(parents=True, exist_ok=True)

	asyncio.run(main(input_dir, output_dir, batch_size))
