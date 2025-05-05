import argparse
from pathlib import Path

from src.data.aggregator_accelerated import DataAggregatorAccelerated
from src.shared.json_tools import load_json_long, write_json_long
from src.shared.logger import setup_logger
from src.data.splitter import split_dataset_by_uid
import time

logger = setup_logger("info")

def main(input_dir, output_dir, pca_path, num_jobs, cpu_batch_size, gpu_batch_size):
	event_feature_extractor = DataAggregatorAccelerated(pca_path=pca_path, num_jobs=num_jobs, cpu_batch_size=cpu_batch_size, gpu_batch_size=gpu_batch_size)
	for file in input_dir.glob("*.json"):
		dataset = load_json_long(file)
		dataset = sorted(dataset, key=lambda x: float(x['content']['timestamp']))
		dataset = split_dataset_by_uid(dataset)

		for key, sub_dataset in dataset.items():
			output_file = output_dir / f'{file.stem}_{key}.json'
			ids = [i.get('id') for i in sub_dataset]
			targets = [i.get('target') for i in sub_dataset]
			sub_dataset = [i.get('content') for i in sub_dataset]
			results = []
			for i, target, log in zip(ids, targets, event_feature_extractor.get_features(sub_dataset)):
				results.append({"id": i, "target": target, "content": log[0]})
				if len(results) >= 1000:
					write_json_long(results, output_file)
					results = []
			if len(results) > 0:
				write_json_long(results, output_file)
				logger.info(f"Processed uid - {key} - {output_file.name}")
		logger.info(f"Processed {file.name}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some data.')
	parser.add_argument('-input-dir', '-i', type=Path, help='The input directory containing data files')
	parser.add_argument('-output-dir', '-o', type=Path, help='The output directory to save processed data')
	parser.add_argument('-pca-path', '-pca', type=Path, help='Path of PCA model')
	parser.add_argument('-num_proc', '-n', type=int, help='Number of parallel processes', default=4)
	parser.add_argument('-gpu_batch_size', '-gpubs', type=int, default=1000)
	parser.add_argument('-cpu_batch_size', '-cpubs', type=int, default=100)

	args = parser.parse_args()

	input_dir = args.input_dir
	output_dir = args.output_dir
	pca_path = args.pca_path
	num_jobs = args.num_proc
	cpu_batch_size = args.cpu_batch_size
	gpu_batch_size = args.gpu_batch_size

	if not input_dir.exists():
		raise ValueError(f"Input directory {input_dir} does not exist")

	if not output_dir.exists():
		output_dir.mkdir(parents=True, exist_ok=True)

	now = time.time()
	main(input_dir, output_dir, pca_path, num_jobs, cpu_batch_size, gpu_batch_size)
	print(f"Time taken: {time.time() - now} seconds")
