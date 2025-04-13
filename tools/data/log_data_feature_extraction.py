import argparse
from pathlib import Path

from src.data.features.event_feature_extractor import EventFeatureExtractor
from src.shared.json_tools import load_json_long, write_json_long
from src.shared.logger import setup_logger

logger = setup_logger("info")


def main(input_dir, output_dir, batch_size):
	event_feature_extractor = EventFeatureExtractor(batch_size=batch_size)
	for file in input_dir.glob("*.json"):
		output_file = output_dir / f'{file.stem}.json'
		dataset = load_json_long(file)
		dataset = event_feature_extractor.extract_features(dataset)

		if len(dataset) > 0:
			write_json_long(dataset, output_file)
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

	main(input_dir, output_dir, batch_size)
