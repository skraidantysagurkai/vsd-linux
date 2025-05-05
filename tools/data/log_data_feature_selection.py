import argparse
from pathlib import Path

from src.shared.json_tools import load_json_long, write_json_long
from src.shared.logger import setup_logger
import time
import numpy as np

logger = setup_logger("info")
TOP_INDXS = [25, 15, 11, 18, 19, 20, 21, 16, 14, 13, 12, 26, 17, 0, 6, 27, 32, 42, 33, 9, 7, 4, 2, 43, 5, 38, 3, 1, 8]

def main(input_dir, output_dir):
    for file in input_dir.glob("*.json"):
        dataset = load_json_long(file)
        X = np.array([i['content'] for i in dataset])
        X = X[:, TOP_INDXS]
        for i, x in enumerate(X):
            dataset[i]['content'] = x.tolist()
        write_json_long(dataset, output_dir / file.name)
        logger.info(f"Processed {file.name}")


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
    now = time.time()
    main(input_dir, output_dir)
    print(f"Total time taken: {time.time() - now} seconds")
