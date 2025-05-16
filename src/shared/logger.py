import logging
import torch

def setup_logger(log_level):
	global logger
	logging.basicConfig(
		level=log_level.upper(),
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)
	logger = logging.getLogger(__name__)

	# torch compile logs
	if log_level == "debug":
		torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

	return logger
