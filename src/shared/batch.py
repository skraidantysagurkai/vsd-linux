from itertools import islice


def batch_iterable(iterable, batch_size=10):
	iterable = iter(iterable)
	while batch := list(islice(iterable, batch_size)):
		yield batch
