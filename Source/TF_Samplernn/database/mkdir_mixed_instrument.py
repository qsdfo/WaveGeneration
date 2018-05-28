import os

def main(single_instrument_root):
	single_instrument_dirs = os.listdir(single_instrument_root)
	for single_instrument_dir_0 in single_instrument_dirs:
		for single_instrument_dir_1 in single_instrument_dirs:
			


if "__name__" == "__main__":
	single_instrument_root = '/fast-1/leo/WaveGeneration/Data/single_instrument'
	main(single_instrument_root)