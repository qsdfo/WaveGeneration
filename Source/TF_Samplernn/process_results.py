import os
import numpy as np


def main(results_root):
	# Read all validation_loss.npy files
	config_folders = os.listdir(results_root)
	dict_result = {}
	for config_folder in config_folders:
		abs_config_folder = os.path.join(results_root, config_folder)
		if not os.path.isdir(abs_config_folder):
			continue
		val_tab = np.load(os.path.join(abs_config_folder,"validation_loss.npy"))
		# Get max
		min_loss = val_tab.min()
		dict_result[int(config_folder)] = min_loss
	
	# Write in a csv file
	with open(results_root + '/results.csv', 'w') as ff:
		for key, value in dict_result.items():
			ff.write("{:d};{:.5f}\n".format(key, value))


if __name__=="__main__":
	results_root = "/fast-1/leo/WaveGeneration/logdir"
	main(results_root)