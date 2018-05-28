import os
import json
import csv
import numpy as np


def main(results_root):
	# Read all validation_loss.npy files
	config_folders = os.listdir(results_root)
	dict_result = []
	fieldnames = None
	best_score = None
	mean_score = []
	best_config = None
	for config_folder in config_folders:
		abs_config_folder = os.path.join(results_root, config_folder)
		if not os.path.isdir(abs_config_folder):
			continue
		try:
			val_tab = np.load(os.path.join(abs_config_folder,"validation_loss.npy"))
		except:
			continue
		# Get max
		min_loss = val_tab.min()
		mean_score.append(min_loss)
		this_dict_result = {}
		this_dict_result["config_id"] = int(config_folder)
		this_dict_result["loss"] = min_loss

		# Get Hparams
		with open(abs_config_folder + '/hparam.json', 'r') as fp:
			hparam = json.load(fp)
		this_dict_result.update(hparam)

		if best_score is None:
			best_score = min_loss
		elif best_score > min_loss:
			best_score = min_loss
			best_config = config_folder

		dict_result.append(this_dict_result)
		if fieldnames is None:
			fieldnames = list(this_dict_result.keys())

	# Write in a csv file
	with open(results_root + '/results.csv', 'w') as ff:
		writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=';')
		writer.writeheader()
		for this_dict_result in dict_result:
			writer.writerow(this_dict_result)

	return np.mean(mean_score), best_score, best_config

if __name__=="__main__":
	db_list =["oboe_no_cond", "clarinet_no_cond", "violin_no_cond", "contrabass_no_cond", "flute_no_cond"]
	for db in db_list:
		results_root = "/slow-2/leo/WaveGeneration/TF_Samplernn/" + db
		mean_score, best_score, best_config = main(results_root)
