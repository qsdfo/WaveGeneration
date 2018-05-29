import csv
import math
import numpy as np
import matplotlib.pyplot as plt

tiers_map = {'8,2,2': 0, '8,4,2': 1, '16,4,2': 2, '16,8,4': 3}
archi_map = {'1000': 0, '1000,1000': 1, '2000,2000': 2}

def main(root, instrus, parameter_0):
	color_map = {"clarinet_no_cond": 'b', 
		"violin_no_cond": 'g', 
		"flute_no_cond": 'r', 
		"oboe_no_cond": 'c', 
		"contrabass_no_cond": 'k'}
	for instru in instrus:
		x = {}
		# Load csv
		csv_file = root + '/' + instru + '/results.csv'
		with open(csv_file, 'r') as ff:
			reader = csv.DictReader(ff, delimiter=";")
			for row in reader:
				loss = float(row["loss"])
				if math.isnan(loss):
					continue
				if row[parameter_0] in list(x.keys()):
					x[row[parameter_0]].append(loss)
				else:
					x[row[parameter_0]] = [loss]

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

		xs = []
		labels = []
		for k, v in x.items():
			labels.append(k)
			xs.append(np.asarray(v))

		# Scatter plot with colors for instrument
		ax.boxplot(xs)
		ax.yaxis.grid(True)
		ax.set_xticks([y + 1 for y in range(len(labels))])
		ax.set_xlabel('Parameter')
		ax.set_ylabel('Loss')
		plt.setp(ax, xticks=[y + 1 for y in range(len(labels))], xticklabels=labels)
		plt.savefig(instru + "_" + parameter_0 + '.pdf')

if __name__ == '__main__':
	instrus = ["clarinet_no_cond", "violin_no_cond", "flute_no_cond", "oboe_no_cond", "contrabass_no_cond"]
	# instrus = ["clarinet_no_cond"]
	root = '/Users/leo/Recherche/WaveGeneration/Experiences/Hparams/SampleRNN/single_instruments'
	main(root, instrus, "rnns")
