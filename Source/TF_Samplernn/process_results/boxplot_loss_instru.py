import csv
import re
import math
import numpy as np
import matplotlib.pyplot as plt

color_map = {"clarinet_no_cond": 0, 
		"violin_no_cond": 1, 
		"flute_no_cond": 2, 
		"oboe_no_cond": 3, 
		"contrabass_no_cond": 4}

def main(root, instrus):	
	xs = []
	labels = []
	for instru in instrus:
		# Load csv
		csv_file = root + '/' + instru + '/results.csv'
		losses=[]
		with open(csv_file, 'r') as ff:
			reader = csv.DictReader(ff, delimiter=";")
			for row in reader:
				loss = float(row["loss"])
				if math.isnan(loss):
					continue
				losses.append(loss)
		xs.append(np.asarray(losses))
		labels.append(re.sub("_no_cond", "", instru))
	
	fig, ax = plt.subplots(nrows=1, ncols=1)
	
	# Scatter plot with colors for instrument
	ax.boxplot(xs)
	ax.yaxis.grid(True)
	ax.set_xticks([y + 1 for y in range(len(labels))])
	ax.set_xlabel('Model')
	ax.set_ylabel('Loss')
	plt.setp(ax, xticks=[y + 1 for y in range(len(labels))],
		 xticklabels=labels)
	# plt.show()
	plt.savefig("boxplot_loss_model.pdf")

if __name__ == '__main__':
	instrus = ["clarinet_no_cond", "violin_no_cond", "flute_no_cond", "oboe_no_cond", "contrabass_no_cond"]
	root = '/Users/leo/Recherche/WaveGeneration/Experiences/Hparams/SampleRNN/single_instruments'
	main(root, instrus)