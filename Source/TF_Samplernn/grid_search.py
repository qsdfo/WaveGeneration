from sklearn.model_selection import ParameterGrid
import json

if __name__ =="__main__":

	# Switch for debugging when not on Guillimin
	LOCAL = True

	result_folder = "/tmp/leo/WaveGeneration/logdir"
	# Carreful ! Will erase previous results
	init_directory(result_folder)

	# Hparams
	hparams = {
		"l2_regularization_strength": [0, 1e-3],
		"sample_size": [2**15+8],
		"seq_len": [1024],
		"frames": [
			"32,16,8",
			"16,8,4", "16,4,2",
			"8,4,2", "8,2,2"
		],
		"rnns": [
			"2000, 2000",
			"1000, 1000",
			"1000"
		],
		"mlps": [		# Last layer will automatically be added to map toward a q_level space
			"1000,1000",
			"2000,2000"
		],
		"emb_size": [256],
		"q_levels": [256],
	}

	hparams = list(ParameterGrid(hparams))

	for index, hparam in enumerate(hparams):
		# Create folder
		param_folder = result_folder + '/' str(index)
		os.mkdir(param_folder)

		# Write config
		with open(param_folder + '/hparam.json', 'w') as fp:
    		json.dump(hparam, fp)

		# Pbs script
		file_pbs = param_folder + '/submit.pbs'

		script_name = "WaveGen_" + str(index)

		text_pbs = """#!/bin/bash

#PBS -j oe
#PBS -N job_outputs/""" + script_name + """
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=5:00:00

""" 
		
		if not LOCAL:
			text_pbs += """./start_env.sh
module load foss/2015b
module load Tensorflow/1.0.0-Python-3.5.2
source ~/Virtualenvs/tf_3/bin/activate

SRC=/home/crestel/WaveGeneration/Source/TF_Samplernn
cd $SRC"""
		
		text_pbs += """
python main.py \
	--num_gpus=0 \
	--batch_size=64 \
	--data_dir=\
	--logdir_root=""" + param_folder + """ \
	--checkpoint_every=10 \
	--num_steps=100000 \
	--learning_rate=1e-3 \
	--l2_regularization_strength=""" + hparam["l2_regularization_strength"] + """ \
	--optimizer=adam \
	--momentum=0.9 \
	--sample_rate=8000 \
	--sample_size=""" + hparam["sample_size"] + """ \
	--sliding_ratio=0.75 \
	--silence_threshold=0.01 \
	--seq_len=""" + hparam["seq_len"] + """ \
	--frames=""" + hparam["frames"] + """ \
	--q_levels=""" + hparam["q_levels"] + """ \
	--rnns=""" + hparam["rnns"] + """ \
	--mlps=""" + hparam["mlps"] + """ \
	--emb_size=""" + hparam["emb_size"] + """ \
	--max_checkpoints=5 \
	--num_example_generated=5 \
	--load_existing_model=False \
	--summary=False"""

		with open(file_pbs, 'wb') as f:
			f.write(text_pbs)

		if LOCAL:
			subprocess.check_output("./" + file_pbs, shell=True)
		else:
			subprocess.check_output('qsub ' + file_pbs, shell=True)