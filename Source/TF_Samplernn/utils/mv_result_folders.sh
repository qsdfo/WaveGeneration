# SOURCE="/fast-1/leo/WaveGeneration/TF_Samplernn/logdir/single_instrument/clarinet"
# DEST="/slow-2/leo/WaveGeneration/TF_Samplernn/clarinet_no_cond"

# for i in $(seq 90 95);
# do
# 	# echo $SOURCE/$i 
# 	# echo $DEST
# 	mv $SOURCE/$i $DEST
#     echo $i
# done

SOURCE="/fast-1/leo/WaveGeneration/TF_Samplernn/logdir/single_instrument/violin"
DEST="/slow-2/leo/WaveGeneration/TF_Samplernn/violin_no_cond"

for i in $(seq 63 83);
do
	# echo $SOURCE/$i 
	# echo $DEST
	mv $SOURCE/$i $DEST
    echo $i
done
