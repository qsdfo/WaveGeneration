# Sample RNN

Fichiers changés :
  three_tier_conditional.py
  four_tier.py


## Four-tier
Rien de bien différent, juste il faut préciser une taille pour la nouvelle échelle temporelle (frame_size_mid) :

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u \
    models/four_tier/four_tier.py --exp AXIS1 --seq_len 512 --big_frame_size 16 --frame_size_mid 8 \
    --frame_size 2 --weight_norm True --emb_size 64 --skip_conn False --dim 32 \
    --n_rnn 2 --rnn_type LSTM --learn_h0 False --q_levels 16 --q_type linear \
    --batch_size 128 --generation False

Attention, il faut toujours que la taille de la fenêtre du dessous divise la taille de la fenêtre du dessus. Ici 8 (frame_size_mid) divise 16 (big_frame_size) et 2 (frame_size) divise 8 (frame_size_mid).

## Conditional
### Important
- les séquences générées pendant la phase d'entrainement sont générées avec un conditionnement égal à 1 et n'ont donc pas d'intérêt

### Build database
* créer des fichiers .csv correspondant aux .wav (même longueur, une condition par sample). Le conditionnement est un entier (commencer à zéros, sinon ça crée des classes vides -> dimensions jamais vues pendant la training phase)
* découper en sous-parties les .wav :
    python preprocess_conditional.py chemin/vers/la/db
* concaténer les sous-parties en matrices de train/test :
    python _2npy_wav_conditional.py

### Train model


### Generate
- On génére à partir de fichiers de conditionnement au format .csv (texte)
- La liste des fichiers à générer est elle-même écrite dans un fichier d'indice au format .csv
- Le chemin de ce fichier est passé par l'argument CONDITIONING_INDEX
- checkpoint : path/to/pkl/params (in result folders)

Par exemple, si je veux générer des exemples à partir de deux fichiers de conditionnement rangés dans /home/daniele/cond_1.csv et /home/daniele/cond_2.csv, j'écris un fichier index_conditioning.csv qui contient :

/home/daniele/cond_1.csv
/home/daniele/cond_2.csv

et je lance le script three_tier_conditional.py avec par exemple les commandes suivantes :
