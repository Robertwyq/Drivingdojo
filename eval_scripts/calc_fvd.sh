# Calculate FVD from the video
FAKE_PATH=.
REAL_PATH=.

REAL_FEAT_CACHE="0" # Preprocessed features of the real video dataset

python ./eval_scripts/calc_fvd.py --fake_path $FAKE_PATH --real_path $REAL_PATH --n_runs 10 --gpu 0 --batch_size 16 --num_frames 16