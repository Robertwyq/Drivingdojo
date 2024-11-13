# Calculate FID from the video

FAKE_DIR=.
REAL_DIR=.

N_FRAMES=10000 # The number of frames randomly selected from the generated videos and real ones.

# To calculate the FID between the generated video dataset and the preprocessed real video features, run this command:
python eval_scripts/calc_fid_videos.py  $FAKE_DIR $REAL_DIR  --n_runs 10 --n_frames $N_FRAMES
