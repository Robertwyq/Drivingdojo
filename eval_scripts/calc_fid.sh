# Calculate FID
FAKE_DIR=.
REAL_DIR=.
# To process a real image dataset and save its features, uncomment and run this command:
# REAL_STATS=[file_name].npz
# python --save-stats $REAL_DIR $REAL_STATS --batch-size 768

# To calculate the FID between the generated image dataset and the preprocessed real image features, run this command:
python eval_scripts/calc_fid.py  $FAKE_DIR $REAL_DIR --batch-size 768 --gpu 0
