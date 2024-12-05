META_PATH=$1
SCRIPT_PATH=$2
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 0 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 1 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 2 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 3 & 
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 4 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 5 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 6 &
bash run_colmap/colmap.bash $META_PATH $SCRIPT_PATH 7 &