
# file_list=(070823_s20-302_1701219822.0_1701219840.0)
# file_list=(061400_s20-122_1710115970.0_1710115986.0)
META_PATH=$1
SCRIPT_PATH=$2
IND=$3
export CUDA_VISIBLE_DEVICES=$IND
file_list=($(python run_colmap/listdir.py $META_PATH/images $IND))
cp $SCRIPT_PATH/database.py $META_PATH/images/
cd $META_PATH/images
# 2670674176367830809_180_000_200_000
# 迭代文件名列表并执行命令
for file_name in "${file_list[@]}"; do
    echo $file_name
    cd $file_name
    rm -r sparse
    # mv sparse sparse_0
    rm -r database.db
    rm database.py
    mkdir images
    mv ./*.jpg images/

    # colmap automatic_reconstructor --workspace_path ./ --image_path ./images --single_camera 1 --dense 0 --data_type video --quality high --camera_model RADIAL
    cp ../database.py ./
    python database.py
    mkdir sparse
    PROJECT_PATH=./
    colmap feature_extractor --database_path $PROJECT_PATH/database.db --image_path $PROJECT_PATH/images  --ImageReader.mask_path /mnt/nvme1n1p1/wqt_pack/dojo_aif_metric/output/masks/$file_name
    colmap exhaustive_matcher --database_path $PROJECT_PATH/database.db --SiftMatching.use_gpu 1
    colmap mapper --database_path $PROJECT_PATH/database.db --image_path $PROJECT_PATH/images --output_path $PROJECT_PATH/sparse --Mapper.min_num_matches 5 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0 --Mapper.ba_local_num_images 15 --Mapper.init_max_forward_motion 1.0

    # cd sparse/0
    # colmap model_converter --input_path . --output_path . --output_type TXT
    #
    if [ -d "sparse/0" ]; then
        cd sparse/0
        colmap model_converter --input_path . --output_path . --output_type TXT 
        cd ../..
    fi
    cd ..
done

# colmap feature_extractor --database_path $PROJECT_PATH/database.db --image_path $PROJECT_PATH/images --ImageReader.camera_model PINHOLE
# colmap exhaustive_matcher --database_path $PROJECT_PATH/database.db --SiftMatching.use_gpu 1
# colmap mapper --database_path $PROJECT_PATH/database.db --image_path $PROJECT_PATH/images --output_path $PROJECT_PATH/sparse --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0

