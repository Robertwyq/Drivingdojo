export MODEL_NAME="../pretrain/stable-video-diffusion-img2vid-xt" # SD官方模型路径 
export DATASET_NAME="../data/nusc_video_front.pkl"

WORK=nusc_fsdp_svd_front_576320_30f
EXP=nusc_fsdp_svd_front

accelerate launch --config_file examples/world_model/configs/fsdp.yaml --main_process_port 12000 examples/world_model/train_video_svd_fsdp.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=40000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="work_dirs/$WORK" \
  --dataloader_num_workers 6 \
  --nframes 30 \
  --fps 5 \
  --image_height 320 \
  --image_width 576 \
  --conditioning_dropout_prob=0.2 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 20000 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --gradient_checkpointing \
  --report_to wandb \


