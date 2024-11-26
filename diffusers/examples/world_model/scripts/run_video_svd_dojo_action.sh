export MODEL_NAME="../pretrain/stable-video-diffusion-img2vid-xt" # SD官方模型路径 
export DATASET_NAME="../data/drivingdojo_all_action.pkl"

WORK=dojo_svd_front_action_576320_50f
EXP=dojo_svd_front_action

accelerate launch --main_process_port 12003 --mixed_precision="fp16" --num_processes 8 examples/world_model/train_video_svd_act.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps=40000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir="work_dirs/$WORK" \
  --dataloader_num_workers 6 \
  --nframes 50 \
  --fps 5 \
  --image_height 320 \
  --image_width 576 \
  --interval 1 \
  --conditions 'ego'\
  --conditioning_dropout_prob=0.2 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 10000 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --gradient_checkpointing \
  --report_to wandb \


