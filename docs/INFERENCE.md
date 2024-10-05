


## DrivingDojo Inference

### Download the pretrained model
Huggingface: [img2video_1024_14f](https://huggingface.co/Yuqi1997/DrivingWorldModel/tree/main/img2video_1024_14f)
Huggingface: [img2video_576_30f_action](https://huggingface.co/Yuqi1997/DrivingWorldModel/tree/main/img2video_576_30f_action)

### inference
```bash
cd diffusers/examples/world_model
# img -> video
python inference/img2video.py
# img -> video with action
python inference/img2video_action.py
```