# Drivingdojo
DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model

## Get Started
Our code is based on the open-source project diffusers.

### setup
```bash
conda create -n dojo python=3.8
cd diffusers
pip install .
cd examples/text_to_image/
pip install -r requirements.txt
```

### Download the pretrained model
Huggingface: [img2video_1024_14f](https://huggingface.co/Yuqi1997/DrivingWorldModel/tree/main/img2video_1024_14f)

### inference
```bash
cd diffusers/examples/world_model
# img -> video
python inference/img2video.py
```

## Acknowledgement 
Many thanks to the following open-source projects:
* [diffusers](https://github.com/huggingface/diffusers)