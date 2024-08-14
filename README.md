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

### inference
```bash
cd diffusers/examples/world_model
# img -> video
python inference/img2video.py
```

## Acknowledgement 
Many thanks to the following open-source projects:
* [diffusers](https://github.com/huggingface/diffusers)