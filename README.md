# Drivingdojo

The official implementation of the paper:

**DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model**
> [Yuqi Wang](https://robertwyq.github.io/), Ke Chen, [Jiawei He](https://jiaweihe.com/), Qitai Wang, Hengchen Dai, Yuntao Chen, Fei Xia, and Zhaoxiang Zhang
>
> 🎬 [video demos](https://drivingdojo.github.io/)

## 🕹️ Getting Started
Our code is based on the open-source project diffusers. The source code is organized in [diffusers/examples/world_model](diffusers/examples/world_model)

- [Installation](docs/INSTALL.md)
- [Dataset](docs/DATASET.md)
- [Training](docs/TRAINING.md)
- [Inference](docs/INFERENCE.md)

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

## Acknowledgement 
Many thanks to the following open-source projects:
* [diffusers](https://github.com/huggingface/diffusers)