# Drivingdojo

The official implementation of the paper:

**DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model**
> [Yuqi Wang](https://robertwyq.github.io/), Ke Cheng, [Jiawei He](https://jiaweihe.com/), Qitai Wang, Hengchen Dai, Yuntao Chen, Fei Xia, and [Zhaoxiang Zhang](https://zhaoxiangzhang.net/)
>
> 📑 [paper](https://arxiv.org/abs/2410.10738) 🎬 [video demos](https://drivingdojo.github.io/) 📖 [dataset](https://huggingface.co/datasets/Yuqi1997/DrivingDojo) 📢[zhihu](https://zhuanlan.zhihu.com/p/1551246719)

<div id="top" align="center">
<p align="left">
<image src="assets/dojo2.gif" width="500px" >
</p>
</div>

## 🚀 News
- [2024-11] The code is support finetuned Stable Video Diffusion on multiple driving dataset.
- [2024-10] Our dataset DrivingDojo is released on **Huggingface**.
- [2024-9] Our paper is accepted by **NeurIPS 2024**.


## 🕹️ Getting Started
Our code is based on the open-source project diffusers. The source code is organized in [diffusers/examples/world_model](diffusers/examples/world_model)

The following table shows the supported finetuned methods and datasets in this repo, and we will update it continuously.

| Method / Dataset      | DrivingDojo | nuScenes | Waymo | OpenDV2K |
|-----------------------|-------|------------|------|-------------|
| Stable Video Diffusion  |   ✓   |      ✓     |   ✓    |            |
| Action-Conditioned Video Generation |   ✓   |           |       |            |


### 📦 Installation
- [Installation](docs/INSTALL.md)
- [Dataset](docs/DATASET.md)
- [Training](docs/TRAINING.md)
- [Inference](docs/INFERENCE.md)

## 🌟Citation
if you find our work useful in your research, please consider citing:
```bibtex
@article{wang2024drivingdojo,
  title={DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model},
  author={Wang, Yuqi and Cheng, Ke and He, Jiawei and Wang, Qitai and Dai, Hengchen and Chen, Yuntao and Xia, Fei and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2410.10738},
  year={2024}
}
```

## Acknowledgement 
Many thanks to the following open-source projects:
* [diffusers](https://github.com/huggingface/diffusers)