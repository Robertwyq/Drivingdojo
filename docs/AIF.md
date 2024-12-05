# Action Instruction Following Metric

## Required Environments:
- `detectron2` ([GitHub Repository](https://github.com/facebookresearch/detectron2))
- `colmap` ([GitHub Repository](https://github.com/colmap/colmap))
- `tqdm`
- `pyquaternion`
- `scipy`

## Usage

### 0. Structure of Generated Images

Please organize your generated images as follows:

```plaintext
dojo_aif_metric/
│
├── output/images/
│   ├── scene1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── scene2/
│   └── ...
│
└── output/motions/
    ├── scene1.pkl
    ├── scene2.pkl
    └── ...
```

### 1. generate foreground masks for generated images:
    python maskrcnn_mask.py --image_folder output/images --output_folder output/masks

### 2. run colmap:
please modify the image width/height/intrinsic in run_colmap/database.py, line 327-330, then run:

    bash run_colmap/run.bash ./output

### 3. compute AIF metric:
    python run_colmap/eval_poses.py --meta_folder ./output