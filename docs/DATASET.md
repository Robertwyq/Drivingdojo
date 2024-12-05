# Dataset 

| Dataset      | videos |
|------------------|----------|
| nuScenes         | 1k     |
| Waymo            | 1k     |
| DrivingDojo      | 18k     |

## DrivingDojo Dataset

Download the DrivingDojo dataset from the huggingface website: [DrivingDojo](https://huggingface.co/datasets/Yuqi1997/DrivingDojo). Due to size limitations, the videos are split across multiple repositories, such as DrivingDojo-Extra1, DrivingDojo-Extra2, and so on.
Totally there are 45 tar.gz files, each containing about 400 videos. 

The subset of the dataset (action, interplay, open) is in data/dojo_subset.json.
```bash

The dataset structure is as follows:
- DrivingDojo
  - videos
    - video1
    - video2
    - ...
  - action_info
    - video1
    - video2
    - ...
  - camera_info_extrinsic_and_intrinsic
    - video1
    - video2
    - ...
  - meta.json
```

### 1. video generation
```bash
cd diffusers/examples/world_model/data_process/drivingdojo

python generate.py --path /path/to/drivingdojo/videos --save_path /path/to/drivingdojo_all.pkl --min_frames 30
```

### 2. action-conditioned video generation
```bash
cd diffusers/examples/world_model/data_process/drivingdojo

python generate_action.py --json_path /path/to/drivingdojo/meta.json --root /path/to/drivingdojo --output_pkl /path/to/drivingdojo_all_action.pkl --min_frames 30
```

## nuScenes Dataset
Download the nuScenes dataset from the official website: [nuScenes](https://www.nuscenes.org/)

### 1. video generation
```bash
cd diffuers
# Extract 12Hz images from the nuScenes dataset
python examples/world_model/data_process/nusc_video.py --output_dir /path/to/extracted_videos
# Generate the training pickle
python examples/world_model/data_process/pickle_generation.py --video_path /path/to/extracted_videos --output_dir /path/to/pickle
```

## Waymo Dataset

### 1. video generation
```bash
cd diffuers

python examples/world_model/data_process/waymo_video.py --output_dir /path/to/extracted_videos
# Generate the training pickle
python examples/world_model/data_process/pickle_generation.py --video_path /path/to/extracted_videos --output_dir /path/to/pickle
```
