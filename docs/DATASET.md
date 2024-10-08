# Dataset 

| Dataset      | videos |
|------------------|----------|
| nuScenes         | 1k     |
| Waymo            | 1k     |
| DrivingDojo      | 18k     |

## DrivingDojo Dataset


## nuScenes Dataset
Download the nuScenes dataset from the official website: [nuScenes](https://www.nuscenes.org/)

```bash
cd diffuers
# Extract 12Hz images from the nuScenes dataset
python examples/world_model/data_process/nusc_video.py --output_dir /path/to/extracted_videos
# Generate the training pickle
python examples/world_model/data_process/pickle_generation.py --video_path /path/to/extracted_videos --output_dir /path/to/pickle
```

## Waymo Dataset
```bash
cd diffuers

python examples/world_model/data_process/waymo_video.py --output_dir /path/to/extracted_videos
# Generate the training pickle
python examples/world_model/data_process/pickle_generation.py --video_path /path/to/extracted_videos --output_dir /path/to/pickle
```
