# Dataset 

| Dataset      | videos |
|------------------|----------|
| nuScenes         | 1000     |

## nuScenes Dataset
Download the nuScenes dataset from the official website: [nuScenes](https://www.nuscenes.org/)

```bash
cd diffuers
# Extract 12Hz images from the nuScenes dataset
python examples/world_model/data_process/nusc_video.py --output_dir /path/to/extracted_videos
```