import os
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a pickle file of video paths.')
    parser.add_argument('--video_path', type=str, required=True, 
                        help='Directory containing video data.')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Output path for the pickle file.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    scenes = sorted(os.listdir(args.video_path))
    video_dict = {}

    for scene in scenes:
        scene_path = os.path.join(args.video_path, scene)
        sensors = sorted(os.listdir(scene_path))
        scene_dict = {}
        
        for sensor in sensors:
            sensor_path = os.path.join(scene_path, sensor)
            imgs = sorted(os.listdir(sensor_path))
            imgs = [os.path.join(sensor_path, img) for img in imgs]
            scene_dict[sensor] = imgs
        
        video_dict[scene] = scene_dict

    with open(args.output_path, 'wb') as f:
        pickle.dump(video_dict, f)

if __name__ == '__main__':
    main()
