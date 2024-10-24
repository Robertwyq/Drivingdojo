import os
import json
import pickle
from tqdm import tqdm
import argparse

def main(json_path, root, output_pkl, min_frames):
    # Load the JSON metadata
    with open(json_path, 'r') as f:
        data = json.load(f)

    dojo_action = {}

    # Process each entry in the metadata
    for key in tqdm(data.keys()):
        data_info = data[key]
        image_files = data_info['videos']
        ego_files = data_info['action_info']
        
        frame_list = []

        # Iterate through image and ego files, excluding the last image
        for image, ego in zip(image_files[:-1], ego_files):
            item = {}
            item['img'] = os.path.join(root, image)

            # Read the ego information
            ego_path = os.path.join(root, ego)
            with open(ego_path, 'r') as f:
                lines = f.readlines()
            
            item['ego'] = lines[0] if lines else None  # Handle potential empty lines
            frame_list.append(item)

        # Ensure at least min_frames
        if len(frame_list) >= min_frames:
            dojo_action[key] = frame_list
        else:
            print('Skipping:', key)  # More informative skip message
            
    # Output the results
    print('Total valid entries:', len(dojo_action.keys()))

    # Save the processed data to a pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(dojo_action, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process navigation actions from metadata.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON metadata file.')
    parser.add_argument('--root', type=str, required=True, help='Root directory containing video and ego files.')
    parser.add_argument('--output_pkl', type=str, required=True, help='Path to output the pickle file.')
    parser.add_argument('--min_frames', type=int, default=30, help='Minimum number of frames required for each action.')

    args = parser.parse_args()
    
    main(args.json_path, args.root, args.output_pkl, args.min_frames)
