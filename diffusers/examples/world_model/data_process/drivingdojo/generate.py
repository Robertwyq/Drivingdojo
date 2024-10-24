import os
import pickle
import argparse
from tqdm import tqdm

# 设置命令行参数
parser = argparse.ArgumentParser(description='Filter videos by minimum frames.')
parser.add_argument('--path', type=str, required=True, help='Path to the directory containing videos.')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the output pickle file.')
parser.add_argument('--min_frames', type=int, default=30, help='Minimum number of frames required for a video to be included.')
args = parser.parse_args()

# 获取参数
path = args.path
save_path = args.save_path
files = sorted(os.listdir(path))

result_file = {}
count = 0

# 遍历视频文件夹
for file in tqdm(files):
    imgs = sorted(os.listdir(os.path.join(path, file)))
    imgs = [os.path.join(path, file, img) for img in imgs]
    
    # 仅保存帧数大于等于min_frames的视频
    if len(imgs) >= args.min_frames:
        item = {}
        item['CAM_FRONT'] = imgs
        result_file[file] = item
        count += len(imgs)

print(count)
print(len(result_file))

# 保存结果
with open(save_path, 'wb') as f:
    pickle.dump(result_file, f)

print('finished')
