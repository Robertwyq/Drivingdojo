import os, re
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms as tr
import random, json
import pickle as pkl
import tqdm
from collections import defaultdict
from utils.image_datasets import combine_text_conds, get_text_cond_tokens, TEXT_CONDITIONS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .image_datasets import default_loader
from ipdb import set_trace

class VideoNuscenesDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        ego = False,
        img_size=(720,480),
        **kargs
        ):

        super().__init__()
        self.loader = default_loader
        self.init_caption = init_caption 
        self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms
        self.image_size = img_size
        self.camera_view = ['CAM_FRONT']
        self.camera_captions = {'cam01': 'front camera view',
                                'CAM_BACK': 'back camera view',
                                'CAM_FRONT': 'front camera view',
                                'CAM_FRONT_RIGHT':'front right camera view',
                                'CAM_FRONT_LEFT':'front left camera view',
                                'CAM_BACK_RIGHT':'back right camera view',
                                'CAM_BACK_LEFT':'back left camera view'
                                }
        self.ego = ego

        if self.ego:
            self.videos, self.conds = self._make_dataset_video_ego(data_root, video_length)
        else:
            self.videos, self.conds = self._make_dataset_video(data_root, video_length)

        self.default_transforms = tr.Compose(
            [
                tr.ToTensor(),
            ]
        )
    
    def __getitem__(self, index):
        video, conds = self.videos[index], self.conds[index]
        init_frame = random.randint(0,len(video)-self.video_length)
        video = video[init_frame:init_frame+self.video_length] # video begin with a random frame
        conds = conds[init_frame:init_frame+self.video_length]
        assert(len(video) == self.video_length)

        # make clip tensor
        if self.multi_view:
            print('not support yet')
        else:
            frames = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=self.image_size)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w

            # frames_low = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=(288,160))
            # frames_low = torch.cat(frames_low, 1) # c,t,h,w
            # frames_low = frames_low.transpose(0, 1) # t,c,h,w
            
            example = dict()
            example["pixel_values"] = frames
            # example["pixel_values_low"] = frames_low
            example["images"] = self.load_and_transform_frames(video, self.loader, img_size=self.image_size)
            if self.ego:
                frames_ego = torch.tensor(conds)
                example["ego_values"] = frames_ego / (0.5)
        
        return example

    def __len__(self):
        return len(self.videos)
    
    def _make_dataset_video_ego(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_ego_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            for cam_v in self.camera_view:
                view_video = []
                ego_video = []
                for frame_i in range(len(frames[cam_v])):
                    frame = frames[cam_v][frame_i]
                    frame_path = frame
                    view_video.append(frame_path)
                    ego_video.append(frames['ego'][frame_i])
                
                output_videos.append(view_video)
                output_ego_videos.append(ego_video)

        return output_videos, output_ego_videos

    def _make_dataset_video(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            for cam_v in self.camera_view:
                view_video = []
                text_video = []
                for frame in frames[cam_v]:
                    view_video.append(frame)
                    text_video.append(self.camera_captions[cam_v])
                
                output_videos.append(view_video)
                output_text_videos.append(text_video)

        return output_videos, output_text_videos
    
    def _make_dataset(self, info_path, nframes):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        if not self.multi_view:
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
                for cam_v in self.camera_view:
                    view_video = []
                    text_video = []
                    for frame in frames:
                        view_video.append(frame[cam_v])
                        text_video.append(self.camera_captions[cam_v])
                    
                    output_videos.append(view_video)
                    output_text_videos.append(text_video)
        else:
            for video_name, frames in video_info.items():
                output_videos.append(frames)

        return output_videos, output_text_videos

    def load_and_transform_frames(self, frame_list, loader, img_transform=None, img_size=(576,320)):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            # hard code here!
            # img_crop = img.crop([0,100,img.size[0],img.size[1]])
            # img = img_crop.resize((384,192))
            img = img.resize(img_size)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip
