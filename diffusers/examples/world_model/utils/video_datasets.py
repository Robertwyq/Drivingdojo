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
            img = img.resize(img_size)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

class VideoCondDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        conditions = None,
        interval=1,
        img_size=(576,320),
        **kargs
        ):

        super().__init__()
        self.loader = default_loader
        self.init_caption = init_caption 
        self.multi_view = multi_view
        self.video_length = video_length
        self.img_transform = video_transforms
        self.conditions = conditions
        self.interval = interval
        self.img_size = img_size
        
        self.videos, self.conds = self._make_dataset(data_root, video_length, conditions = conditions)

        self.default_transforms = tr.Compose(
            [tr.ToTensor(),]
        )
        self.cond_transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        

    def __getitem__(self, index):

        video = self.videos[index]
        init_frame = random.randint(0,len(video)-self.video_length*self.interval)
        video = video[init_frame:init_frame+self.video_length*self.interval:self.interval] # video begin with a random frame
        
        if 'box' in self.conditions:
            conds_box = self.conds['box'][index]
            conds_box = conds_box[init_frame:init_frame+self.video_length*self.interval:self.interval]
        if 'text' in self.conditions:
            conds_text = self.conds['text'][index]
            conds_text = conds_text[init_frame:init_frame+self.video_length*self.interval:self.interval]
        if 'ego' in self.conditions:
            conds_ego = self.conds['ego'][index]
            conds_ego = conds_ego[init_frame:init_frame+self.video_length*self.interval:self.interval]

        assert(len(video) == self.video_length)

        # make clip tensor
        if self.multi_view:
            raise NotImplementedError
        else:
            frames = self.load_and_transform_frames(video, self.loader, self.img_transform)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w

            example = dict()

            if 'box' in self.conditions:
                frames_box = self.load_and_transform_frames_box(conds_box, self.loader, self.cond_transforms)
                frames_box = torch.cat(frames_box,1)
                frames_box = frames_box.transpose(0,1)
                example["box_values"] = frames_box
            if 'ego' in self.conditions:
                frames_ego = self.load_and_transform_frames_ego(conds_ego)
                frames_ego = torch.cat(frames_ego,1)
                frames_ego = frames_ego.transpose(0,1)
                example["ego_values"] = frames_ego / (0.1) 
    
            example["pixel_values"] = frames
            example['images']=self.load_and_transform_frames(video, self.loader)
        
        return example

    def __len__(self):
        return len(self.videos)
    
    def _make_dataset(self, info_path, nframes, conditions):
        with open(info_path, 'rb') as f:
            video_info = pkl.load(f)
        
        output_videos = []
        output_text_videos = []
        output_box_videos = []
        output_ego_videos = []
        
        if not self.multi_view:
            for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Meituan dataset"):
                video = []
                text_video = []
                box_video = []
                ego_video = []
                for frame in frames:
                    video.append(frame['img'])
                    if 'text' in conditions:
                        text_video.append(frame['description'])
                    if 'ego' in conditions:
                        ego_video.append(frame['ego'])
                    if 'box' in conditions:
                        box_video.append(frame['box'])
        
                output_videos.append(video)
                output_text_videos.append(text_video)
                output_ego_videos.append(ego_video)
                output_box_videos.append(box_video)
        else:
            for video_name, frames in video_info.items():
                output_videos.append(frames)
        
        output_cond_videos = {}
        if 'box' in conditions:
            output_cond_videos['box'] = output_box_videos
        if 'text' in conditions:
            output_cond_videos['text'] = output_text_videos
        if 'ego' in conditions:
            output_cond_videos['ego'] = output_ego_videos

        return output_videos, output_cond_videos

    def load_and_transform_frames_ego(self, ego_list):
        clip = []
        for ego in ego_list:
            ego_new = ego[1:-1]
            dxyz = ego_new.split()
            action = torch.tensor([float(dxyz[2]),float(dxyz[0])]) # dz (front) , dx (right)
            action = action.view(action.size(0),1)
            clip.append(action)
        
        return clip


    def load_and_transform_frames(self, frame_list, loader, img_transform=None):
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
            img = img.resize(self.img_size)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

    def load_and_transform_frames_map(self, frame_list, loader, img_transform=None):
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
            if img_transform is not None:
                img = img_transform(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

    def load_and_transform_frames_box(self, frame_list, loader, img_transform=None):
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        self.img_w = 384

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            crop_img = img.crop([0,100,img.size[0],img.size[1]])
            crop_img = crop_img.resize((self.img_w,self.img_w//2))
            if img_transform is not None:
                img = img_transform(crop_img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip
        assert(isinstance(frame_list, list)), "frame_list must be a list not {}".format(type(frame_list))
        clip = []

        self.img_w = 384

        for frame in frame_list:
            if isinstance(frame, tuple):
                fpath, label = frame
            elif isinstance(frame, dict):
                fpath = frame["img_path"]
            else:
                fpath = frame
            
            img = loader(fpath)
            crop_img = img.crop([0,100,img.size[0],img.size[1]])
            crop_img = crop_img.resize((self.img_w,self.img_w//2))
            if img_transform is not None:
                img = img_transform(crop_img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip

class VideoCondNuscenesDataset(data.Dataset):
    def __init__(self,
        data_root,
        video_length,
        video_transforms=None,
        text_tokenizer=None,
        init_caption='',
        multi_view=False,
        ego = False,
        img_size=(720,480),
        code_path=None,
        interval=1,
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
        self.code_path = code_path
        self.ego = ego
        self.interval = interval

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
    
        # Adjust init_frame to ensure there are enough frames for the specified interval
        max_init_frame = len(video) - self.video_length * self.interval
        if max_init_frame < 0:
            raise ValueError("Video is too short for the specified length and interval.")
        
        init_frame = random.randint(0, max_init_frame)

        
        # Sample frames according to the interval
        video = video[init_frame:init_frame + self.video_length * self.interval:self.interval]
        conds = conds[init_frame:init_frame + self.video_length * self.interval:self.interval]
        
        assert len(video) == self.video_length, "Incorrect number of frames after applying interval."
        # make clip tensor
        if self.multi_view:
            print('not support yet')
        else:
            frames = self.load_and_transform_frames(video, self.loader, self.img_transform, img_size=self.image_size)

            frames = torch.cat(frames, 1) # c,t,h,w
            frames = frames.transpose(0, 1) # t,c,h,w
            
            # token condition
            frames_token = self.load_and_transform_tokens(conds)
            frames_token = [i.unsqueeze(0) for i in frames_token]
            # frames_token = torch.cat(frames_token, 1).squeeze(0) # t,hw
            frames_token = torch.cat(frames_token, 0) # t,hw
            
            example = dict()
            example["pixel_values"] = frames
            example["pixel_token"] = frames_token
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
        output_tokens = []
        for video_name, frames in tqdm.tqdm(video_info.items(), desc="Making Nuscenes dataset"):
            for cam_v in self.camera_view:
                view_video = []
                token_video = []
                for frame in frames[cam_v]:
                    view_video.append(frame)
                    # frame_id = frame.split('/')[-1].split('.')[0][-3:]
                    # token_path = os.path.join(self.code_path, video_name, f"{frame_id}.npy")
                    scene_id = frame.split('/')[-2]
                    token_path = os.path.join(self.code_path, scene_id, f"{frame.split('/')[-1].split('.')[0]}_512288_ds16.npy")
                    
                    token_video.append(token_path)
                
                output_videos.append(view_video)
                output_tokens.append(token_video)

        return output_videos, output_tokens
    
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
            img = img.resize(img_size)
            if img_transform is not None:
                img = img_transform(img)
            else:
                img = self.default_transforms(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            clip.append(img)
        return clip
    
    def load_and_transform_tokens(self, token_list):
        assert(isinstance(token_list, list)), "token_list must be a list not {}".format(type(token_list))
        clip = []

        for token in token_list:
            if isinstance(token, tuple):
                fpath, label = token
            elif isinstance(token, dict):
                fpath = token["img_path"]
            else:
                fpath = token
            
            token = torch.from_numpy(np.load(fpath))
            clip.append(token)
        return clip
