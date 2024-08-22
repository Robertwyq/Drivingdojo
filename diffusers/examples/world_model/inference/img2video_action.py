import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)
from diffusers.utils import load_image, export_to_video
import imageio
from src.pipeline.pipeline_stable_video_diffusion_custom import StableVideoDiffusionPipelineV2
from src.model.unet_spatial_temporal_condition_custom import UNetSpatioTemporalConditionModelV2
from src.encoder.unified_encoder import EgoCondEncoder


def load_and_transform_frames_ego(num_frames=30):
    clip = []
    for i in range(num_frames):
        # dz (front) , dx (right) 
        action = torch.tensor([0.5,0.0]) # straight
        # action = torch.tensor([0.0,0.0]) # stop
        # action = torch.tensor([-0.2,0.0]) # back
        action = action.view(action.size(0),1)
        clip.append(action)
        
    return clip

model_path = 'demo_model/img2video_576_30f_action'

img_path = 'demo_img/0010_CameraFpgaP0H120.jpg'
output_folder = './output'
os.makedirs(output_folder,exist_ok=True)

### setting
length = 30
w = 576
h = 320
### setting

pipe = StableVideoDiffusionPipelineV2.from_pretrained(
    model_path,  torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

image = load_image(img_path)

ego_clip = load_and_transform_frames_ego(length)
frames_ego = torch.cat(ego_clip,1)
frames_ego = frames_ego.transpose(0,1)
frames_ego = frames_ego / (0.1)

generator = torch.manual_seed(42)

frames = pipe(image, ego = frames_ego, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0.01, fps=5, generator=generator).frames[0]

export_path = os.path.join(output_folder, 'test_action.gif')

# export to gif
imageio.mimsave(export_path, frames, format='GIF', duration=200, loop=0)

