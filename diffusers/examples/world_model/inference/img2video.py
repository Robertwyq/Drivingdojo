import torch
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image
import imageio

# Setting
length = 14
w = 1024
h = 576

# model path
model_path = 'demo_model/img2video_1024_14f'

# initial image path
img_path = 'demo_img/0010_CameraFpgaP0H120.jpg'

output_folder = './output'
os.makedirs(output_folder,exist_ok=True)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

image = load_image(img_path)

generator = torch.manual_seed(42)
frames = pipe(image, width=w, height=h,num_frames=length, num_inference_steps=25, noise_aug_strength=0.01, fps = 5, generator=generator).frames[0]

export_path = os.path.join(output_folder, 'test.gif')

# export to gif
imageio.mimsave(export_path, frames, format='GIF', duration=200, loop=0)
