from diffusers import (
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    )
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model_path = "onkarsus13/controlnet_stablediffusion_scenetextEraser"

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained(model_path)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device).manual_seed(1)

image = Image.open("/DATA/ocr_team_2/onkar2/test/all_images/223.jpg").resize((512, 512))
mask_image = Image.open('/DATA/ocr_team_2/onkar2/test/all_mask/223.png').resize((512, 512))

image = pipe(
    image,
    mask_image,
    [mask_image],
    num_inference_steps=20,
    generator=generator,
    controlnet_conditioning_scale=1.0,
    guidance_scale=1.0
).images[0]

image.save('test1.png')


