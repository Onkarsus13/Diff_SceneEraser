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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def poly_to_mask(poly):
    filee = open(poly, 'r')
    mask = np.zeros((512, 512))
    lines = filee.readlines()
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(',')
        line = [int(i) for i in line]

        polygon = line
        width = 512
        height = 512

        img = Image.fromarray(np.zeros((512, 512), dtype='uint8'))
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += np.array(img)
    mask = np.expand_dims((mask > 0).astype('uint8'), axis=2)

    return Image.fromarray(np.concatenate((mask, mask, mask), axis=2)*255)


pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained('controlnet_scenetext_eraser/')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to(torch.device('cuda:1'))

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda:1").manual_seed(1)

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


