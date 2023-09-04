This is the trained model for the controlnet-stablediffusion for the scene text eraser. We have to customized the pipeline for the controlnet-stablediffusion-inpaint


To training the model we had to use the SCUT-Ensnet dataset

Installation

```
cd Diff_SceneTextEraser
pip install -e ".[torch]"
```

You can get the changes in the official repositoy

Inference


```
python test_eraser.py
```

Check the Inference code

```python
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "onkarsus13/controlnet_stablediffusion_scenetextEraser"

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained(model_path)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device).manual_seed(1)

image = Image.open("<path to scene text image>").resize((512, 512))
mask_image = Image.open('<path to the corrospoinding mask image>').resize((512, 512))

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

```

You will find the models checkpoints [here](https://huggingface.co/onkarsus13/controlnet_stablediffusion_scenetextEraser/tree/main)
