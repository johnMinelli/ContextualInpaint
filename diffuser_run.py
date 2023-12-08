import codecs
import json
import types

import tensorboard
from diffusers.models.attention_processor import Attention

import wandb
from accelerate.logging import get_logger
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,
                       StableDiffusionInpaintPipeline, DDIMScheduler)
import numpy as np
import torch

import cv2
from PIL import Image

from attention_forward import new_forward
from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Eval_args

logger = get_logger(__name__)

args = Eval_args().parse_args()


# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=torch.float32) if args.controlnet_model_name_or_path is not None else None
pipe = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet, 
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Apply custom attention
for module in pipe.unet.modules():
    if isinstance(module, Attention):
        # placeholder function for the original forward
        module.ori_forward = module.forward
        module.forward = types.MethodType(new_forward, module)
        module.cfg = {"t_align": 200}

# optimization
if args.enable_cpu_offload:
    pipe.enable_model_cpu_offload()
if args.enable_xformers_memory_efficient_attention:
    pipe.enable_xformers_memory_efficient_attention()

# randomization
if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=args.device).manual_seed(args.seed)


validation_data = json.load(codecs.open(args.evaluation_file, 'r', 'utf-8-sig'))
validation_prompts = validation_data['validation_prompts']

def raiser(): raise ValueError("number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`")
replicate = lambda x: x if len(x) == len(validation_prompts) else x * len(validation_prompts) if len(x) == 1 else [None] * len(validation_prompts) if len(x) == 0 else raiser()
validation_images = replicate(validation_data['validation_images'])
validation_masks = replicate(validation_data['validation_masks'])
validation_controls = replicate(validation_data['validation_controls'])
validation_neg_prompts = replicate(validation_data['validation_neg_prompts'])

image_logs = []
for prompt, neg_prompt, image, mask, control in zip(validation_prompts, validation_neg_prompts, validation_images, validation_masks, validation_controls):
    image = Image.open(image).convert("RGB") if image is not None else None
    mask = Image.open(mask).convert("L") if mask is not None else None
    mask_control = Image.open(mask).convert("RGB") if mask is not None else None
    control = Image.open(control).convert("RGB") if control is not None else None
    neg_prompt = neg_prompt if type(neg_prompt) == list else [neg_prompt]*len(prompt)

    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast(f"cuda"):
            pred_image = pipe(prompt=prompt, begative_prompt=neg_prompt, height=args.resolution, width=args.resolution,
                              strength=1.0, controlnet_conditioning_scale=0.8, guidance_scale=7.5, guess_mode=False, num_inference_steps=args.steps, generator=generator).images[0]
        images.append(pred_image)

    image_logs.append({"reference": image, "images": images, "prompt": prompt})

if args.log == "tensorboard":
    for log in image_logs:
        images = log["images"]
        prompt = log["prompt"]
        image = log["reference"]
        formatted_images = [np.asarray(image)]

        for image in images:
            formatted_images.append(np.asarray(image))

        formatted_images = np.stack(formatted_images)

        tensorboard.writer.add_images(prompt, formatted_images, 0, dataformats="NHWC")
elif args.log == "wandb":
    run = wandb.init(entity="johnminelli", project="train_controlnet", resume=args.log_run_id)
    formatted_images = []
    for log in image_logs:
        images = log["images"]
        prompt = log["prompt"]
        image = log["reference"]

        formatted_images.append(wandb.Image(image, caption="Controlnet conditioning"))

        for image in images:
            image = wandb.Image(image, caption=prompt)
            formatted_images.append(image)

    run.log({"evaluation": formatted_images})
    wandb.finish()
else:
    logger.warning(f"image logging not implemented for {args.log}")