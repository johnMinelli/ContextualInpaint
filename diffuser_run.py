import codecs
import json

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection

import wandb
from accelerate.logging import get_logger
from diffusers import ControlNetModel, DDIMScheduler
import numpy as np
import torch

from PIL import Image

from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Eval_args
from utils.utils import replicate

logger = get_logger(__name__)

args = Eval_args().parse_args()
CONTROLNET_OBJ_MASKING = False


# load control net and stable diffusion v1-5
controlnet = None
if len(args.controlnet_model_name_or_path)>1:
    assert CONTROLNET_OBJ_MASKING, "`CONTROLNET_OBJ_MASKING` is False but more than 1 model is provided. This is not an intended use."
    controlnet = [ControlNetModel.from_pretrained(net_path, torch_dtype=torch.float32) for net_path in args.controlnet_model_name_or_path] if args.controlnet_model_name_or_path is not None else None
elif len(args.controlnet_model_name_or_path)==1:
    logger.warn(f"`CONTROLNET_OBJ_MASKING` is {CONTROLNET_OBJ_MASKING}. Make sure you are running the intended controlnet model.")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path[0], torch_dtype=torch.float32)
controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(torch.device("cuda"))

pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet, 
        controlnet_text_encoder=controlnet_text_encoder, 
        safety_checker=None,
        controlnet_prompt_seq_projection=CONTROLNET_OBJ_MASKING,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# optimization
if args.enable_cpu_offload:
    pipeline.enable_model_cpu_offload()
if args.enable_xformers_memory_efficient_attention:
    pipeline.enable_xformers_memory_efficient_attention()

# randomization
if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

validation_data = json.load(codecs.open(args.evaluation_file, 'r', 'utf-8-sig'))

# Access the values as a dictionary
images = validation_data['images']
n = len(images)
conditioning = replicate(validation_data.get('conditioning', []), n)
masks = replicate(validation_data.get('masks', []), n)
prompts = replicate(validation_data.get('prompts', []), n)
neg_prompts = replicate(validation_data.get('neg_prompts', []), n)
control_prompts = replicate(validation_data.get('control_prompts', []), n)
focus_prompts = replicate(validation_data.get('focus_prompts', []), n)


image_logs = []
for prompt, neg_prompt, image, mask, conditioning, control_prompt, focus_prompt in zip(prompts, neg_prompts, images, masks, conditioning, control_prompts, focus_prompts):
    image = Image.open(image).convert("RGB") if image is not None else None
    mask_conditioning = Image.open(mask).convert("RGB") if mask is not None else None
    mask = Image.open(mask).convert("L") if mask is not None else None
    conditioning = Image.open(conditioning).convert("RGB") if conditioning is not None else None
    focus_prompt = [focus_prompt] if focus_prompt else None

    images = []
    for _ in range(args.num_validation_images):
        with torch.no_grad():
            with torch.autocast(f"cuda"):
                pred_image = pipeline(prompt=prompt, controlnet_prompt=control_prompt, negative_prompt=neg_prompt, focus_prompt=focus_prompt,
                                      image=image, mask_image=mask, conditioning_image=[mask_conditioning]*len(controlnet) if type(controlnet)==list else mask_conditioning, height=512, width=512,
                                      strength=1.0, controlnet_conditioning_scale=1., num_inference_steps=50, guidance_scale=9, guess_mode=True, generator=generator).images[0]
            images.append(pred_image)

    image_logs.append({"reference": image, "images": images, "prompt": prompt})

if args.log == "tensorboard":
    writer = SummaryWriter()
    for log in image_logs:
        images = log["images"]
        prompt = log["prompt"]
        image = log["reference"]
        formatted_images = [np.asarray(image)]

        for image in images:
            formatted_images.append(np.asarray(image))

        formatted_images = np.stack(formatted_images)

        writer.add_images(prompt, formatted_images, 0, dataformats="NHWC")
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
    # show images here
    for log in image_logs:
        for image in log["images"]:
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(image)
            plt.show()
