import codecs
import json
import os

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection

import wandb
from diffusers import ControlNetModel, DDIMScheduler
import numpy as np
import torch

from PIL import Image

from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Eval_args
from utils.utils import not_image

args = Eval_args().parse_args()
CONTROLNET_OBJ_MASKING = True


controlnet = None
if len(args.controlnet_model_name_or_path)>1:
    assert CONTROLNET_OBJ_MASKING, "`CONTROLNET_OBJ_MASKING` is False but more than 1 model is provided. This is not an intended use."
    controlnet = [ControlNetModel.from_pretrained(net_path, torch_dtype=torch.float32) for net_path in args.controlnet_model_name_or_path] if args.controlnet_model_name_or_path is not None else None
elif len(args.controlnet_model_name_or_path)==1:
    print(f"WARN: `CONTROLNET_OBJ_MASKING` is {CONTROLNET_OBJ_MASKING}. Make sure you are running the intended controlnet model.")
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

# Access data json ("if is in the json it will be used")
generation_template = json.load(codecs.open(args.evaluation_file, 'r', 'utf-8-sig'))
images_rgb = [os.path.join(generation_template['images_path_rgb'], f) for f in sorted(os.listdir(generation_template['images_path_rgb'])) if not not_image(f)]
images_mask = [os.path.join(generation_template['images_path_mask'], f) for f in sorted(os.listdir(generation_template['images_path_mask'])) if not not_image(f)]
images_cond = [os.path.join(generation_template['images_path_cond'], f) for f in sorted(os.listdir(generation_template['images_path_cond'])) if not not_image(f)]
texts = generation_template['cond_prompts']
assert len(images_rgb) == len(images_mask) == len(images_cond), f"The number of images in specified paths must match: RGB {images_rgb}, MASK {len(images_mask)}, COND {len(images_cond)}."

image_logs = []
for id in range(args.num_validation_images):
    image_id = np.random.randint(len(images_rgb))
    text_id = np.random.randint(len(texts))

    prompt = texts[text_id].get("prompt", None)
    neg_prompt = texts[text_id].get("neg_prompt", generation_template.get("global_neg_prompt", ""))
    control_prompt = [prompt, texts[text_id]["control"]] if type(controlnet)==list else texts[text_id].get("control", prompt) if controlnet is not None else None
    focus_prompt = texts[text_id].get("focus", "" if type(controlnet)==list else None)

    image = Image.open(images_rgb[image_id]).convert("RGB")
    mask_conditioning = [Image.open(images_cond[image_id]).convert("RGB")]*len(controlnet) if type(controlnet)==list else Image.open(images_cond[image_id]).convert("RGB")
    mask = Image.open(images_mask[image_id]).convert("L")

    images = []
    for h, hyperparams in enumerate([(gm, gs, round(c1, 3), round(c2, 3)) for gm in [True] for gs in range(7,8) for c1 in [0.90] for c2 in [0.95]]):  # True,7-8,.85-.9,.9-.95
        gm,gs,c1,c2 = hyperparams
        with torch.autocast(f"cuda"):
            pred_image = pipeline(prompt=prompt, controlnet_prompt=control_prompt, negative_prompt=neg_prompt, focus_prompt=focus_prompt,
                                  image=image, mask_image=mask, conditioning_image=mask_conditioning, height=512, width=512,
                                  strength=1.0, controlnet_conditioning_scale=[c1, c2], num_inference_steps=50, guidance_scale=gs, guess_mode=gm, generator=generator).images[0]
        # for n, im in enumerate(pred_image):
        #     images.append({"id": f"{i}_{h}_{n}", "image":im, "desc": f"gm:{gm}, gs:{gs}, c1:{c1}, c2:{c2}"})
    log = {"source": image, "category": texts[text_id]["category"], "prediction": pred_image, "source_filename": os.path.basename(images_mask[image_id]), "prompt": prompt, "control": control_prompt[-1] if type(controlnet)==list else control_prompt}

    if args.log == "tensorboard":
        writer = SummaryWriter()
        images = log["predictions"]
        prompt = log["prompt"]
        image = log["source"]
        formatted_images = [np.asarray(image)]

        for image in images:
            formatted_images.append(np.asarray(image))

        formatted_images = np.stack(formatted_images)

        writer.add_images(prompt, formatted_images, 0, dataformats="NHWC")
    elif args.log == "wandb":
        run = wandb.init(entity="johnminelli", project="train_controlnet", resume=args.log_run_id)
        formatted_images = []
        images = log["predictions"]
        prompt = log["prompt"]
        image = log["source"]

        formatted_images.append(wandb.Image(image, caption="Controlnet conditioning"))

        for image in images:
            image = wandb.Image(image, caption=prompt)
            formatted_images.append(image)
    
        run.log({"evaluation": formatted_images})
        wandb.finish()
    else:
        # show images here
        out_path = os.path.join("./out", "gen_data")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            with open(os.path.join(out_path, "data.csv"), 'w') as file:
                file.write('label_id,split,phone_binary,cigarette_binary,food_binary\n')

        filename = os.path.splitext(log["source_filename"])[0]+"_"+str(id)+os.path.splitext(log["source_filename"])[1]
        log["prediction"].save(os.path.join(out_path, filename))
        csv_line = ",".join([os.path.splitext(filename)[0], "train", str(int(log["category"] == "phone")), str(int(log["category"] == "cigarette")), str(int(log["category"] == "food"))])
        with open(os.path.join(out_path, "data.csv"), 'a') as file:
            file.write(csv_line+'\n')

            # fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
            # for ax in axs:
            #     ax.axis('off')
            #     ax.margins(0, 0)
            #     ax.xaxis.set_major_locator(plt.NullLocator())
            #     ax.yaxis.set_major_locator(plt.NullLocator())
            # 
            # # Plot source image and predicted image
            # axs[0].imshow(log["source"])
            # axs[1].imshow(image["image"])
            # 
            # prompt = '\n'.join([log["prompt"][i:i + 75] for i in range(0, len(log["prompt"]), 75)])
            # plt.suptitle('"'+prompt+'"')
            # axs[0].set_title(log["source_filename"])
            # axs[1].set_title(image["desc"])
            # 
            # # Add a common title on top
            # plt.savefig(os.path.join(out_path, str(image["id"])+"_"+log["source_filename"]))
