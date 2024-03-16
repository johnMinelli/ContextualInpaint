import codecs
import json
import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import diffusers
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.pipelines.controlnet import MultiControlNetModel
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, DDIMScheduler

from dataset import ProcDataset
from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from third.verizon.ludovicodl.nn.models.object_in_hand import get_model
from third.verizon.ludovicodl.training.learning import MultiTask
from utils.parser import Optim_args
from utils.data_utils import proc_collate_fn, not_image
from vsource.evaluation.tasks import PhoneUsageTask, CigaretteTask, FoodTask

logger = get_logger(__name__)
args = Optim_args().parse_args()

csv_header = "label_id,split,phone_binary,cigarette_binary,food_binary"
cat2classficationhead = lambda cat: "1,0,0" if cat in ["phone"] else \
                                    "0,1,0" if cat in ["cigarette"] else \
                                    "0,0,1" if cat in ["food", "drink"] else \
                                    "0,0,0"
# enable pipe behaviour with text encoder projection required by the obj_masking controlnet 
CONTROLNET_OBJ_MASKING = True

# init ControlNet(s)
controlnet = None
if len(args.controlnet_model_name_or_path)>1:
    assert CONTROLNET_OBJ_MASKING, "`CONTROLNET_OBJ_MASKING` is False but more than 1 model is provided. This is not an intended use."
    controlnet = [ControlNetModel.from_pretrained(net_path, torch_dtype=torch.float32) for net_path in args.controlnet_model_name_or_path] if args.controlnet_model_name_or_path is not None else None
elif len(args.controlnet_model_name_or_path)==1:
    print(f"WARN: `CONTROLNET_OBJ_MASKING` is {CONTROLNET_OBJ_MASKING}. Make sure you are running the intended controlnet model.")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path[0], torch_dtype=torch.float32)
controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(torch.device("cuda"))

# init pipeline
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

tasks = [PhoneUsageTask(), CigaretteTask(), FoodTask()]
cg_task = MultiTask(tasks=tasks, name='multi-task-dfc-distraction')
classifier = get_model(backbone="efficientnet_v2_s", tasks=tasks)

# Load accelerator and setup for optimized execution
logging_dir = Path("./out/oih-optim", "logs")
accelerator_project_config = ProjectConfiguration(project_dir="./out/oih-optim", logging_dir=logging_dir)
accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=None, log_with=None, project_config=accelerator_project_config, )

# Make one log on every process with the configuration for debugging.
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger.info(accelerator.state, main_process_only=False)
transformers.utils.logging.set_verbosity_warning()
diffusers.utils.logging.set_verbosity_info()

# optimization
if args.enable_cpu_offload:
    pipeline.enable_model_cpu_offload()
if args.enable_xformers_memory_efficient_attention:
    pipeline.enable_xformers_memory_efficient_attention()

pipeline.vae.requires_grad_(False)
pipeline.unet.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.controlnet_text_encoder.requires_grad_(False)
pipeline.controlnet.requires_grad_(False)
classifier.requires_grad_(False)
classifier.eval()
classifier.load_state_dict((lambda f,l: {k:v  for k,v in zip(f.keys(), l.values())})(classifier.state_dict(), torch.load("./out/best_model_31_val_score=2.6616.pt", map_location=accelerator.device)))


if args.gradient_checkpointing:
    if isinstance(pipeline.controlnet, MultiControlNetModel):
        [n.enable_gradient_checkpointing() for n in pipeline.controlnet.nets]
    elif pipeline.controlnet is not None:
        pipeline.controlnet.enable_gradient_checkpointing()

# randomization
if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    np.random.seed(args.seed)

# Dataset
task_to_class = {'phone_binary': ["phone"], 'cigarette_binary': ["cigarette"], 'food_binary': ["food", "drink"]}
class_to_task = {'phone':'phone_binary', 'cigarette':'cigarette_binary', 'food':'food_binary', 'drink':'food_binary'}
categories = {"food":0.31, "drink":0.07, "phone":0.52, "cigarette":0.10, "default":0.0}
procedural_dataset = ProcDataset(args.evaluation_file, args.num_validation_images, categories, len(controlnet) if controlnet is not None else 0)
procedural_dataloader = torch.utils.data.DataLoader(procedural_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.dataloader_num_workers, collate_fn=proc_collate_fn)

# Prepare everything with accelerator library
pipeline.unet, pipeline.controlnet, pipeline.vae, classifier, procedural_dataloader = accelerator.prepare(pipeline.unet, pipeline.controlnet, pipeline.vae, classifier, procedural_dataloader)

# text_encoder to device and cast to weight_dtype
pipeline.text_encoder.to(accelerator.device, dtype=torch.float32)
pipeline.controlnet_text_encoder.to(accelerator.device, dtype=torch.float32)

if args.log == "wandb":
    wandb.init(entity="johnminelli", project="train_controlnet")

# cycle taking batches of procedurally generated prompts 
for gen_id, batch in enumerate(procedural_dataloader):
    or_prompt_embeds = pipeline.encode_prompt(batch["prompt"], accelerator.device, False, pipeline.text_encoder, num_images_per_prompt=1, return_tuple=False)
    prompt_embeds = or_prompt_embeds.clone()
    prompt_embeds.requires_grad = True
    optimizer = torch.optim.AdamW([prompt_embeds], lr=0.001, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )
    num_optim_steps = 20
    num_inference_steps = 20
    cg_scale = 10

    # optimize generation for given prompts
    print(f"Generation {gen_id}, {str(batch['category'][0])}")
    for i in range(num_optim_steps):
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        # generate the image from text
        pred_images = pipeline(prompt_embeds=prompt_embeds, controlnet_prompt=batch["control_prompt"], negative_prompt=batch["neg_prompt"], focus_prompt=batch["focus_prompt"],
                              image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["mask_conditioning"], height=512, width=512,
                              strength=1.0, controlnet_conditioning_scale=[0.90, 0.95], num_inference_steps=num_inference_steps, guidance_scale=8, guess_mode=True, generator=generator, output_type="pt", gradient_checkpointing=args.gradient_checkpointing).images
        # classify the output image
        out = cg_task.forward_pass(x=pred_images, y={k:torch.tensor([1 if cat in v else 0 for cat in batch["category"]]).to(accelerator.device) for k,v in task_to_class.items()}, model=classifier, device=accelerator.device)
        scores = torch.cat([torch.sigmoid(out[class_to_task[cat]]["out"]) for cat in batch["category"]])
        if i==0 and any(scores<0.15): break  # early exit
        # get optimization direction
        log_probs = torch.log_softmax(torch.cat([1 - scores, scores], -1), dim=-1)[:, (scores[:,0].detach()<=0.5).int()].mean() * cg_scale
        # logging
        print("SCORES:", scores.detach().cpu().numpy(), " - DIFF:", (or_prompt_embeds-prompt_embeds).sum().detach().cpu().numpy())
        wandb.log({f"Generation/Gen_{gen_id}_image": wandb.Image(pred_images[0].permute(1, 2, 0).detach().cpu().numpy(), caption=str(batch["category"][0])), f"Generation/Gen_{gen_id}_score":scores[0][0].detach().cpu().numpy(), f"Generation/Gen_{gen_id}_delta":(or_prompt_embeds-prompt_embeds).sum().detach().cpu().numpy()})
        plt.imshow(pred_images[0].permute(1, 2, 0).detach().cpu()); plt.title(str(batch["category"][0])); plt.show()
        # optimize
        optimizer.zero_grad()
        log_probs.backward()
        optimizer.step()

    # log&save final result
    for i, pred_image in enumerate(pred_images):
        log = {"source": batch["image"][i], "category": batch["category"][i], "predictions": pred_image, "source_filename":os.path.basename(batch["image_name"][i]), "prompt": batch["prompt"][i], "control": batch["control_prompt"][i][-1] if type(controlnet)==list else batch["control_prompt"][i]}
        # do some logging here
