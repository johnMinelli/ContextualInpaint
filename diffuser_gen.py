import os
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, DDIMScheduler

from dataset import ProcDataset, proc_collate_fn
from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Eval_args

args = Eval_args().parse_args()

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
    np.random.seed(args.seed)

# Dataset
batch_size = 8
categories = {"food":0.07, "drink":0.07, "phone":0.52, "cigarette":0.10, "default":0.24}
procedural_dataset = ProcDataset(args, categories, len(controlnet) if controlnet is not None else 0)
procedural_dataloader = torch.utils.data.DataLoader(procedural_dataset, shuffle=True, batch_size=batch_size, num_workers=0, collate_fn=proc_collate_fn)

for gen_id, batch in enumerate(procedural_dataloader):
    gm,gs,c1,c2 = True, np.random.choice([7,8]), 0.90, 0.95
    with torch.no_grad():
        with torch.autocast(f"cuda"):
            pred_images = pipeline(prompt=batch["prompt"], controlnet_prompt=batch["control_prompt"], negative_prompt=batch["neg_prompt"], focus_prompt=batch["focus_prompt"],
                                  image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["mask_conditioning"], height=512, width=512,
                                  strength=1.0, controlnet_conditioning_scale=[c1, c2], num_inference_steps=50, guidance_scale=gs, guess_mode=gm, generator=generator).images

    for i, pred_image in enumerate(pred_images):
        log = {"source": batch["image"][i], "category": batch["category"][i], "predictions": pred_image, "source_filename":os.path.basename(batch["image_name"][i]), "prompt": batch["prompt"][i], "control": batch["control_prompt"][i][-1] if type(controlnet)==list else batch["control_prompt"][i]}
    
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
            # save locally
            out_path = os.path.join("./out", "train_data_gen")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
                with open(os.path.join(out_path, "train.csv"), 'w') as file:
                    file.write(csv_header+'\n')

            # for hp_id in range(len(log["predictions"])):  # hp fixed
            filename = os.path.splitext(log["source_filename"])[0]+"_"+str((gen_id*batch_size)+i)+os.path.splitext(log["source_filename"])[1]
            log["predictions"].save(os.path.join(out_path, filename))

            csv_line = ",".join([os.path.splitext(filename)[0], "train", cat2classficationhead(log["category"])])

            with open(os.path.join(out_path, "train.csv"), 'a') as file:
                file.write(csv_line+'\n')

                # show images here
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
