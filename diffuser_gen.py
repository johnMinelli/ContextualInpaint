import os
import numpy as np
import torch
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_unet_state_dict_to_peft
from lycoris import create_lycoris, LycorisNetwork
from peft import set_peft_model_state_dict, LoraConfig
from safetensors.torch import load_file

import wandb
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, DDIMScheduler, DDPMScheduler

from dataset import ProcDataset
from utils.data_utils import proc_collate_fn
from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Eval_args

import torch
import csv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")

HAND_TOKEN = 2463


def save_locally(log, out_path, out_csv):
    # save locally
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        with open(os.path.join(out_path, out_csv), 'w') as file:
            file.write(csv_header + '\n')

    filename = os.path.splitext(log["source_filename"])[0] + "_" + str(log["id"]) + \
               os.path.splitext(log["source_filename"])[1]
    log["image"].save(os.path.join(out_path, filename))

    csv_line = ",".join([os.path.splitext(filename)[0], "train", cat2classficationhead(log["category"])])

    with open(os.path.join(out_path, out_csv), 'a') as file:
        file.write(csv_line + '\n')

def check_oih_dino(images, objects):
    def check_overlap(hand_boxes, obj_boxes):
        hand_centers_x = (hand_boxes[:, 0] + hand_boxes[:, 2]) / 2
        hand_centers_y = (hand_boxes[:, 1] + hand_boxes[:, 3]) / 2
        hand_sizes_x = hand_boxes[:, 2] - hand_boxes[:, 0]
        hand_sizes_y = hand_boxes[:, 3] - hand_boxes[:, 1]
        max_distances = (torch.max(hand_sizes_x, hand_sizes_y) / 2) * 2

        obj_centers_x = (obj_boxes[:, 0] + obj_boxes[:, 2]) / 2
        obj_centers_y = (obj_boxes[:, 1] + obj_boxes[:, 3]) / 2
        obj_sizes_x = obj_boxes[:, 2] - obj_boxes[:, 0]
        obj_sizes_y = obj_boxes[:, 3] - obj_boxes[:, 1]
        # max_obj_distances = (torch.max(obj_sizes_x, obj_sizes_y) / 2) + 10

        distances = torch.sqrt((hand_centers_x[:, None] - obj_centers_x[None, :]) ** 2 + (
                    hand_centers_y[:, None] - obj_centers_y[None, :]) ** 2)
        return torch.any(distances < max_distances[:, None])

    with torch.no_grad():
        text_prompt = [". ".join(p+["hand. face."]) for p in objects]
        inputs = processor(images=images, text=text_prompt, return_tensors="pt", padding=True).to("cuda")
        outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.2, text_threshold=0.3, target_sizes=[im.size[::-1] for im in images])

        valid = []
        for b in range(len(images)):
            obj_pred_boxes = [results[b]["boxes"][i] for i, l in enumerate(results[b]["labels"]) if l in list(filter(lambda x: x != "" and x != ", ", objects[b]))]
            hand_pred_boxes = [results[b]["boxes"][i] for i, l in enumerate(results[b]["labels"]) if l in ["hand"]]
            face_pred_boxes = [results[b]["boxes"][i] for i, l in enumerate(results[b]["labels"]) if l in ["face"]]
            valid.append(False if (len(obj_pred_boxes) == 0 or len(hand_pred_boxes) == 0) else len(face_pred_boxes) > 0 and bool(check_overlap(torch.stack(hand_pred_boxes), torch.stack(obj_pred_boxes))))

        return valid

args = Eval_args().parse_args()

csv_header = "label_id,split,phone_binary,cigarette_binary,food_binary"
cat2classficationhead = lambda cat: "1,0,0" if cat in ["phone"] else \
                                    "0,1,0" if cat in ["cigarette"] else \
                                    "0,0,1" if cat in ["food", "drink"] else \
                                    "0,0,0"

# init ControlNet(s)
controlnet = None
controlnet_text_encoder = None
num_controlnets = 0
class_cond = False
if args.controlnet_model_name_or_path is not None:
    controlnet = [ControlNetModel.from_pretrained(net_path, torch_dtype=torch.float32) for net_path in args.controlnet_model_name_or_path]
    for net_path, net in zip(args.controlnet_model_name_or_path, controlnet):
        w_dict = load_file(os.path.join(net_path, "diffusion_pytorch_model.safetensors"))
        if w_dict.get("class_embedding.weight", None) is not None:
            class_cond = True
            net.class_embedding = torch.nn.Embedding(4, 1280)
        net.load_state_dict(w_dict)
    num_controlnets = len(controlnet)
    if num_controlnets == 1:
        controlnet = controlnet[0]
    controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(torch.device("cuda"))

# init pipeline
pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_prompt_seq_projection=False,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

# init LoRA modules
# unet_lora_config = LoraConfig(r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],)
# pipeline.unet.add_adapter(unet_lora_config)
# lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict("./out/lora/model_output_20240524/checkpoint-15000")
# # lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict("./out/stabilityai2_lora4096-verizon/checkpoint-1000")
# unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
# unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
# b = set_peft_model_state_dict(pipeline.unet, unet_state_dict, adapter_name="default")
fix = "lora/checkpoint-238000"
fix = "unet_lora64_32-fix/lora-fixed/checkpoint-136000"
LycorisNetwork.apply_preset({"target_name": [".*attn.*"]})
lyco = create_lycoris(pipeline.unet, 1.0, linear_dim=64, linear_alpha=32, algo="lora").cuda()
lyco.apply_to()
lyco_state = torch.load(f"./out/{fix}/lycorice.ckpt", map_location=torch.device("cuda"))
lyco.load_state_dict(lyco_state, strict=False)
lyco.restore()
lyco = lyco.cuda()
lyco.merge_to(1.0)

# memory optimization
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

# data
fix = "lora_fix"
gen_start_index = 0
categories = {"food":0.07, "drink":0.07, "phone":0.52, "cigarette":0.10, "default":0.24}
valid_counts = {category: 0 for category in categories}
invalid_counts = {category: 0 for category in categories}
procedural_dataset = ProcDataset(args.evaluation_file, args.num_validation_images, categories, num_controlnets)
procedural_dataloader = torch.utils.data.DataLoader(procedural_dataset, shuffle=True, batch_size=args.batch_size, num_workers=0, collate_fn=proc_collate_fn)

for gen_id, batch in enumerate(procedural_dataloader):
    with torch.no_grad():
        with torch.autocast(f"cuda"):
            pred_images, _ = pipeline(prompt=batch["prompt"], controlnet_prompt=batch["control_prompt"], negative_prompt=batch["neg_prompt"], focus=HAND_TOKEN, class_conditional=torch.tensor(batch["class"], device=pipeline.device) if class_cond else None,
                                  image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["mask_conditioning"], height=512, width=512, self_guidance_scale=0,
                                  strength=1.0, controlnet_conditioning_scale=[0.0, 0.0], num_inference_steps=40, guidance_scale=8, guess_mode=True, generator=generator, gradient_checkpointing=args.gradient_checkpointing, return_dict=False)
            object_names = [[ctrl_obj] if cat!="default" else ["food", "drink", "phone", "cigarette"] for cat, ctrl_obj in zip(batch["category"], [b[-1] for b in batch["control_prompt"]] if type(controlnet) == list else batch["control_prompt"])]
            oih_results = check_oih_dino(pred_images, object_names)

    # log&save final result
    for i, (pred_image, oih_result) in enumerate(zip(pred_images, oih_results)):
        log = {"id":gen_start_index+(gen_id*args.batch_size)+i, "source": batch["image"][i], "category": batch["category"][i], "image": pred_image, "oih_flag": oih_result, "source_filename":os.path.basename(batch["image_name"][i]), "prompt": batch["prompt"][i], "control": batch["control_prompt"][i][-1] if type(controlnet)==list else batch["control_prompt"][i]}

        save_locally(log, out_path=os.path.join("./out", f"train_data_gen_{fix}"), out_csv="train.csv")
        if log["oih_flag"] or (log["category"]=="default" and not log["oih_flag"]):
             save_locally(log, out_path=os.path.join("./out", f"train_data_gen_th_{fix}"), out_csv="train.csv")
             valid_counts[log["category"]] += 1
        else: invalid_counts[log["category"]] += 1

    # update sampling proportions re-balancing the original up to 0.8
    total_valid = sum(valid_counts.values())
    observed_distribution = {category: count / total_valid if total_valid > 0 else 0 for category, count in valid_counts.items()}
    modified_probs = {category: max(0, categories[category] - (observed_distribution[category] * 0.8)) for category in categories}
    normalized_probs = {category: prob / sum(modified_probs.values()) for category, prob in modified_probs.items()}
    procedural_dataset.update_cat_proportions(normalized_probs)

print("Valid:", valid_counts)
print("Invalid:", invalid_counts)
