import json
import os
import numpy as np
from lycoris import create_lycoris, LycorisNetwork
from peft import set_peft_model_state_dict, LoraConfig
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor

import wandb
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, PNDMScheduler, DDPMScheduler, UniPCMultistepScheduler, DDIMScheduler

from data.vcoco.categories import object_categories
from utils.dataset import ProcDataset, DiffuserDataset

from preprocess.dataset_ditribution import read_json, count_triplets
from utils.data_utils import proc_collate_fn
from pipeline.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.dataset import SynthDataset
from utils.parser import Eval_args

import torch
import csv
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")

HAND_TOKEN = 2463

def save_locally(log, out_path, out_json):
    # save locally 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        with open(os.path.join(out_path, out_json), 'w') as file:
            file.write('[')

    image = log["image"]
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes for subjects
    for subject_bbox in log["hoi_subjects"]:
        draw.rectangle(subject_bbox, outline="red", width=2)
    # Draw bounding boxes for objects
    for object_bbox in log["hoi_objects"]:
        draw.rectangle(object_bbox, outline="blue", width=2)

    for hoi_ann in log["hoi_annotation"]:
        draw.rectangle(log["annotations"][hoi_ann["subject_id"]]["bbox"], outline="green", width=2)
        draw.rectangle(log["annotations"][hoi_ann["object_id"]]["bbox"], outline="green", width=2)
    # save image
    filename = os.path.splitext(log["source_filename"])[0] + "_" + str(log["id"]) + os.path.splitext(log["source_filename"])[1]
    image.save(os.path.join(out_path, filename))
    # append hoi
    hoi_annotation = log["hoi_annotation"] + [{"subject_id":  len(log["annotations"])+s_i, "category_id": log["role_category"], "object_id": len(log["annotations"])+len(log["hoi_subjects"])+o_i} for s_i, _ in enumerate(log["hoi_subjects"]) for o_i, _ in enumerate(log["hoi_objects"])]
    annotations = log["annotations"] + [{"category_id": log["subject_category"], "bbox": subject_bbox} for subject_bbox in log["hoi_subjects"]]+ \
                   [{"category_id": log["object_category"], "bbox": object_bbox} for object_bbox in log["hoi_objects"]]
    json_line = {"file_name": filename, "hoi_annotation": hoi_annotation, "annotations": annotations}
    with open(os.path.join(out_path, out_json), 'a') as outfile:
        json.dump(json_line, outfile)
        outfile.write(', ')


def check_hoi_dino(images, masks, subjects, objects, excluded_detections):
    def check_overlap(candidate, references, th=None, iou_th=None): 
            """
            Check IoU of candidate respect references boxes
            :param candidate: list of a box coordinates [x0, y0, x1, y1]
            :param references: list of list of boxes [[x0, y0, x1, y1], ...]
            :return: True if candidate IoU > 0.9 for any of the references 
            """
            for reference in references:
                # coordinates of the intersection rectangle
                xA = max(candidate[0], reference[0])
                yA = max(candidate[1], reference[1])
                xB = min(candidate[2], reference[2])
                yB = min(candidate[3], reference[3])
                # area of intersection rectangle
                interWidth = max(0, xB - xA)
                interHeight = max(0, yB - yA)
                interArea = interWidth * interHeight
                # area of both the prediction and reference rectangle
                boxAArea = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
                boxBArea = (reference[2] - reference[0]) * (reference[3] - reference[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)

                if (iou_th is not None and iou > iou_th) or (th is not None and interArea > th * boxBArea):
                    return True
            return False
        
    inputs = processor(images=images, text=[f"{s}." for s in subjects], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)
    results_subj = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.5, text_threshold=0.3, target_sizes=[im.size[::-1] for im in images])
    inputs = processor(images=[pil_to_tensor(im)*m for im, m in zip(images, masks)], text=[f"{o}." for o in objects], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs)
    results_obj = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.45, text_threshold=0.3, target_sizes=[im.size[::-1] for im in images])
    masks_boxes = [[np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)] if (len(x_indices := np.where(mask[0] == 1)[1]) > 0 and len(y_indices := np.where(mask[0] == 1)[0]) > 0) else (0, 0, 0, 0) for mask in masks]
    
    boxes = []
    for i in range(len(images)):
        # subjects are detected in all image canvas therefore we compare and drop detections "overlapping" excluded_detections (i.e. [(DINO detections + GT boxes) - HOI instance boxes] ). Note the thresholds of DINO detection here should >= (i.e. strict-ier) than those used in DINO detection during preprocessing.
        subj_pred_boxes = [b for b, l in zip(results_subj[i]["boxes"], results_subj[i]["labels"]) if (l==subjects[i] and not check_overlap(b, excluded_detections[i], iou_th=0.7)) and check_overlap(b, masks_boxes, th=0.15)]
        # objects are detected only in inpainted region, thus found instance should be valid and no additional filtering is necessary. DINO thresholds can be tuned if the generation quality is not the best.
        obj_pred_boxes = [b for b, l in zip(results_obj[i]["boxes"], results_obj[i]["labels"]) if (l==objects[i])]
        boxes.append({"hoi_subjects": torch.stack(subj_pred_boxes) if len(subj_pred_boxes)>0 else torch.tensor([]).to("cuda"),
                      "hoi_objects": torch.stack(obj_pred_boxes) if len(obj_pred_boxes)>0 else torch.tensor([]).to("cuda")})
    return boxes

args = Eval_args().parse_args()
# init ControlNet(s)
controlnet = None
controlnet_text_encoder = None
num_controlnets = 0
class_cond = False
if args.controlnet_model_name_or_path is not None:
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=torch.float32)
    w_dict = load_file(os.path.join(args.controlnet_model_name_or_path, "diffusion_pytorch_model.safetensors"))
    if w_dict.get("class_embedding.weight", None) is not None:
        class_cond = True
        controlnet.class_embedding = torch.nn.Embedding(4, 1280)
    controlnet.load_state_dict(w_dict)
    num_controlnets = 1
    controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(torch.device("cuda"))

# init pipeline
pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_prompt_seq_projection=True,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

# init LoRA modules
if args.lora_path is not None:
    LycorisNetwork.apply_preset({"target_module": ["ResnetBlock2D"]})  # by module (e.g. Attention, Transformer2DModel, ResnetBlock2D) or by wildcard (e.g. {"target_name": [".*attn.*"]})
    lyco = create_lycoris(pipeline.unet, 1.0, linear_dim=64, linear_alpha=32, algo="lora").cuda()
    lyco.apply_to()
    lyco_state = torch.load(f"{args.lora_path}/lycorice.ckpt", map_location=torch.device("cuda"))
    lyco.load_state_dict(lyco_state, strict=False)
    lyco.cuda()
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
fix = "hicodet_cg25"
gen_start_index = 0
valid_counts = {}
invalid_counts = {}
hoi_categories =  count_triplets(read_json(["/data01/gio/ctrl/data/hicodet/annotations/trainval_hico.json"]))
weights = {cat: 1 / count for cat, count in hoi_categories.items()}
tot_w = sum(weights.values())
cat_probabilities = {cat: w / tot_w for cat, w in weights.items()}
num_images = (max(hoi_categories.values()) * len(hoi_categories)) - sum([min(v, max(hoi_categories.values())) for v in hoi_categories.values()])

dataset = SynthDataset(args.train_data_dir[0], args.train_data_file, num_images if args.num_validation_images is None else min(num_images, args.num_validation_images), cat_probabilities, 512, dilated_conditioning_mask=False)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=0, drop_last=True)
sampling_distribution = dataset.categories

for gen_id, batch in enumerate(dataloader):
    with torch.no_grad():
        with torch.autocast(f"cuda"):
            pred_images = pipeline(prompt=batch["txt"], controlnet_prompt=batch["ctrl_txt"], negative_prompt=batch["neg_txt"], aux_focus_prompt=batch["neg_txt"], dynamic_masking=True,
                                  image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["conditioning"], height=512, width=512, 
                                  strength=1.0, controlnet_conditioning_scale=1.0, num_inference_steps=30, guidance_scale=7.5, self_guidance_scale=0,
                                  guess_mode=True, generator=generator, gradient_checkpointing=args.gradient_checkpointing, return_dict=False)
            hoi_results = check_hoi_dino(pred_images, batch["mask"], [object_categories[int(sc)] for sc in batch["subject_category"]], [object_categories[int(oc)] for oc in batch["object_category"]], [json.loads(b2i) for b2i in batch["boxes_to_ignore"]])

    # log&save final result
    for i, (pred_image, hoi_result) in enumerate(zip(pred_images, hoi_results)):
        log = {"id":gen_start_index+(gen_id*args.batch_size)+i, "source": batch["image"][i], "source_filename":os.path.basename(batch["image_name"][i]), 
               "image": pred_image, "prompt": batch["txt"][i], "control": batch["ctrl_txt"][i][-1] if type(controlnet)==list else batch["ctrl_txt"][i],
               "subject_category": int(batch["subject_category"][i]), "object_category": int(batch["object_category"][i]), "role_category": int(batch["role_category"][i]),
               "hoi_annotation":json.loads(batch["hoi_annotation"][i]), "annotations": json.loads(batch["annotations"][i]), **{k:v.cpu().tolist() if isinstance(v, torch.Tensor) else v for k,v in hoi_result.items()}}

        category = (int(batch["subject_category"][i]), int(batch["role_category"][i]), int(batch["object_category"][i]))
        save_locally(log, out_path=os.path.join("./out", f"train_data_gen_{fix}"), out_json="gen.json")
        if len(log["hoi_subjects"])>0 and len(log["hoi_objects"])>0:
             save_locally(log, out_path=os.path.join("./out", f"train_data_gen_th_{fix}"), out_json="gen.json")
             valid_counts[category] = valid_counts.get(category, 0)+1
        else: invalid_counts[category] = invalid_counts.get(category, 0)+1

    total_valid = sum(valid_counts.values())
    observed_distribution = {category: count / total_valid if total_valid > 0 else 0 for category, count in valid_counts.items()}

    new_sampling_distribution = {cat: max(0, val - (observed_distribution.get(cat, 0)*0.8)) for cat, val in sampling_distribution.items()}
    total = sum(new_sampling_distribution.values())
    new_sampling_distribution = {category: prob / total for category, prob in new_sampling_distribution.items()}
    dataset.update_cat_proportions(new_sampling_distribution)

print("Valid:", valid_counts)
print("Invalid:", invalid_counts)
