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
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import randn_tensor
from lycoris import create_lycoris, LycorisNetwork, create_lycoris_from_weights
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn

from torchvision.transforms import RandomAffine, RandomPerspective, RandomApply
import wandb
from hopfield_energy import hopfield_loss
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.pipelines.controlnet import MultiControlNetModel
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from transformers import CLIPTextModelWithProjection
from diffusers import ControlNetModel, DDIMScheduler

from dataset import ProcDataset
from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from third.verizon.ludovicodl.nn.models.object_in_hand import get_model
from third.verizon.ludovicodl.training.learning import MultiTask
from utils.parser import Optim_args
from utils.data_utils import not_image, proc_collate_fn
from vsource.evaluation.tasks import PhoneUsageTask, CigaretteTask, FoodTask

HAND_TOKEN = 2463


def save_locally(log, out_path, out_csv):
    # save locally
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        with open(os.path.join(out_path, out_csv), 'w') as file:
            file.write(csv_header + '\n')

    filename = os.path.splitext(log["source_filename"])[0] + "_" + str(log["id"]) + \
               os.path.splitext(log["source_filename"])[1]
    save_image(log["predictions"], os.path.join(out_path, filename))

    csv_line = ",".join([os.path.splitext(filename)[0], "train", cat2classficationhead(log["category"])])

    with open(os.path.join(out_path, out_csv), 'a') as file:
        file.write(csv_line + '\n')


def read_energy_memory(root_folder):
    # read memory
    memory = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pt'):
                filepath = os.path.join(dirpath, filename)
                class_name = filename.split('_')[1].split('.')[0]
                tensor = torch.load(filepath)
                if class_name in memory:
                    memory[class_name].append(tensor)
                else:
                    memory[class_name] = [tensor]
    for class_name, class_tensors in memory.items():
        memory[class_name] = torch.stack(class_tensors)
    return memory


logger = get_logger(__name__)
args = Optim_args().parse_args()

if args.log == "wandb":
    wandb.init(entity="johnminelli", project="train_controlnet")

csv_header = "label_id,split,phone_binary,cigarette_binary,food_binary"
cat2classficationhead = lambda cat: "1,0,0" if cat in ["phone"] else \
                                    "0,1,0" if cat in ["cigarette"] else \
                                    "0,0,1" if cat in ["food", "drink"] else \
                                    "0,0,0"
task_to_class = {'phone_binary': ["phone"], 'cigarette_binary': ["cigarette"], 'food_binary': ["food", "drink"]}
class_to_task = {'phone':['phone_binary'], 'cigarette':['cigarette_binary'], 'food':['food_binary'], 'drink':['food_binary'], 'default':['phone_binary', 'cigarette_binary', 'food_binary']}

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
        controlnet_prompt_seq_projection=True,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float32
    ).to(torch.device("cuda"))
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# init LoRA modules
# unet_lora_config = LoraConfig(r=256, lora_alpha=128, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],)
# pipeline.unet.add_adapter(unet_lora_config)
# lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict("./out/lora/model_output_20240524/checkpoint-15000")
# unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
# unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
# b = set_peft_model_state_dict(pipeline.unet, unet_state_dict, adapter_name="default")
fix = "unet_lora64_32-fix/lora-weighted/checkpoint-150000"
# LycorisNetwork.apply_preset({"target_name": [".*attn.*"]})  # Transformer2DModel
LycorisNetwork.apply_preset({"target_module": ["Attention"], })
lyco = create_lycoris(pipeline.unet, 1.0, linear_dim=64, linear_alpha=32, algo="lora").cuda()
lyco.apply_to()
lyco_state = torch.load(f"./out/{fix}/lycorice.ckpt", map_location=torch.device("cuda"))
lyco.load_state_dict(lyco_state, strict=False)
lyco.restore()
lyco = lyco.cuda()
lyco.merge_to(1.0)

# classifier setup
tasks = [PhoneUsageTask(), CigaretteTask(), FoodTask()]
cg_task = MultiTask(tasks=tasks, name='multi-task-dfc-distraction')
classifier_real = get_model(backbone="efficientnet_v2_s", tasks=tasks)
classifier_gen = get_model(backbone="efficientnet_v2_s", tasks=tasks)

# accelerator setup for optimized execution
logging_dir = Path("./out/oih-optim", "logs")
accelerator_project_config = ProjectConfiguration(project_dir="./out/oih-optim", logging_dir=logging_dir)
accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision=None, log_with=None, project_config=accelerator_project_config, )
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger.info(accelerator.state, main_process_only=False)
transformers.utils.logging.set_verbosity_warning()
diffusers.utils.logging.set_verbosity_info()

# memory optimization
if args.enable_cpu_offload:
    pipeline.enable_model_cpu_offload()
if args.enable_xformers_memory_efficient_attention:
    pipeline.enable_xformers_memory_efficient_attention()
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

pipeline.vae.requires_grad_(False)
pipeline.unet.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
if num_controlnets > 0:
    pipeline.controlnet_text_encoder.requires_grad_(False)
    pipeline.controlnet.requires_grad_(False)
classifier_real.requires_grad_(False)
classifier_real.eval()
classifier_gen.requires_grad_(False)
classifier_gen.eval()
classifier_real.load_state_dict((lambda f,l: {k:v for k,v in zip(f.keys(), l.values())})(classifier_real.state_dict(), torch.load("./out/best_model_31_val_score=2.6616.pt", map_location=accelerator.device)))
classifier_gen.load_state_dict((lambda f,l: {k:v for k,v in zip(f.keys(), l.values())})(classifier_gen.state_dict(), torch.load("./out/oih-emb_lora_fix_ctrl_50th/checkpoint_63594.pt", map_location=accelerator.device)["model"]))
real_data = read_energy_memory('/data01/gio/ctrl/data/embeddings_best')

# data
# categories = {"food":0.07, "drink":0.07, "phone":0.52, "cigarette":0.10, "default":0.24}
categories = {"food":0.5, "drink":0.5, "phone":0., "cigarette":0., "default":0.}
procedural_dataset = ProcDataset(args.evaluation_file, args.num_validation_images, categories, num_controlnets)
procedural_dataloader = torch.utils.data.DataLoader(procedural_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.dataloader_num_workers, collate_fn=proc_collate_fn)

# setup accelerator library
pipeline.unet, pipeline.controlnet, pipeline.vae, classifier_real, classifier_gen, procedural_dataloader = accelerator.prepare(pipeline.unet, pipeline.controlnet, pipeline.vae, classifier_real, classifier_gen, procedural_dataloader)
pipeline.text_encoder.to(accelerator.device, dtype=torch.float32)
if num_controlnets > 0:
    pipeline.controlnet_text_encoder.to(accelerator.device, dtype=torch.float32)

augmentations = nn.Sequential(
            RandomApply([RandomAffine(degrees=15, translate=(0.1, 0.1))], p=0.7),
            RandomPerspective(0.7, p=0.7),
        )
# generation loop
skip = 0
for gen_id, batch in enumerate(procedural_dataloader):
    if gen_id<skip:
        continue
    num_inference_steps = 35
    start_optim_step = 0  # 0 --> start_optim_step --> num_inference_steps
    num_optim_steps = 1  # min: 0; max: `num_inference_steps`-1
    free_guidance_scale = 8
    classifier_guidance_scale = 5.
    self_guidance_scale = 0
    controlnet_scale = 1.
    grads = []
    scores = []
    energies = []
    # optimize generation for given prompts
    print(f"Generation {gen_id}, {str(batch['category'][0])}")
    for i in range(num_optim_steps):
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        # generate the image from text
        with torch.no_grad():
            pred_images, cg_image = pipeline(prompt=batch["prompt"], controlnet_prompt=batch["control_prompt"], negative_prompt=batch["neg_prompt"], aux_focus_prompt=batch["focus_prompt"], aux_focus_token=None, dynamic_masking=True, class_conditional=torch.tensor(batch["class"], device=accelerator.device),
                                  image=batch["image"], mask_image=batch["mask"], conditioning_image=batch["mask_conditioning"], height=512, width=512, self_guidance_scale=self_guidance_scale,
                                  strength=1.0, controlnet_conditioning_scale=controlnet_scale, num_inference_steps=num_inference_steps, guidance_scale=free_guidance_scale, cg_step=start_optim_step,
                                  cg_values=grads.copy(), guess_mode=True, generator=generator, output_type="pt", gradient_checkpointing=args.gradient_checkpointing, return_dict=False)
        # classify the output image
        # pred_images_aug = torch.cat([pred_images, augmentations(pred_images.repeat(7,1,1,1))])
        # out_real = cg_task.forward_pass(x=pred_images_aug, y={k:torch.tensor([1 if cat in v else 0 for cat in batch["category"]]*8).to(accelerator.device) for k,v in task_to_class.items()}, model=classifier_real, device=accelerator.device)
        # out_gen = cg_task.forward_pass(x=pred_images_aug, y={k:torch.tensor([1 if cat in v else 0 for cat in batch["category"]]*8).to(accelerator.device) for k,v in task_to_class.items()}, model=classifier_gen, device=accelerator.device)
        # score = torch.stack([torch.cat([torch.sigmoid(out_real[task]["out"]) for task in class_to_task[cat]]).mean() for cat in batch["category"]])
        # log_neg_probs = torch.log(1-score)
        # Force in-distribution sampling
        # distribution_energy = hopfield_loss(real_data["cigarette"].squeeze(), real_data["noclass"].squeeze(), out_real["last_layer"].mean(0)[None,:], beta_b=10)[1]["onesided_energy"]
        # logging
        # print("SCORE:", score.detach().cpu().numpy(), "ENERGY:", distribution_energy.detach().cpu().numpy())
        # scores.append(score.detach().cpu().numpy()[0])
        # energies.append(distribution_energy.detach().cpu().numpy()[0])
        # wandb.log({f"Generation/Gen_{gen_id}_image": wandb.Image(pared_images[0].permute(1, 2, 0).detach().cpu().numpy(), caption=str(batch["category"][0])), f"Generation/Gen_{gen_id}_score":scores[0][0].detach().cpu().numpy(), f"Generation/Gen_{gen_id}_delta":(or_prompt_embeds-prompt_embeds).sum().detach().cpu().numpy()})
        plt.imshow(pred_images[0].permute(1, 2, 0).detach().cpu()); plt.title(str(gen_id)+". "+str(batch['category'][0])); plt.show()

        # if float(out_real['entropy'])<0.1 and float(out_gen['entropy'])>0.1:
        #     del out_real; del out_gen; del log_neg_probs; break
        # get optimization direction
        # if i < num_optim_steps-1:
            # if len(energies)>1 and np.abs(energies[-1])>np.abs(energies[-2]):
            #     grads[-1] = torch.zeros_like(grads[-1])
            #     energies.pop(-1)                                  out_real['entropy']+(1-out_gen['entropy'])
        #     grads.append(torch.autograd.grad(log_neg_probs, cg_image)[0] * classifier_guidance_scale)
        # del out_real; del out_gen; del log_neg_probs
    # plt.plot(scores); plt.title(f"{str(batch['category'][0])} {num_optim_steps} out of {num_inference_steps}"); plt.show()

    # log&save final result
    for i, pred_image in enumerate(pred_images):
        log = {"id": (gen_id*args.batch_size)+i, "source": batch["image"][i], "category": batch["category"][i], "predictions": pred_image, "source_filename": os.path.basename(batch["image_name"][i]), "prompt": batch["prompt"][i], "control": batch["control_prompt"][i][-1] if type(controlnet) == list else batch["control_prompt"][i]}

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
            save_locally(log, out_path=os.path.join("./out", f"train_data_gen_optim_f_{fix}"), out_csv="train.csv")
