#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import codecs
import copy
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from lycoris import create_lycoris, LycorisNetwork
from diffusers.training_utils import cast_training_params

from utils.dataset import DiffuserDataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPVisionModel, CLIPTextModel, CLIPTextModelWithProjection, \
    CLIPVisionModelWithProjection

import diffusers
from diffusers import (AutoencoderKL, ControlNetModel, UNet2DConditionModel, DDPMScheduler, PNDMScheduler)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipeline.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Train_args
from utils.utils import import_text_encoder_from_model_name_or_path, replicate, mask_block, compute_centroid

if is_wandb_available():
    import wandb
    # wandb.init(project="train_controlnet", resume="o0emanqi")


logger = get_logger(__name__)
# Implementation of training procedure with CLIP images/text. The images used are the cropped and resized objects present in the prompt.json with the name of "ctrl_txt_image".
# Note, CLIPVisualEnc and CLIPTextEnc give in output (a different number of tokens, but this is not relevant, and) an embedding with a sifferent embedding size. In the normal implementation it does exists a projection layer being applied only to the last/first(?) token of the two output and project it in the common hidden dimension (that is the same as the CLIPTextEnc). This is standard to use it for all the tokens and have sequences of tokens representing image or text at same dimension.
MULTIMODAL_CROSS_ATTN = False
# Implementation of a weighted loss in the segmented area of the object (from Segment Anything on top of GroundingDINO detection). The object mask should be present in the prompt.json with name "obj_mask".
LOSS_IN_SAM = False

def log_validation(vae, text_encoder, controlnet_text_encoder, controlnet_image_encoder, tokenizer, unet, lyco_state, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")
    # get trained network
    controlnet = accelerator.unwrap_model(controlnet)
    # costly deepcopy but necessary if you mod cross attention layers
    unet_clone = copy.deepcopy(unet)
    LycorisNetwork.apply_preset({"target_module": ["ResnetBlock2D"]})
    lyco = create_lycoris(unet_clone, 1.0, linear_dim=args.rank, linear_alpha=args.alpha_rank, algo=args.lycorice_algo).cuda()
    lyco.apply_to()
    lyco = lyco.cuda()
    lyco.load_state_dict(lyco_state, strict=False)
    lyco.restore()
    lyco.cuda()
    lyco.merge_to(1.0)

    pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_image_encoder=controlnet_image_encoder,
        controlnet_prompt_seq_projection=False,
        tokenizer=tokenizer,
        unet=unet_clone,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.enable_cpu_offload:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    validation_data = json.load(codecs.open(args.validation_file, 'r', 'utf-8-sig'))

    # Access the values as a dictionary
    images = validation_data['validation_images']
    n = len(images)
    conditioning = replicate(validation_data.get('validation_conditioning', []), n)
    masks = replicate(validation_data.get('validation_masks', []), n)
    prompts = replicate(validation_data.get('validation_prompts', []), n)
    neg_prompts = replicate(validation_data.get('validation_neg_prompts', []), n)
    control_prompts = replicate(validation_data.get('validation_control_prompts', []), n)
    focus_prompts = replicate(validation_data.get('validation_focus_prompts', []), n)
    class_conditional = replicate(torch.tensor(validation_data.get('validation_class', [])).to(accelerator.device), n)

    image_logs = []
    for prompt, neg_prompt, image, mask, condition, control_prompt, focus_prompt, class_ in zip(prompts, neg_prompts, images, masks, conditioning, control_prompts, focus_prompts, class_conditional):
        image = Image.open(image).convert("RGB") if image is not None else None
        mask = Image.open(mask).convert("L") if mask is not None else None
        mask_conditioning = Image.open(condition).convert("RGB") if condition is not None else None
        focus_prompt = [focus_prompt] if focus_prompt else None

        with torch.no_grad():
            pred_images = []
            for i in range(args.num_validation_images):
                with torch.autocast(f"cuda"):
                    pred_image = pipeline(prompt=prompt, controlnet_prompt=control_prompt, negative_prompt=neg_prompt, aux_focus_prompt=focus_prompt, dynamic_masking=i>1, class_conditional=class_,
                                          image=image, mask_image=mask, conditioning_image=mask_conditioning, height=512, width=512, self_guidance_scale=0,
                                          strength=1.0, controlnet_conditioning_scale=1.0, num_inference_steps=50, guidance_scale=7.5, guess_mode=True, generator=generator).images[0]
                pred_images.append(pred_image)

            image_logs.append({"reference": image, "images": pred_images, "prompt": ", ".join([str(prompt), str(control_prompt)])})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                prompt = log["prompt"]
                image = log["reference"]

                formatted_images = [np.asarray(image)]
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                prompt = log["prompt"]
                image = log["reference"]

                formatted_images.append(wandb.Image(image, caption="Reference"))
                for image in images:
                    if image is not None:
                        image = wandb.Image(image, caption=prompt)
                        formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


class Trainer():
    def __init__(self, args):
        self.args = args
        logging_dir = Path(args.output_dir, args.logging_dir)
        self.safety_checker = None

        # Load the accelerator
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.log,
            project_config=accelerator_project_config,
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

            if args.push_to_hub:
                self.repo_id = create_repo(
                    repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
                ).repo_id

        # Load the tokenizer
        if args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
        elif args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )

        # Dataset
        self.train_dataset = [DiffuserDataset(data_dir, args.train_data_file, 512, self.tokenizer, apply_transformations=False, dilated_conditioning_mask=False) for data_dir in args.train_data_dir]
        self.train_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(self.train_dataset), shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, drop_last=True)

        # import correct text encoder class
        text_encoder_cls = import_text_encoder_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        # Custom encoder since sd2 doesn't have the weights for the visual counterpart necessary for obj_masking controlnet 
        self.controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        self.controlnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)  # , local_files_only=True
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet")
            net_for_w = copy.deepcopy(self.unet)
            if self.unet.config.in_channels != 4:
                net_for_w.conv_in.weight = torch.nn.Parameter(self.unet.conv_in.weight[:,:4])
                net_for_w.config.in_channels = 4
            self.controlnet = ControlNetModel.from_unet(net_for_w)
            if args.use_classemb:
                self.controlnet.class_embedding = torch.nn.Embedding(4, 1280)

        def unwrap_model(model):
            model = self.accelerator.unwrap_model(model)
            from diffusers.utils.torch_utils import is_compiled_module
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                self.lyco.save_weights(os.path.join(output_dir, "lycorice.ckpt"), None, None)

        def load_model_hook(models, input_dir):
            unet_ = None
            text_encoder_ = None

            while len(models) > 0:
                model = models.pop()
                # if isinstance(model, type(unwrap_model(self.unet))):
                #     unet_ = model
                # else:
                #     raise ValueError(f"unexpected save model: {model.__class__}")

            # mmm you don't have access to the lyco net here
            # lyco_state = torch.load(os.path.join(input_dir, "lycorice.ckpt"))
            # self.lyco.load_state_dict(lyco_state)

            # Make sure the trainable params are in float32. This is again needed since the base models are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [unet_]
                if args.train_text_encoder:
                    models.append(text_encoder_)

                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models, dtype=torch.float32)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.controlnet_image_encoder.requires_grad_(False)
        self.controlnet_text_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        LycorisNetwork.apply_preset({"target_module": ["ResnetBlock2D"]})  # by module (e.g. Attention, Transformer2DModel, ResnetBlock2D) or by wildcard (e.g. {"target_name": [".*attn.*"]})
        self.lyco = create_lycoris(self.unet, 1.0, linear_dim=args.rank, linear_alpha=args.alpha_rank, algo=args.lycorice_algo).cuda()
        self.lyco.apply_to()
        self.lyco.cuda()

        # Setup LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers

        # Settings for optimization
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * self.accelerator.num_processes
            )

        # Optimizer
        if args.use_8bit_adam:
            try:  # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        self.optimizer = optimizer_class(
            list(filter(lambda p: p.requires_grad, self.lyco.parameters())),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Scheduler and math around the number of training steps.
        override_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            override_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Prepare everything with accelerator library
        self.unet, self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(self.unet, self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler)

        # Move vae, and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet_text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.controlnet_image_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        # Init class to have access to utility functions (refactoring to static methods would be nicer)
        self.pipe_utils = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(args.pretrained_model_name_or_path, vae=self.vae, text_encoder=self.text_encoder, controlnet_image_encoder=self.controlnet_image_encoder, controlnet_text_encoder=self.controlnet_text_encoder, tokenizer=self.tokenizer, unet=self.unet, controlnet=self.controlnet, safety_checker=None, revision=args.revision, torch_dtype=self.weight_dtype)

        # Initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

        # Train variables
        self.global_step = 0
        self.first_epoch = 0
        self.initial_global_step = 0
        self.num_samples = len(self.train_dataset)
        self.num_batches_per_epoch = len(self.train_dataloader)
        self.total_batch_size = args.train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if override_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterward we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        args.resume = "latest"
        # Load the weights and states from a previous save, if necessary
        if args.resume:
            if args.resume != "latest":
                path = os.path.basename(args.resume)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{args.resume}' does not exist. Starting a new training run."
                )
                args.resume = None
                self.initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(args.output_dir, path))
                lyco_state = torch.load(os.path.join(os.path.join(args.output_dir, path), "lycorice.ckpt"))
                self.lyco.load_state_dict(lyco_state)
                self.global_step = int(path.split("-")[1])

                self.initial_global_step = self.global_step
                self.first_epoch = self.global_step // num_update_steps_per_epoch


    def train(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {self.num_samples}")
        logger.info(f"  Num batches each epoch = {self.num_batches_per_epoch}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        progress_bar = tqdm(range(0, args.max_train_steps), initial=self.initial_global_step, desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.first_epoch, args.num_train_epochs):
            self.unet.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Convert images to latent space
                    image = batch["image"].to(dtype=self.weight_dtype)
                    mask = batch.get("mask").to(dtype=self.weight_dtype) if "mask" in batch else None
                    conditioning_image = batch["conditioning"].to(dtype=self.weight_dtype) if "conditioning" in batch else None
                    bs, _, h, w = image.shape
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.accelerator.device).long()

                    # Latents from image
                    latents = self.pipe_utils._encode_vae_image(image)
                    # Sample random noise and add it to the latents according to the noise magnitude at each timestep (forward diffusion process)
                    noise = torch.randn_like(latents)

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents_model_input = noisy_latents
                    timesteps_model_input = timesteps

                    # Mask and latents from masked_image
                    if mask is not None:
                        masked_image = image.clone()
                        masked_image[mask.expand(image.shape) > 0.5] = 0  # 0 masking in a [-1,1] image because this is what the unet wants
                        mask, masked_image_latents = self.pipe_utils.prepare_mask_latents(mask, masked_image, 1, h, w, self.weight_dtype, self.accelerator.device, generator=None, do_classifier_free_guidance=False)

                    flag_image_as_hid_prompt = (np.random.random() > 0.5) if MULTIMODAL_CROSS_ATTN else False

                    encoder_hidden_states = self.pipe_utils.encode_prompt(batch["txt"], self.accelerator.device, False, self.text_encoder, num_images_per_prompt=1, negative_prompt=batch["no_txt"], return_tuple=False)
                    encoder, ctrl_prompt = (self.controlnet_image_encoder, batch["ctrl_txt_image"]) if flag_image_as_hid_prompt else (self.controlnet_text_encoder, batch["ctrl_txt"])
                    encoder_hidden_states_ctrl = self.pipe_utils.encode_prompt(ctrl_prompt, self.accelerator.device, False, encoder, num_images_per_prompt=1, negative_prompt=batch["no_txt"], return_tuple=False)

                    # Apply projection layer to all sequence tokens to match size between the two encoders. Apparently there is no loss in performance so we do it anyway to avoid misuse later on 
                    # encoder_hidden_states_ctrl = self.controlnet_image_encoder.visual_projection(encoder_hidden_states_ctrl) \
                    #     if flag_image_as_hid_prompt else self.controlnet_text_encoder.text_projection(encoder_hidden_states_ctrl)

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents_model_input, timesteps_model_input,
                        encoder_hidden_states=encoder_hidden_states_ctrl,
                        controlnet_cond=conditioning_image,
                        class_labels=batch["class"].squeeze() if self.args.use_classemb else None,
                        return_dict=False,
                    )

                    if self.args.enable_cpu_offload:
                        torch.cuda.empty_cache()

                    if mask is not None and self.unet.config.in_channels == 9:
                        # if controlnet is trained with MASK_CTRL_OUT this should be uncommented
                        # mid_block_res_sample = mask_block(conditioning_image[:, :1], mid_block_res_sample)
                        # down_block_res_samples = [mask_block(conditioning_image[:, :1], block) for block in down_block_res_samples]
                        noisy_latents_model_input_unet = torch.cat([noisy_latents_model_input, mask, masked_image_latents], dim=1)

                    # Predict the noise residual
                    noise_pred = self.unet(
                        noisy_latents_model_input_unet, timesteps_model_input,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    # Object loss enhance
                    if LOSS_IN_SAM and batch.get("obj_mask", None) is not None and batch.get("obj_mask", None).any():
                        obj_mask = F.interpolate(batch.get("obj_mask"), size=(h // self.pipe_utils.vae_scale_factor, w // self.pipe_utils.vae_scale_factor))
                        obj_mask = obj_mask.to(device=self.accelerator.device, dtype=self.weight_dtype)
                        loss += loss*obj_mask*0.1

                    loss = loss.mean()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = list(filter(lambda p: p.requires_grad, list(self.lyco.parameters())))
                        self.accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    if self.accelerator.is_main_process:
                        if self.global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{self.global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if args.validation_file is not None and self.global_step % args.validation_steps == 0:
                            log_validation(self.vae, self.text_encoder, self.controlnet_text_encoder, self.controlnet_image_encoder, self.tokenizer, self.unet, self.lyco.state_dict(), self.controlnet, self.args, self.accelerator, self.weight_dtype, self.global_step)
                            torch.cuda.empty_cache()

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= args.max_train_steps:
                    break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unet = self.unet.to(torch.float32)
            self.lyco.merge_to(1.0)
            unet.save_pretrained(args.output_dir)

        self.accelerator.end_training()


if __name__ == "__main__":
    args = Train_args().parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    Trainer(args).train()
