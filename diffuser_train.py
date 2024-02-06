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
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import load_file, save_file

from dataset import DiffuserDataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPVisionModel, CLIPTextModel, CLIPTextModelWithProjection, \
    CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipelines.pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetImg2ImgInpaintPipeline
from utils.parser import Train_args
from utils.utils import import_text_encoder_from_model_name_or_path, replicate, mask_block

if is_wandb_available():
    import wandb
    wandb.init(project="train_controlnet", resume="gjn070jq")


logger = get_logger(__name__)
TRAIN_OBJ_MASKING = False

def log_validation(vae, text_encoder, controlnet_text_encoder, controlnet_image_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")
    # get trained network
    controlnet = accelerator.unwrap_model(controlnet)
    # costly deepcopy but necessary if you mod cross attention layers
    unet_clone = copy.deepcopy(unet)

    pipeline = StableDiffusionControlNetImg2ImgInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        controlnet_text_encoder=controlnet_text_encoder,
        controlnet_image_encoder=controlnet_image_encoder,
        controlnet_prompt_seq_projection=TRAIN_OBJ_MASKING,
        tokenizer=tokenizer,
        unet=unet_clone,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
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

    image_logs = []
    for prompt, neg_prompt, image, mask, conditioning, control_prompt, focus_prompt in zip(prompts, neg_prompts, images, masks, conditioning, control_prompts, focus_prompts):
        image = Image.open(image).convert("RGB") if image is not None else None
        mask_conditioning = Image.open(mask).convert("RGB") if mask is not None else None
        mask = Image.open(mask).convert("L") if mask is not None else None
        conditioning = Image.open(conditioning).convert("RGB") if conditioning is not None else None
        focus_prompt = [focus_prompt] if focus_prompt else None

        images = []
        for i in range(args.num_validation_images):
            with torch.autocast(f"cuda"):
                pred_image = pipeline(prompt=prompt, controlnet_prompt=control_prompt, negative_prompt=neg_prompt, focus_prompt=focus_prompt, 
                                      image=image, mask_image=mask, conditioning_image=mask_conditioning, height=512, width=512, 
                                      strength=1.0, controlnet_conditioning_scale=0.9, num_inference_steps=50, guidance_scale=7.5, guess_mode=i>1, generator=generator).images[0]
            images.append(pred_image)

        image_logs.append({"reference": image, "images": images, "prompt": ", ".join([str(prompt), str(control_prompt)])})

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

            for log  in image_logs:
                images = log["images"]
                prompt = log["prompt"]
                image = log["reference"]

                formatted_images.append(wandb.Image(image, caption="Reference"))
                for image in images:
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
        self.train_dataset = DiffuserDataset(args, self.tokenizer)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)

        # import correct text encoder class
        text_encoder_cls = import_text_encoder_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        # Custom encoder since sd2 doesn't have the weights for the visual counterpart necessary for obj_masking controlnet 
        self.controlnet_text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.controlnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet")
            net_for_w = copy.deepcopy(self.unet)
            if self.unet.config.in_channels != 4:  # NOTE: hardcoded init with sd2.0 weights
                net_for_w.conv_in = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="unet",revision=args.revision).conv_in
                net_for_w.config.in_channels = 4
            self.controlnet = ControlNetModel.from_unet(net_for_w)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_main_process:
                    i = len(weights) - 1

                    while len(weights) > 0:
                        weights.pop()
                        model = models[i]

                        sub_dir = "controlnet"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                        i -= 1

            def load_model_hook(models, input_dir):
                while len(models) > 0:
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            def enable_model_cpu_offload():
                """
                Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
                to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
                method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
                `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
                """
                if version.parse(accelerate.__version__) >= version.parse("0.17.0.dev0"):
                    from accelerate import cpu_offload_with_hook
                else:
                    raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

                device = torch.device("cuda")

                hook = None
                for cpu_offloaded_model in [self.text_encoder, self.image_encoder, self.controlnet_text_encoder, self.unet, self.vae]:
                    _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prv_module_hook=hook)

                if self.safety_checker is not None:
                    # the safety checker can offload the vae again
                    _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

                # control net hook has be manually offloaded as it alternates with unet
                cpu_offload_with_hook(self.controlnet, device)

                # We'll offload the last model manually.
                self.final_offload_hook = hook

            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)
            if args.enable_cpu_offload: enable_model_cpu_offload()

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.controlnet_image_encoder.requires_grad_(False)
        self.controlnet_text_encoder.requires_grad_(False)
        self.controlnet.train()

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
                self.controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if args.gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()

        # Check that all trainable models are in full precision
        if self.accelerator.unwrap_model(self.controlnet).dtype != torch.float32:
            low_precision_error_string = (
                " Please make sure to always have all model weights in full float32 precision when starting training - even if"
                " doing mixed precision training, copy of the weights should still be float32."
            )
            raise ValueError(
                f"Controlnet loaded as datatype {self.accelerator.unwrap_model(self.controlnet).dtype}. {low_precision_error_string}"
            )

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

        # pass controlnet and (optionally) sd parameters to optimizer
        params = list(self.controlnet.parameters())
        if args.sd_unlock > 0:
            params += self.unet.up_blocks.parameters()
            params += self.unet.conv_norm_out.parameters()

        self.optimizer = optimizer_class(
            params,
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

        # Prepare everything with accelerator library
        self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.controlnet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
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
                # if args.sd_unlock > 0:
                #     a = load_file(os.path.join(args.output_dir, path, "controlnet", "diffusion_pytorch_model.safetensors"))
                #     a.update({f"up_blocks.{k}":v for k,v in self.unet.up_blocks.state_dict().items()})
                #     a.update({f"conv_norm_out.{k}":v for k,v in self.unet.conv_norm_out.state_dict().items()})
                #     shutil.copytree(os.path.join(args.output_dir, path), os.path.join(args.output_dir, path + "_sd"), dirs_exist_ok=True)
                #     path = path + "_sd"
                #     save_file(a, os.path.join(args.output_dir, path, "controlnet", "diffusion_pytorch_model.safetensors"))
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(args.output_dir, path))
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

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=self.initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        training_nets = [self.controlnet]
        for epoch in range(self.first_epoch, args.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                if args.sd_unlock > 0 and step >= args.sd_unlock:
                    args.sd_unlock = -1
                    self.unet.up_blocks.requires_grad_(True)
                    self.unet.conv_norm_out.requires_grad_(True)
                    training_nets = [self.controlnet, self.unet]

                with (self.accelerator.accumulate(*training_nets)):
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

                    # if mask is not None and self.unet.config.in_channels == 4:
                    #     noise[torch.nn.functional.interpolate(mask, size=noise.shape[-2:]).expand(noise.shape)==0]=0

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents_model_input = noisy_latents
                    timesteps_model_input = timesteps

                    # Mask and latents from masked_image
                    if mask is not None:
                        masked_image = image.clone()
                        masked_image[mask.expand(image.shape) > 0.5] = 0  # 0 in [-1,1] image because this is what sd wants
                        mask, masked_image_latents = self.pipe_utils.prepare_mask_latents(mask, masked_image, bs*1, h, w, self.weight_dtype, self.accelerator.device, generator=None, do_classifier_free_guidance=False)

                    # Control Image
                    conditioning_image = self.pipe_utils.prepare_conditioning_image(conditioning_image, bs*1, 1, self.accelerator.device, self.weight_dtype, do_classifier_free_guidance=False)

                    # Prompt to embeddings
                    assert (not TRAIN_OBJ_MASKING and batch["ctrl_txt_image"][0]=="") or (TRAIN_OBJ_MASKING and batch["ctrl_txt_image"][0]!=""), "ERR: TRAIN_OBJ_MASKING is False/True but `ctrl_txt_image` is/is not provided."

                    flag_image_as_hid_prompt = (np.random.random() > 0.5) if TRAIN_OBJ_MASKING else False

                    encoder, ctrl_prompt = (self.controlnet_image_encoder, [os.path.join(args.train_data_dir, p) for p in batch["ctrl_txt_image"]]) \
                        if flag_image_as_hid_prompt else (self.controlnet_text_encoder, batch["ctrl_txt"])
                    encoder_hidden_states_ctrl = self.pipe_utils.encode_prompt(ctrl_prompt, self.accelerator.device, False, encoder, num_images_per_prompt=1, negative_prompt=batch["no_txt"], return_tuple=False)
                    encoder_hidden_states = self.pipe_utils.encode_prompt(batch["txt"], self.accelerator.device, False, self.text_encoder, num_images_per_prompt=1, negative_prompt=batch["no_txt"], return_tuple=False)

                    if TRAIN_OBJ_MASKING:
                        # apply projection layer to all sequence tokens to match size between the two encoders
                        encoder_hidden_states_ctrl = self.controlnet_image_encoder.visual_projection(encoder_hidden_states_ctrl) \
                            if flag_image_as_hid_prompt else self.controlnet_text_encoder.text_projection(encoder_hidden_states_ctrl)

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        noisy_latents_model_input, timesteps_model_input,
                        encoder_hidden_states=encoder_hidden_states_ctrl,
                        controlnet_cond=conditioning_image,
                        return_dict=False,
                    )

                    if mask is not None and self.unet.config.in_channels == 9:
                        if TRAIN_OBJ_MASKING:
                            mid_block_res_sample = mask_block(mask, mid_block_res_sample)
                            down_block_res_samples = [mask_block(mask, block) for block in down_block_res_samples]
                        noisy_latents_model_input = torch.cat([noisy_latents_model_input, mask, masked_image_latents], dim=1)

                    # Predict the noise residual
                    noise_pred = self.unet(
                        noisy_latents_model_input, timesteps_model_input,
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
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = self.controlnet.parameters()
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
                            log_validation(self.vae, self.text_encoder, self.controlnet_text_encoder, self.controlnet_image_encoder, self.tokenizer, self.unet, self.controlnet, self.args, self.accelerator, self.weight_dtype, self.global_step)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= args.max_train_steps:
                    break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            controlnet = self.accelerator.unwrap_model(self.controlnet)
            controlnet.save_pretrained(args.output_dir)

        self.accelerator.end_training()


if __name__ == "__main__":
    args = Train_args().parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    Trainer(args).train()
