import os

import torch
from PIL.Image import Image
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import PretrainedConfig
import torch.nn.functional as F


def not_image(filename: str):
    return not filename.endswith(".jpeg") and not filename.endswith(".jpg") and not filename.endswith(".png")

def raiser(): raise ValueError(
    "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`")

def replicate(x, n):
    return x if len(x) == n else x * n if len(x) == 1 else [None] * n if len(x) == 0 else raiser()

def mask_block(binary_mask, x):
    return F.interpolate(binary_mask.float(), size=x.shape[-2:], mode='nearest').bool().expand_as(x) * x

def split_multi_net(controlnet, attention_mask, encoder_hidden_states, controlnet_cond, conditioning_scale, guess_mode, batch_size, **args):
    down_block_res_samples, mid_block_res_sample = None, None
    if isinstance(controlnet, MultiControlNetModel):
        controlnets = controlnet.nets
        # split the first dim flattened [guess_mode*batch*prompts*num_images] to [prompts,guess_mode*batch*num_images]  
        _, seq, hid = encoder_hidden_states.shape
        _, c, h, w = controlnet_cond.shape
        cfg_chunks = 2 - int(guess_mode)
        encoder_hidden_states = torch.stack(torch.stack(encoder_hidden_states.chunk(cfg_chunks), 0).chunk(batch_size, 1), 1).view(batch_size*cfg_chunks, len(controlnets), seq,hid).permute(1,0,2,3)
        controlnet_cond = torch.stack(torch.stack(controlnet_cond.chunk(cfg_chunks), 0).chunk(batch_size, 1), 1).view(batch_size*cfg_chunks, len(controlnets), c,h,w).permute(1,0,2,3,4)
    else:
        controlnets = [controlnet]
        conditioning_scale = [conditioning_scale]
        encoder_hidden_states = [encoder_hidden_states]
        controlnet_cond = [controlnet_cond]

    for i, (controlnet, enc_hid_state, image, scale) in enumerate(zip(controlnets, encoder_hidden_states, controlnet_cond, conditioning_scale)):
        down_samples, mid_sample = controlnet(encoder_hidden_states=enc_hid_state,
                                              controlnet_cond=image,
                                              conditioning_scale=scale,
                                              guess_mode=guess_mode, **args)
        # Apply attention focused mask
        if attention_mask is not None and (len(controlnets) == 1 or i>0):
            # match the size and apply mask
            mid_sample = mask_block(attention_mask, mid_sample)
            down_samples = [mask_block(attention_mask, block) for block in down_samples]
            
        # Merge samples
        if i == 0:
            down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
        else:
            down_block_res_samples = [samples_prev + samples_curr for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)]
            mid_block_res_sample += mid_sample

    return down_block_res_samples, mid_block_res_sample


def import_text_encoder_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            prompt = log["prompt"]
            image = log["image"]
            image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {prompt}\n"
            images = [image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

