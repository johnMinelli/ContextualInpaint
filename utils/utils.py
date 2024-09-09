import os

import numpy as np
import torch
from PIL.Image import Image
from diffusers.pipelines.controlnet import MultiControlNetModel
from scipy.ndimage import binary_erosion
from transformers import PretrainedConfig
import torch.nn.functional as F


def raiser(): raise ValueError(
    "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`")

def replicate(x, n):
    return x if len(x) == n else x * n if len(x) == 1 else [None] * n if len(x) == 0 else raiser()

def mask_block(binary_mask, x):
    return F.interpolate(binary_mask.float(), size=x.shape[-2:], mode='nearest').bool().expand_as(x) * x

def multi_controlnet_helper(controlnet, encoder_hidden_states, conditioning_image, focus_mask, conditioning_scale, guess_mode, batch_size, **args):
    down_block_res_samples, mid_block_res_sample = None, None
    if isinstance(controlnet, MultiControlNetModel):
        controlnet = controlnet.nets
        # split the first dim flattened [guess_mode*batch*prompts*num_images] to [prompts,guess_mode*batch*num_images]  
        _, seq, hid = encoder_hidden_states.shape
        _, c, h, w = conditioning_image.shape
        cfg_chunks = 2 - int(guess_mode)
        encoder_hidden_states = torch.stack(torch.stack(encoder_hidden_states.chunk(cfg_chunks), 0).chunk(batch_size, 1), 1).view(batch_size*cfg_chunks, len(controlnet), seq,hid).permute(1,0,2,3)
        conditioning_image = torch.stack(torch.stack(conditioning_image.chunk(cfg_chunks), 0).chunk(batch_size, 1), 1).view(batch_size*cfg_chunks, len(controlnet), c,h,w).permute(1,0,2,3,4)
    else:
        controlnet = [controlnet]
        encoder_hidden_states = [encoder_hidden_states]
        conditioning_image = [conditioning_image]
        conditioning_scale = [conditioning_scale]

    for i, (net, enc_hid_state, image, scale) in enumerate(zip(controlnet, encoder_hidden_states, conditioning_image, conditioning_scale)):

        # Apply attention focused mask. It is possible that the conditioning image becomes all black
        if focus_mask is not None:
            _focus_mask = F.interpolate(focus_mask.to(torch.float32), image.shape[-2:]).expand(*image.shape)
            image = torch.logical_and(_focus_mask, image>0.5).to(torch.float32)

        # Remove class conditioning if provided but not supported
        controlnet_args = args.copy()
        if net.class_embedding is None and controlnet_args.get("class_labels", None) is not None:
            del controlnet_args["class_labels"]

        down_samples, mid_sample = net(encoder_hidden_states=enc_hid_state,
                                        controlnet_cond=image,
                                        conditioning_scale=scale,
                                        guess_mode=guess_mode, **controlnet_args)

        # # Apply attention focused mask
        # if focus_mask is not None:
        #     # match the size and apply mask
        #     mid_sample = mask_block(image[:,:1], mid_sample)
        #     down_samples = [mask_block(image[:,:1], block) for block in down_samples]

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


def bilinear_interpolate(matrix, coords):
    # Get dimensions
    batch_size, height, width = matrix.shape
    n_coords = coords.shape[1]

    # Reshape coords for broadcasting
    coords = coords.unsqueeze(2)  # shape: (batch_size, n_coords, 1, 2)

    # Compute the coordinates of the four points around the given coordinates
    floor_coords = torch.floor(coords).long()
    ceil_coords = torch.ceil(coords).long()

    # Clamp coordinates to be within the matrix range
    floor_coords = torch.clamp(floor_coords, 0, height - 1)
    ceil_coords = torch.clamp(ceil_coords, 0, height - 1)

    # Extract the four neighboring points' values
    top_left = torch.stack([matrix[i, floor_coords[i, :, :, 0], floor_coords[i, :, :, 1]] for i in torch.arange(batch_size)])
    top_right = torch.stack([matrix[i, floor_coords[i, :, :, 0], ceil_coords[i, :, :, 1]] for i in torch.arange(batch_size)])
    bottom_left = torch.stack([matrix[i, ceil_coords[i, :, :, 0], floor_coords[i, :, :, 1]] for i in torch.arange(batch_size)])
    bottom_right = torch.stack([matrix[i, ceil_coords[i, :, :, 0], ceil_coords[i, :, :, 1]] for i in torch.arange(batch_size)])

    # Calculate the weights for interpolation
    x_weights = coords[..., 0] - floor_coords[..., 0].float()
    y_weights = coords[..., 1] - floor_coords[..., 1].float()

    # Perform bilinear interpolation
    top_interpolation = top_left * (1 - x_weights) + top_right * x_weights
    bottom_interpolation = bottom_left * (1 - x_weights) + bottom_right * x_weights
    interpolated_values = top_interpolation * (1 - y_weights) + bottom_interpolation * y_weights

    return interpolated_values.squeeze()


def get_interpolated_patch(matrix, coord, p_size=2):
    b, c, h, w = matrix.shape
    batched_patch_coords = []
    for i in range(b):
        x, y = coord[i, 0].item(), coord[i, 1].item()
        patch_coords = torch.tensor([[y + dy, x + dx] for dx in range(-p_size, p_size + 1) for dy in range(-p_size, p_size + 1)]).to(matrix.device)
        patch_coords[:, 0] = torch.clamp(patch_coords[:, 0], 0, h - 1)
        patch_coords[:, 1] = torch.clamp(patch_coords[:, 1], 0, w - 1)
        batched_patch_coords.append(patch_coords)

    return bilinear_interpolate(matrix.view(b*c,h,w), torch.stack(batched_patch_coords).repeat_interleave(c, 0)).view(b, c, p_size*2+1, p_size*2+1)


def compute_centroid(matrix):
    # Get the batch size, height, and width of the attention map
    b, h, w = matrix.size()

    # Create the coordinate grid
    x_coords = torch.arange(0, w, dtype=torch.float32, device=matrix.device)
    y_coords = torch.arange(0, h, dtype=torch.float32, device=matrix.device)
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

    # Reshape tensors for computation
    map_values_flat = matrix.view(b, -1)
    x_grid_flat = x_grid.reshape(-1)
    y_grid_flat = y_grid.reshape(-1)

    # Compute the weighted sum of x and y coordinates
    weighted_sum_x = torch.sum(map_values_flat * x_grid_flat.unsqueeze(0), dim=1)
    weighted_sum_y = torch.sum(map_values_flat * y_grid_flat.unsqueeze(0), dim=1)

    # Compute the total sum of map values
    total_sum = torch.sum(map_values_flat, dim=1)+torch.finfo(torch.float32).eps

    # Compute the centroid coordinates
    centroid_x = (weighted_sum_y / total_sum)
    centroid_y = (weighted_sum_x / total_sum)

    # Stack centroid coordinates along the last dimension
    centroid_coords = torch.stack([centroid_x, centroid_y], dim=1)

    return centroid_coords


def select_random_point_on_border(mask, padding=50):
    batch_size, height, width = mask.shape

    # Create a new mask with padding near the margins
    padded_mask = torch.zeros_like(mask)
    padded_mask[:, padding:height-padding, padding:width-padding] = 1

    # Apply element-wise AND operation between the original mask and the padded mask
    masked = torch.logical_and(mask, padded_mask)

    # Apply erosion operation to reduce the active areas
    # eroded = F.erosion2d(masked.float().unsqueeze(1), torch.ones(1, 1, padding * 2 + 1, padding * 2 + 1)).squeeze(1)

    # Get the border points
    border_points = [torch.nonzero(b) for b in masked]

    # Randomly select a point from the border points
    selected_points = torch.cat([border[torch.randint(0, border.size(0), (1,))] for border in border_points])
    
    return selected_points


