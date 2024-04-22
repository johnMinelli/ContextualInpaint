from typing import List
from diffusers.models.attention_processor import AttnProcessor, Attention

import torch
import torch.nn.functional as F
import math


class AttentionStore():
    def __init__(self, store_averaged_over_steps: bool, batch_size=1, classifier_free_guidance: bool = False, device="cuda"):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = store_averaged_over_steps
        self.batch_size = batch_size
        self.cfg = classifier_free_guidance
        self.smoothing = GaussianSmoothing(device, 3, 2)

    def attach_unet(self, unet):
            attn_procs = {}
            for name in unet.attn_processors.keys():
                if name.startswith("mid_block"):
                    place_in_unet = "mid"
                elif name.startswith("up_blocks"):
                    place_in_unet = "up"
                elif name.startswith("down_blocks"):
                    place_in_unet = "down"
                else:
                    continue

                # Replace attention module
                if "attn2" in name:
                    attn_procs[name] = CrossAttnProcessor(attention_store=self, place_in_unet=place_in_unet)
                else:
                    attn_procs[name] = AttnProcessor()

            unet.set_attn_processor(attn_procs)

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, num_heads: int):
        # inbatch_size = 1 (uncond) + 1 (cond) + n (edit prompts)
        # attn.shape = batch_size * (inbatch_size * head_size), seq_len query, seq_len_key
        skip = int(self.cfg)  # skip unconditional

        attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3) # create batch_size dimension
        attn = attn[:, skip * num_heads:].reshape(self.batch_size, -1, num_heads, *attn.shape[-2:])  # create num_heads dimension
        # attn.shape = batch_size, (inbatch_size - skip), num_heads, seq_len query, seq_len_key
        self.forward(attn, is_cross, place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

    def step(self, store_current_step=True):
        if store_current_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        del self.step_store
        self.step_store = self.get_empty_store()

    def get_stored_attention(self, step: int):
        if self.average:
            attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        else:
            assert (step is not None)
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(self, attention_maps, block_positions: List[str], res: int, is_cross: bool, select: int):
        """From stored attention maps, aggregate the ones specified by `from_where` with given resolution `res`
            across the different attention heads
        :param attention_maps: Attention maps stored
        :param res: Resolution of the block to consider
        :param block_positions: List of possible blocks from which the attention has been captured [`up`, `down`, `mid`]
        :param is_cross: Weather the aggregation should be over `cross` or `self` attention maps
        :param select: Prompt index for which aggregate the attention maps 
        :return: 
        """
        out = [[] for x in range(self.batch_size)]
        num_pixels = res ** 2
        for location in block_positions:
            for att_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                # cycle over the batch dimension  (b*n,p,head_dim,h*w,hid)
                for batch, item in enumerate(att_item):
                    if item.shape[2] == num_pixels:
                        cross_maps = item.reshape(*item.shape[:2], res, res, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        # cat the attention maps of attention heads relative to same selected prompt stored by different attention modules and stack for the batch_size
        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out
    
    def get_cross_attention_mask(self, block_positions, res, stored_attention_index, token_positions, attention_mask_threshold):
        out = self.aggregate_attention(attention_maps=self.step_store, res=res, block_positions=block_positions, is_cross=True, select=stored_attention_index)
        # average over all tokens (do it batch-wise) # 0 -> startoftext
        attn_map = torch.stack([out[b_nim, :, :, mask].mean(2) for b_nim, mask in enumerate(token_positions)], 0)
        # gaussian_smoothing
        smooth_attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
        smooth_attn_map = self.smoothing(smooth_attn_map).squeeze(1)
        # create binary mask
        tmp = torch.quantile(smooth_attn_map.flatten(start_dim=1).to(torch.float32), attention_mask_threshold, dim=1).to(smooth_attn_map.dtype)
        # attn_mask = torch.where(attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, *attn_map.shape[-2:]), 1.0,0.0).unsqueeze(1)
        attn_mask = torch.round(torch.sigmoid(10.0 * (smooth_attn_map - tmp.unsqueeze(1).unsqueeze(1).repeat(1, *smooth_attn_map.shape[-2:]))))
        
        return attn_mask, attn_map


class CrossAttnProcessor:

    def __init__(self, attention_store, place_in_unet):
        self.attn_store = attention_store
        self.place_in_unet = place_in_unet

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        assert (not attn.residual_connection)
        assert (attn.spatial_norm is None)
        assert (attn.group_norm is None)
        assert (hidden_states.ndim != 4)
        assert (encoder_hidden_states is not None)  # is cross

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attn_store(attention_probs, is_cross=True, place_in_unet=self.place_in_unet, num_heads=attn.heads)
        hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing
class GaussianSmoothing:

    def __init__(self, device, kernel_size=3, sigma=0.5):
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32, device=device) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))