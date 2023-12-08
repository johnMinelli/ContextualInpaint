from typing import Optional

from diffusers.utils import USE_PEFT_BACKEND
from einops import rearrange
import torch

def new_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    temb: Optional[torch.FloatTensor] = None,
    scale: float = 1.0,
    **cross_attention_kwargs
):
    # original forward function for batch_size=1 or crossattention role
    if (hidden_states.shape[0] < 2) or encoder_hidden_states is not None:
        del cross_attention_kwargs["t"]
        return self.ori_forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
    else:
        residual = hidden_states

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        t = cross_attention_kwargs.get("t", 1000)
        cfg = cross_attention_kwargs.get("cfg", True)

        num_frames = hidden_states.shape[0]//2 if cfg else hidden_states.shape[0]
        frames_index = torch.arange(num_frames).long()

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        encoder_hidden_states = hidden_states
        
        query = self.to_q(hidden_states, *args)
        key = self.to_k(encoder_hidden_states, *args)
        value = self.to_v(encoder_hidden_states, *args)

        # Reference attention is given between first frame and all others
        if t > self.cfg["t_align"]:
            key = rearrange(key, "(b f) d c -> b f d c", f=num_frames)
            key = torch.cat([key[:, [0] * int(num_frames)], key[:, frames_index]], dim=2)
            key = rearrange(key, "b f d c -> (b f) d c")
    
            value = rearrange(value, "(b f) d c -> b f d c", f=num_frames)
            value = torch.cat([value[:, [0] * int(num_frames)], value[:, frames_index]], dim=2)
            value = rearrange(value, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:###
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = torch.nn.functional.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
