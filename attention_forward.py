import math
from typing import Optional, Iterable

from diffusers.utils import USE_PEFT_BACKEND
from einops import rearrange
import torch


def new_forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None, temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0, **cross_attention_kwargs):
    # original forward function for batch_size=1 or not crossattention role
    if (hidden_states.shape[0] < 2) or encoder_hidden_states is None:
        del cross_attention_kwargs["t"]
        del cross_attention_kwargs["stage"]
        return self.ori_forward(hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)
    else:
        residual = hidden_states

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        t = cross_attention_kwargs.get("t", 1000)
        cfg = cross_attention_kwargs.get("cfg", True)

        num_frames = hidden_states.shape[0] // 2 if cfg else hidden_states.shape[0]
        frames_index = torch.arange(num_frames).long()

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = self.to_q(hidden_states, *args)
        key = self.to_k(encoder_hidden_states, *args)
        value = self.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = torch.nn.functional.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # Note that the attention mask is not split... it should be
        aka = cross_attention_kwargs["stage"].pop()
        if t > self.cfg["t_align"] and num_frames>1 and aka:
            # Split everything in ref and side
            query_split = rearrange(query, "(b f) h d c -> b f h d c", f=num_frames)
            query_ref = rearrange(query_split[:, :1], "b f h d c -> (b f) h d c")
            query_side = rearrange(query_split[:, 1:], "b f h d c -> (b f) h d c")
            key_split = rearrange(key, "(b f) h d c -> b f h d c", f=num_frames)
            key_ref = rearrange(key_split[:, :1], "b f h d c -> (b f) h d c")
            key_side = rearrange(key_split[:, 1:], "b f h d c -> (b f) h d c")
            value_split = rearrange(value, "(b f) h d c -> b f h d c", f=num_frames)
            value_ref = rearrange(value_split[:, :1], "b f h d c -> (b f) h d c")
            value_side = rearrange(value_split[:, 1:], "b f h d c -> (b f) h d c")

            # Compute cross hiddens (backprop branch) for side
            _, hidden_states_side = scaled_dot_product_attention(query_side, key_side, value_side, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            # Reassemble the key_side to be ref prompt and get attention matrix between query_side and key_side
            key = rearrange(key, "(b f) h d c -> b f h d c", f=num_frames)
            key_side = torch.cat([key[:, [0] * int(num_frames-1)]], dim=2)
            key_side = rearrange(key_side, "b f h d c -> (b f) h d c")
            value_side = key_side.clone()  # it doesn't matter

            # Compute attention matrix of sides to manin prompt, sum and detach
            att_mat = scaled_dot_product_attention(query_side, key_side, value_side, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)[0]
            att_mat = rearrange(att_mat, "(b f) h d c -> b f h d c", f=num_frames-1).sum(1)  # .detach()
            # Add side attention to ref dot product's softmax
            _, hidden_states_ref = scaled_dot_product_attention(query_ref, key_ref, value_ref, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, add_att=att_mat)

            hidden_states = torch.cat([rearrange(hidden_states_ref, "(b f) h d c -> b f h d c", f=1), rearrange(hidden_states_side, "(b f) h d c -> b f h d c", f=num_frames - 1)], dim=1)
            hidden_states = rearrange(hidden_states, "b f h d c -> (b f) h d c")
        else:
            _, hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)


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


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 add_att=None) -> Iterable[torch.Tensor]:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    attn_weight_hot = torch.softmax(add_att if add_att is not None else attn_weight, dim=-1)
    attn_weight_hot = torch.dropout(attn_weight_hot, dropout_p, train=True)

    return attn_weight_hot, attn_weight_hot @ value
