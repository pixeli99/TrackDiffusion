# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.embeddings import FourierEmbedder
import os


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    The output of [`TransformerTemporalModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        
        env_var_value = os.environ.get("open_time_embedding")
        if env_var_value == "learnable_frame":
            self.time_embedder = nn.Embedding(16, in_channels)
        if env_var_value == "fourier_frame":
            self.time_fourier_embedder = FourierEmbedder(num_freqs=in_channels // 2)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        num_frames=1,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        env_var_value = os.environ.get("open_time_embedding")
        assert env_var_value is not None

        if env_var_value == "fourier_frame":
            frame_ids = torch.tensor([i for i in range(num_frames)]).float().to(hidden_states)
            normalized_ids = frame_ids / num_frames
            time_fourier_embeddings = self.time_fourier_embedder(normalized_ids.unsqueeze(1))
            hidden_states = hidden_states + time_fourier_embeddings

        elif env_var_value == "learnable_frame":
            hidden_states = hidden_states + self.time_embedder.weight.unsqueeze(0)[:, :num_frames, :]
            
        elif env_var_value == "fourier_pixel":
            frame_ids = torch.arange(num_frames).float().to(hidden_states)
            h_ids = torch.arange(height).float().to(hidden_states)
            w_ids = torch.arange(width).float().to(hidden_states)
            # meshgrid
            frame_ids, h_ids, w_ids = torch.meshgrid(frame_ids, h_ids, w_ids)
            normalized_frame_ids = frame_ids / num_frames
            normalized_h_ids = h_ids / height
            normalized_w_ids = w_ids / width
            
            frame_embedding = self.time_fourier_embedder(normalized_frame_ids.view(-1, 1)).view(num_frames, height, width, -1)
            h_embedding = self.time_fourier_embedder(normalized_h_ids.view(-1, 1)).view(num_frames, height, width, -1)
            w_embedding = self.time_fourier_embedder(normalized_w_ids.view(-1, 1)).view(num_frames, height, width, -1)
            final_embedding = frame_embedding + h_embedding + w_embedding
            
            final_embedding = final_embedding.flatten(1, 2).permute(1, 0, 2)
            hidden_states = hidden_states + final_embedding.repeat(batch_size, 1, 1)

        elif env_var_value == "off":
            pass
        else:
            raise NotImplementedError

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, channel, num_frames)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
