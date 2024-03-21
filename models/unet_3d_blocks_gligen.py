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
import os

import torch
from torchvision.ops import roi_align
import torch.utils.checkpoint as checkpoint
from torch import nn
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, TemporalConvLayer, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel
from .transformer_temporal import TransformerTemporalModel

dec_only = os.environ.get("dec_only") == 'true'
enc_only = os.environ.get("enc_only") == 'true'
print(f'dec_only? {dec_only}')
print(f'enc_only? {enc_only}')
assert not (dec_only and enc_only)
# Assign gradient checkpoint function to simple variable for readability.
g_c = checkpoint.checkpoint

def roi_align_from_boxes(features, normalized_rois, masks, output_size, num_frame):
    """
    Args:
    - features (tensor): shape [bs, num_frame, c, h, w]
    - normalized_rois (tensor): shape [bs, num_frame*20, 4], values are normalized between 0 and 1
    - masks (tensor): shape [bs, num_frame*20], with 1 for valid bboxes and 0 for padded bboxes
    - output_size (tuple): The spatial dimensions of the output

    Returns:
    - output (tensor): shape [total_valid_rois, c, output_size[0], output_size[1]]
    """
    bn, c, h, w = features.shape
    bs = bn // num_frame
    
    # Convert normalized rois to actual pixel values
    rois = torch.zeros_like(normalized_rois)
    rois[..., 0] = normalized_rois[..., 0] * w
    rois[..., 1] = normalized_rois[..., 1] * h
    rois[..., 2] = normalized_rois[..., 2] * w
    rois[..., 3] = normalized_rois[..., 3] * h
    
    # Reshape features tensor: [bs*num_frame, c, h, w]
    features = features.view(bs*num_frame, c, h, w)
    
    # Create a tensor of indices for each frame: [bs, num_frame*20]
    frame_indices = (torch.arange(num_frame*20) // 20).view(1, -1).float().to(rois.device).repeat(bs, 1)
    addition_tensor = torch.arange(bs).view(-1, 1) * num_frame
    frame_indices += addition_tensor.to(frame_indices)
    
    # Flatten masks, rois, batch indices, and frame indices
    flat_masks = masks.view(-1)
    flat_rois = rois.view(bs*num_frame*20, 4)
    flat_frame_indices = frame_indices.view(-1, 1)
    
    # Use masks to filter out valid rois, batch indices, and frame indices
    valid_rois = flat_rois[flat_masks == 1]
    valid_frame_indices = flat_frame_indices[flat_masks == 1].long().squeeze(-1)
    
    # query_masks = torch.zeros(bn, h, w)
    # ins_ids = torch.nonzero(flat_masks).squeeze()
    # ins_ids = ins_ids % 20
    
    # for idx, bbox in enumerate(valid_rois):
    #     assert bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
    #     x1, y1, x2, y2 = (int(coord) for coord in bbox.tolist())
    #     try:
    #         query_masks[valid_frame_indices[idx], y1:y2, x1:x2] = ins_ids[idx].item() + 1
    #     except:
    #         print(valid_frame_indices, ins_ids)
    
    # query_masks = query_masks.flatten()
    
    # token_num = output_size[0] * output_size[1]
    # roi_masks = torch.zeros(valid_rois.shape[0] * token_num)
    # for idx in range(valid_rois.shape[0]):
    #     roi_masks[idx * token_num: (idx + 1) * token_num] = ins_ids[idx] + 1
    # roi_masks = (ins_ids + 1).unsqueeze(-1).repeat(1, token_num).flatten()
    
    # final_attn_mask = torch.full((bn*h*w, roi_masks.shape[0]), -10000.0, dtype=torch.float)
    
    # for val in ins_ids.unique():
    #     val = val.cpu().item() + 1
    #     query_masks_indices = (query_masks == val).nonzero().squeeze(-1)
    #     roi_indices = (roi_masks == val).nonzero().squeeze(-1)
        
    #     final_attn_mask[query_masks_indices[:, None], roi_indices] = 0.

    # Combine valid batch indices and rois
    num_rois = valid_rois.shape[0]
    valid_batch_indices = torch.arange(num_rois).view(-1, 1).to(valid_rois.device)
    rois_with_indices = torch.cat((valid_batch_indices, valid_rois), dim=1).to(features)
    
    # Filter features based on frame indices
    valid_features = features[valid_frame_indices]
    
    # ROI Align
    output = roi_align(valid_features, rois_with_indices, output_size)
    
    return output.flatten(2, 3), None#, final_attn_mask.view(bn, -1, roi_masks.shape[0])


def use_temporal(module, num_frames, x):
    if num_frames == 1:
        if isinstance(module, TransformerTemporalModel):
            return {"sample": x}
        else:
            return x

def custom_checkpoint(module, mode=None):
    if mode == None: raise ValueError('Mode for gradient checkpointing cannot be none.')
    custom_forward = None

    if mode == 'resnet':
        def custom_forward(hidden_states, temb):
            inputs = module(hidden_states, temb)
            return inputs

    if mode == 'attn':
        def custom_forward(
            hidden_states, 
            encoder_hidden_states=None, 
            cross_attention_kwargs=None
        ):
            inputs = module(
                hidden_states,
                encoder_hidden_states,
                cross_attention_kwargs
            )
            return inputs

    if mode == 'temp':
         def custom_forward(hidden_states, num_frames=None):
            inputs = use_temporal(module, num_frames, hidden_states)
            if inputs is None: inputs = module(
                hidden_states, 
                num_frames=num_frames
            )
            return inputs

    return custom_forward

def transformer_g_c(transformer, sample, num_frames):
    sample = g_c(custom_checkpoint(transformer, mode='temp'), 
        sample, num_frames, use_reentrant=False
    )['sample']

    return sample

def cross_attn_g_c(
        attn, 
        temp_attn, 
        resnet, 
        temp_conv, 
        hidden_states, 
        encoder_hidden_states, 
        cross_attention_kwargs, 
        temb, 
        num_frames,
        inverse_temp=False
    ):
    
    def ordered_g_c(idx):

        # Self and CrossAttention
        if idx == 0: return g_c(custom_checkpoint(attn, mode='attn'),
            hidden_states, encoder_hidden_states,cross_attention_kwargs, use_reentrant=False
        )['sample']

        # Temporal Self and CrossAttention
        if idx == 1: return g_c(custom_checkpoint(temp_attn, mode='temp'), 
            hidden_states, num_frames, use_reentrant=False)['sample']

        # Resnets
        if idx == 2: return g_c(custom_checkpoint(resnet, mode='resnet'), 
            hidden_states, temb, use_reentrant=False)
        
        # Temporal Convolutions
        if idx == 3: return g_c(custom_checkpoint(temp_conv, mode='temp'), 
            hidden_states, num_frames, use_reentrant=False
    )

    # Here we call the function depending on the order in which they are called. 
    # For some layers, the orders are different, so we access the appropriate one by index.
    
    if not inverse_temp:
        for idx in [0,1,2,3]: hidden_states = ordered_g_c(idx) 
    else:
        for idx in [2,3,0,1]: hidden_states = ordered_g_c(idx)

    return hidden_states

def up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames):
    hidden_states = g_c(custom_checkpoint(resnet, mode='resnet'), hidden_states, temb, use_reentrant=False)
    hidden_states = g_c(custom_checkpoint(temp_conv, mode='temp'), 
        hidden_states, num_frames,  use_reentrant=False
    )
    return hidden_states

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        if os.environ.get("injector_on") == 'enc':
            attention_type = "gated_injector"
        elif os.environ.get("injector_on") == 'enc_fuse':
            if os.environ.get("track_query", 'false') == 'true':
                attention_type = "gated_injector_fuse_q"
            else:
                attention_type = "gated_injector_fuse"
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=True,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        if os.environ.get("injector_on") == 'dec':
            attention_type = "gated_injector"
        elif os.environ.get("injector_on") == 'dec_fuse':
            if os.environ.get("track_query", 'false') == 'true':
                attention_type = "gated_injector_fuse_q"
            else:
                attention_type = "gated_injector_fuse"
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=True,
        upcast_attention=False,
        attention_type='default'
    ):
        super().__init__()

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1
            )
        ]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    8,
                    in_channels // 8,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        if self.gradient_checkpointing:
            hidden_states = up_down_g_c(
                    self.resnets[0], 
                    self.temp_convs[0], 
                    hidden_states, 
                    temb, 
                    num_frames
                )
        else:
            hidden_states = self.resnets[0](hidden_states, temb)
            hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
            
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(
                        attn, 
                        temp_attn, 
                        resnet, 
                        temp_conv, 
                        hidden_states, 
                        encoder_hidden_states, 
                        cross_attention_kwargs, 
                        temb, 
                        num_frames
                    )
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                
                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

                hidden_states = resnet(hidden_states, temb)

                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        tracklet_attentions = []
        temp_convs = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )
            attentions.append(
                Transformer2DModel(
                    8,
                    out_channels // 8,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        tracklet_attentions.append(
            Transformer2DModel(
                8,
                out_channels // 8,
                in_channels=out_channels,
                num_layers=1,
                cross_attention_dim=out_channels,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=True,
                upcast_attention=upcast_attention,
                attention_type="default",
            )
        )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        
        if 'enc_fuse' in os.environ.get("injector_on"):
            self.null_track_feat = nn.Embedding(1, out_channels)
        
        env_var_value = os.environ.get("open_box_attn")
        if env_var_value == 'enable' and not dec_only:
            print('init encoder tracklet attention')
            self.tracklet_attentions = nn.ModuleList(tracklet_attentions)
        else:
            self.tracklet_attentions = None

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions,
        ):
        
            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(
                        attn, 
                        temp_attn, 
                        resnet, 
                        temp_conv, 
                        hidden_states, 
                        encoder_hidden_states, 
                        cross_attention_kwargs, 
                        temb, 
                        num_frames,
                        inverse_temp=True
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)

                # _cross_attention_kwargs = cross_attention_kwargs.copy()
                if 'enc' in os.environ.get("injector_on"):
                    assert True
                    gligen = _cross_attention_kwargs['gligen']
                    out_patch_size = 1
                    tracklet_feat, _ = roi_align_from_boxes(hidden_states, gligen['boxes'], gligen['masks'], (out_patch_size, out_patch_size), num_frames)
                    tracklet_feat = tracklet_feat.transpose(1, 2).flatten(0, 1)
                    _cross_attention_kwargs['gligen']['tracklet_feat'] = tracklet_feat.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)

                    if 'fuse' in os.environ.get("injector_on"):
                        flat_masks = gligen['masks'].flatten()
                        indices = flat_masks.nonzero(as_tuple=False).squeeze()
                        padding_feat = self.null_track_feat.weight.repeat(hidden_states.shape[0] * 20, 1).to(tracklet_feat)
                        padding_feat[indices] = tracklet_feat
                        _cross_attention_kwargs['gligen']['padding_feat'] = padding_feat

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

            output_states += (hidden_states,)
        
        if not dec_only and self.tracklet_attentions is not None:
            for tracklet_attn in self.tracklet_attentions:
                env_var_value = os.environ.get("open_box_attn")
                dynamic_size = os.environ.get("dynamic_size") == 'enable'
                out_patch_size = max(1, hidden_states.shape[-2] // 8) if dynamic_size else 2
                assert env_var_value is not None
                if env_var_value == "enable":
                    gligen = cross_attention_kwargs['gligen']
                    tracklet_feat, final_attn_mask = roi_align_from_boxes(hidden_states, gligen['boxes'], gligen['masks'], (out_patch_size, out_patch_size), num_frames)
                    tracklet_feat = tracklet_feat.transpose(1, 2).flatten(0, 1).unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)
                    hidden_states = tracklet_attn(
                        hidden_states,
                        encoder_hidden_states=tracklet_feat,
                        cross_attention_kwargs=None,
                        attention_mask=final_attn_mask.to(hidden_states),
                    ).sample

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        self.gradient_checkpointing = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, num_frames=1):
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)

                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type='default'
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []
        tracklet_attentions = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )
            attentions.append(
                Transformer2DModel(
                    8,
                    out_channels // 8,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    attention_type=attention_type,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
        tracklet_attentions.append(
            Transformer2DModel(
                8,
                out_channels // 8,
                in_channels=out_channels,
                num_layers=1,
                cross_attention_dim=out_channels,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=True,
                upcast_attention=upcast_attention,
                attention_type="default",
            )
        )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if 'dec_fuse' in os.environ.get("injector_on"):
            self.null_track_feat = nn.Embedding(1, out_channels)
        
        env_var_value = os.environ.get("open_box_attn")
        if env_var_value == 'enable' and not enc_only:
            print('init decoder tracklet attention')
            self.tracklet_attentions = nn.ModuleList(tracklet_attentions)
        else:
            self.tracklet_attentions = None

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.gradient_checkpointing:
                hidden_states = cross_attn_g_c(
                        attn, 
                        temp_attn, 
                        resnet, 
                        temp_conv, 
                        hidden_states, 
                        encoder_hidden_states, 
                        cross_attention_kwargs, 
                        temb, 
                        num_frames,
                        inverse_temp=True
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)

                _cross_attention_kwargs = cross_attention_kwargs.copy()
                if 'dec' in os.environ.get("injector_on"):
                    gligen = _cross_attention_kwargs['gligen']
                    out_patch_size = 1
                    tracklet_feat, _ = roi_align_from_boxes(hidden_states, gligen['boxes'], gligen['masks'], (out_patch_size, out_patch_size), num_frames)
                    tracklet_feat = tracklet_feat.transpose(1, 2).flatten(0, 1)
                    _cross_attention_kwargs['gligen']['tracklet_feat'] = tracklet_feat.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)
                    if 'fuse' in os.environ.get("injector_on"):
                        flat_masks = gligen['masks'].flatten()
                        indices = flat_masks.nonzero(as_tuple=False).squeeze()
                        padding_feat = self.null_track_feat.weight.repeat(hidden_states.shape[0] * 20, 1).to(tracklet_feat)
                        padding_feat[indices] = tracklet_feat
                        _cross_attention_kwargs['gligen']['padding_feat'] = padding_feat

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=_cross_attention_kwargs,
                ).sample

                if num_frames > 1:
                    hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

        if not enc_only and self.tracklet_attentions is not None:
            for tracklet_attn in self.tracklet_attentions:
                env_var_value = os.environ.get("open_box_attn")
                dynamic_size = os.environ.get("dynamic_size") == 'enable'
                assert env_var_value is not None
                if env_var_value == "enable":
                    gligen = cross_attention_kwargs['gligen']
                    out_patch_size = max(1, hidden_states.shape[-2] // 8) if dynamic_size else 2
                    tracklet_feat, final_attn_mask = roi_align_from_boxes(hidden_states, gligen['boxes'], gligen['masks'], (out_patch_size, out_patch_size), num_frames)
                    tracklet_feat = tracklet_feat.transpose(1, 2).flatten(0, 1).unsqueeze(0).repeat(hidden_states.shape[0], 1, 1)
                    hidden_states = tracklet_attn(
                        hidden_states,
                        encoder_hidden_states=tracklet_feat,
                        cross_attention_kwargs=None,
                        attention_mask=final_attn_mask.to(hidden_states),
                    ).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        self.gradient_checkpointing = False
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)

                if num_frames > 1:
                    hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states