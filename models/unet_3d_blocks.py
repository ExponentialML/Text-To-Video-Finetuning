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

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, TemporalConvLayer, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.transformer_temporal import TransformerTemporalModel

class DoDBlock(nn.Module):
    """
    A downconvolution layer with masked video latents
    Gets the masked video latents (the first and the last frame) and makes a masked convolution
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: Always 3D, downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 dims=3,
                 depth=0,
                 out_channels=None,
                 padding=1,
                 is_up=False,):
        super().__init__()
        self.channels = channels
        self.out_channels = min((out_channels or channels) * (2 ** depth), 1280) if not is_up else (out_channels or channels)
        self.dims = dims
        self.is_up = is_up
        stride = 2**depth if dims != 3 else (1, 2**depth, 2**depth) # if depth is zero, the stride is 1

        # Convolution block, which should be initialized with zero weights and biases
        # (zero conv)
        self.conv_w = nn.Conv2d(
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        self.conv_b = nn.Conv2d(
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        # Conv for masking
        self.mask_conv_w = nn.Conv2d(
            1, # only black and white
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        self.mask_conv_b = nn.Conv2d(
            1, # only black and white
            self.out_channels,
            3,
            stride=stride,
            padding=padding)

    # h - hidden states, x_c - frame conditioning, x_m - masked video latents
    def forward(self, h, x_c=None, x_m=None):

        # When no frame conditioning is provided (top DoD iteration)
        # return the untouched hidden states
        if x_c is None or x_m is None:
            return h

        # Add image conditioning as linear operation
        
        #print('IS UP ', self.is_up)
        #print('h', h.shape)
        #print('xc', x_c.shape)
        #print('xm', x_m.shape)
        
        #print('cw', self.conv_w.weight.shape)

        # get weights and biases from frame conditioning
        # vid convolution (initialized with zero weights and biases at first)
        x_c_w = self.conv_w(x_c)
        x_c_b = self.conv_b(x_c)
        
        #print('xcw', x_c_w.shape)
        #print('xcb', x_c_b.shape)
        
        h = x_c_w * h + x_c_b + h # uses hadamard product

        # Use masked video latents to mask the convolution
        x_m_w = self.mask_conv_w(x_m)
        x_m_b = self.mask_conv_b(x_m)

        h = x_m_w * h + x_m_b + h # uses hadamard product
        
        return h
    
    def _init_weights(self):
        # Zero initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Assign gradient checkpoint function to simple variable for readability.
g_c = checkpoint.checkpoint

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
            inputs = module(hidden_states, num_frames=num_frames)
            return inputs

    return custom_forward

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
        ).sample

        # Temporal Self and CrossAttention
        if idx == 1: return g_c(custom_checkpoint(temp_attn, mode='temp'), 
            hidden_states, num_frames, use_reentrant=False).sample

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
    hidden_states = g_c(custom_checkpoint(resnet, mode='resnet'), hidden_states, temb)
    hidden_states = g_c(custom_checkpoint(temp_conv, mode='temp'), 
        hidden_states, num_frames
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
    infinet=None,
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
            infinet=infinet,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
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
            infinet=infinet,
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
    infinet=None,
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
            infinet=infinet,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
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
            infinet=infinet,
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
            )
        ]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    in_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
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
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
                hidden_states = resnet(hidden_states, temb)
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
        infinet=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
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
                )
            )

            if infinet is not None:
                infinet.input_blocks_injections.append(DoDBlock(
                        infinet.in_channels,
                        2,
                        len(infinet.input_blocks_injections),
                        out_channels,
                        is_up=False,
                    )
                )

            attentions.append(
                Transformer2DModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
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
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

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
        dod_block=None,
        x_c=None,
        x_m=None,
    ):
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
        
            if self.gradient_checkpointing:
                # TODO: Infinet is not implemented here yet!
                # so don't use it for now with gradient checkpointing on
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
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

                if dod_block is not None:
                    hidden_states = dod_block(hidden_states, x_c, x_m)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

            output_states += (hidden_states,)

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
        infinet=None,
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
                )
            )

            if infinet is not None:
                infinet.input_blocks_injections.append(DoDBlock(
                        infinet.in_channels,
                        2, # dims
                        len(infinet.input_blocks_injections),
                        out_channels,
                        is_up=False,
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

    def forward(self, hidden_states, temb=None, num_frames=1, dod_block=None, x_c=None, x_m=None,):
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            
            if dod_block is not None:
                hidden_states = dod_block(hidden_states, x_c, x_m)

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
        infinet=None,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

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
                )
            )

            if infinet is not None:
                #print(len(infinet.input_blocks_injections))
                #print(len(infinet.output_blocks_injections))
                infinet.output_blocks_injections.append(DoDBlock(
                        infinet.in_channels,
                        2,
                        max(0, 3 - len(infinet.output_blocks_injections)),#max(0, len(infinet.input_blocks_injections) - len(infinet.output_blocks_injections)),
                        (out_channels // 2**(len(infinet.output_blocks_injections)-1)) if len(infinet.output_blocks_injections) > 1 else out_channels,
                        is_up=True,
                    )
                )

            attentions.append(
                Transformer2DModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
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
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

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
        dod_block=None,
        x_c=None,
        x_m=None,
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
                # TODO: Infinet is not implemented here yet!
                # so don't use it with gradient checkpointing
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
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)

                if dod_block is not None:
                    hidden_states = dod_block(hidden_states, x_c, x_m)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

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
        infinet=None,
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        self.gradient_checkpointing=False

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
                )
            )

            if infinet is not None:
                #print(len(infinet.input_blocks_injections))
                #print(len(infinet.output_blocks_injections))
                infinet.output_blocks_injections.append(DoDBlock(
                        infinet.in_channels,
                        2,
                        max(0, 3 - len(infinet.output_blocks_injections)),#max(0, len(infinet.input_blocks_injections) - len(infinet.output_blocks_injections)),
                        (out_channels // 2**(len(infinet.output_blocks_injections)-1)) if len(infinet.output_blocks_injections) > 1 else out_channels,
                        is_up=True,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=1, dod_block=None, x_c=None, x_m=None,):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.gradient_checkpointing:
                hidden_states = up_down_g_c(resnet, temp_conv, hidden_states, temb, num_frames)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            
            if dod_block is not None:
                hidden_states = dod_block(hidden_states, x_c, x_m)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
