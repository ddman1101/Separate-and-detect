from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from latent_diffusion.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from latent_diffusion.modules.attention import SpatialTransformer
import torch

# dummy replace
def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1).contiguous()  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    """
    Upsampling layer specialized for 2D latent space that upsamples along the time dimension.
    Used for upsampling operations inside UNet2D.
    """
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=(1, 3),  # convolve only along the time dimension
                padding=(0, padding),
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.Identity()

    def forward(self, x):
        # x shape: [B*num_stems, C, 1, T] -> output [B*num_stems, C, 1, T*2]
        assert x.shape[1] == self.channels, f"Expected x.shape[1] == {self.channels}, got {x.shape[1]}"
        assert x.shape[2] == 1, f"Expected height dimension to be 1, got {x.shape[2]}"
        
        # Upsample along the time dimension
        B, C, H, T = x.shape
        x_upsampled = F.interpolate(x, size=(H, T * 2), mode='nearest')
        
        # Apply convolution if requested
        if self.use_conv:
            x_upsampled = self.op(x_upsampled)
        
        return x_upsampled


class Upsample1D(nn.Module):
    """
    Upsampling layer specialized for 1D latent space; upsample only along time.
    Used for upsampling operations inside UNet2D.
    """
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=(1, 3),  # convolve only along the time dimension
                padding=(0, padding),
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.Identity()

    def forward(self, x):
        # x shape: [B*num_stems, C, 1, T] -> output [B*num_stems, C, 1, T*2]
        assert x.shape[1] == self.channels, f"Expected x.shape[1] == {self.channels}, got {x.shape[1]}"
        assert x.shape[2] == 1, f"Expected height dimension to be 1, got {x.shape[2]}"
        
        # Upsample along the time dimension
        B, C, H, T = x.shape
        x_upsampled = F.interpolate(x, size=(H, T * 2), mode='nearest')
        
        # Apply convolution if requested
        if self.use_conv:
            x_upsampled = self.op(x_upsampled)
        
        return x_upsampled


class TransposedUpsample(nn.Module):
    "Learned 2x upsampling without padding"

    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(
            self.channels, self.out_channels, kernel_size=ks, stride=2
        )

    def forward(self, x):
        return self.up(x)


class Downsample1D(nn.Module):
    """
    专门用于4D tensors的downsampling层，在时间维度上进行下采样
    用于处理mixture的downsampling，mixture的channels数量与VAE的latent_channels相同
    """
    def __init__(self, channels, use_conv, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=(1, 3),  # Only convolve in time dimension
                stride=(1, 2),       # Only stride=2 in time dimension
                padding=(0, padding),
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # Only pooling in time dimension

    def forward(self, x):
        # x shape: [B, num_stems, C, T] -> need to process to [B*num_stems, C, T] for 1D operation
        # Note: here channels should be VAE's latent_channels (8), not UNet's model_channels (128)
        assert x.shape[2] == self.channels, f"Expected x.shape[2] == {self.channels}, got {x.shape[2]}"
        
        # Process 4D tensor containing num_stems
        B, num_stems, C, T = x.shape
        
        # Reshape to [B*num_stems, C, T] for 1D operation
        x_reshaped = x.view(B * num_stems, C, T)
        
        # Apply 1D downsampling
        if self.use_conv:
            # Use 2D convolution for downsampling (in time dimension)
            # Need to reshape to [B*num_stems, C, 1, T] for 2D convolution
            x_reshaped_2d = x_reshaped.unsqueeze(2)  # [B*num_stems, C, 1, T]
            x_downsampled_2d = self.op(x_reshaped_2d)  # Use self.op (2D convolution)
            x_downsampled = x_downsampled_2d.squeeze(2)  # [B*num_stems, C, T_downsampled]
        else:
            # Use average pooling for downsampling
            x_downsampled = F.avg_pool1d(x_reshaped, kernel_size=2, stride=2)
        
        # Reshape back to [B, num_stems, C, T_downsampled]
        T_downsampled = x_downsampled.shape[-1]
        x_output = x_downsampled.view(B, num_stems, C, T_downsampled)
        
        return x_output


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Downsample_one_dim(nn.Module):
    """
    A downsampling module that specifically downsamples a specified dimension of a 5D tensor (batch, channel, depth, height, width),
    using either a convolution or average pooling. This class uses utility functions to handle different dimensions.
    
    Parameters:
        channels (int): Number of channels in the input and output (if not using a different output channel count).
        use_conv (bool): Whether to use a convolution for downsampling.
        target_dim (int): The dimension to downsample (2 for depth, 3 for height, 4 for width).
        out_channels (int, optional): Specifies a different number of output channels. Defaults to the same as input channels.
        dims (int): Specifies if the convolution or pooling is 1D, 2D, or 3D.
    """
    def __init__(self, channels, use_conv, dims=3, target_dim=2, out_channels=None):
        super(Downsample_one_dim, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.target_dim = target_dim
        self.dims = dims

        # Create stride and kernel size to downsample only the target dimension
        stride = [1] * dims
        kernel_size = [1] * dims
        stride[target_dim - 2] = 2
        kernel_size[target_dim - 2] = 2

        if use_conv:
            # Use a convolutional layer to downsample
            self.op = conv_nd(dims, self.channels, self.out_channels, kernel_size=tuple(kernel_size), stride=tuple(stride), padding=0)


        else:
            # Use average pooling
            self.op = avg_pool_nd(dims, kernel_size=tuple(kernel_size), stride=tuple(stride))

    def forward(self, x):
        # Ensure the input has the expected number of channels
        assert x.shape[1] == self.channels, f"Expected channel dimension {self.channels}, got {x.shape[1]}"
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # Key correction: ensure emb and h have the same data type
        # If using AMP, emb may be float16, but h is float32
        if emb.dtype != h.dtype:
            emb = emb.to(dtype=h.dtype)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        # Key correction: ensure x and h have the same data type, avoid mismatch in skip_connection data type
        if x.dtype != h.dtype:
            x = x.to(dtype=h.dtype)
        
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).contiguous()
        qkv = self.qkv(self.norm(x)).contiguous()
        h = self.attention(qkv).contiguous()
        h = self.proj_out(h).contiguous()
        return (x + h).reshape(b, c, *spatial).contiguous()


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = (
            qkv.reshape(bs * self.n_heads, ch * 3, length).contiguous().split(ch, dim=1)
        )
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length).contiguous()

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(bs * self.n_heads, ch, length).contiguous(),
        )
        return a.reshape(bs, -1, length).contiguous()

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        extra_film_condition_dim=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        extra_film_use_concat=False,  # If true, concatenate extrafilm condition with time embedding, else addition
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        no_condition=False,
        # ====== Multi-task branch parameters ======
        use_onset_branch=False,  # whether to use onset branch
        use_timbre_branch=False,  # whether to use timbre branch
        onset_branch_channels=64,  # onset branch channels
        timbre_branch_channels=64,  # timbre branch channels
        onset_output_frames=1024,  # onset output frames
        timbre_feature_dim=7,  # timbre feature dimension
        num_stems=5  # stems number
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.extra_film_condition_dim = extra_film_condition_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.extra_film_use_concat = extra_film_use_concat
        time_embed_dim = model_channels * 4
        self.no_condition = no_condition
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        assert not (
            self.num_classes is not None and self.extra_film_condition_dim is not None
        ), "As for the condition of theh UNet model, you can only set using class label or an extra embedding vector (such as from CLAP). You cannot set both num_classes and extra_film_condition_dim."

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.use_extra_film_by_concat = (
            self.extra_film_condition_dim is not None and self.extra_film_use_concat
        )
        self.use_extra_film_by_addition = (
            self.extra_film_condition_dim is not None and not self.extra_film_use_concat
        )

        if self.extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)
            print(
                "+ Use extra condition on UNet channel using Film. Extra condition dimension is %s. "
                % self.extra_film_condition_dim
            )
            if self.use_extra_film_by_concat:
                print("\t By concatenation with time embedding")
            elif self.use_extra_film_by_addition:
                print("\t By addition with time embedding")

        if use_spatial_transformer and (
            self.use_extra_film_by_concat or self.use_extra_film_by_addition or self.no_condition
        ):
            print(
                "+ Spatial transformer will only be used as self-attention. Because you have choose to use film as your global condition."
            )
            spatial_transformer_no_context = True
        else:
            spatial_transformer_no_context = False

        if use_spatial_transformer and not spatial_transformer_no_context:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None and not spatial_transformer_no_context:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.downsample_layers = nn.ModuleList(                 # List to store corresponding Downsample layers
            [
                th.nn.Identity()
                #Downsample(self.model_channels, False, dims=dims) # Add Downsample layer corresponding to each added block
            ]


        )  

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            no_context=spatial_transformer_no_context,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.downsample_layers.append(th.nn.Identity()
                    # Downsample(self.model_channels, False, dims=dims)
                    ) # Add Downsample layer corresponding to each added block
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.downsample_layers.append(Downsample(1, False, dims=dims))  # Add Downsample layer corresponding to each added block
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                no_context=spatial_transformer_no_context,
            ),
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # self.downsample_layers_after_middle_block = Downsample_one_dim(ch, conv_resample, dims=dims, target_dim = 2)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            no_context=spatial_transformer_no_context,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

        self.shape_reported = False
        
        # ====== Multi-task branch initialization ======
        self.use_onset_branch = use_onset_branch
        self.use_timbre_branch = use_timbre_branch
        self.onset_branch_channels = onset_branch_channels
        self.timbre_branch_channels = timbre_branch_channels
        self.onset_output_frames = onset_output_frames
        self.timbre_feature_dim = timbre_feature_dim
        self.num_stems = num_stems
        
        # Onset branch - from latent
        if self.use_timbre_branch:
            self.timbre_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),      # (B,S,C,1,1,1)
                nn.Flatten(start_dim=3),              # (B,S,C)
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, timbre_feature_dim),   # → 7 (timbre feature dimension)
                nn.Tanh()
            )

        if self.use_onset_branch:
            # latent time length: 8*32*2 = 512, then use fc → 1024
            self.onset_reduce = nn.Conv3d(
                in_channels=32, out_channels=32, kernel_size=1
            )
            self.onset_temporal = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
                nn.ReLU(),
                nn.Conv3d(32, 32, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
                nn.ReLU(),
            )
            self.onset_fc = nn.Linear(32 * 32, onset_output_frames)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional. an [N, extra_film_condition_dim] Tensor if film-embed conditional
        :return: an [N x C x ...] Tensor of outputs.
        """
        if not self.shape_reported:
            print("The shape of UNet input is", x.size())
            self.shape_reported = True

        assert (y is not None) == (
            self.num_classes is not None or self.extra_film_condition_dim is not None
        ), "must specify y if and only if the model is class-conditional or film embedding conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if self.use_extra_film_by_addition:
            emb = emb + self.film_emb(y)
        elif self.use_extra_film_by_concat:
            emb = th.cat([emb, self.film_emb(y)], dim=-1)

        h = x.type(self.dtype)
        h, mix = th.chunk(h, chunks=2, dim=2)

        mix_list = []
        mix = mix[:, 0:1, :, :, :]
        mix_list.append(mix)
        for i in range(len(self.downsample_layers)):
            mix = self.downsample_layers[i](mix)
            mix_list.append(mix)

        # for module in self.input_blocks:
        for i, module in enumerate(self.input_blocks):
            mix_i = mix_list[i]
            mix_to_add =  mix_i.repeat(1, h.shape[1], 1, 1, 1)
            h = h + mix_to_add        
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        h_latent = h
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)

            mix_i = mix_list[len(mix_list)-i-1]
            mix_to_add =  mix_i.repeat(1, h.shape[1], 1, 1, 1)
            h = h + mix_to_add
            h = module(h, emb, context)
        h = h.type(x.dtype)
        # ====== Multi-task branch prediction ======
        onset_output = None
        timbre_output = None
        
        if self.use_onset_branch:
            # Predict onset from latent
            # Expect h_latent ~ [B, S, C, H, W] (designed for multi-stem)
            x = h_latent
            try:
                if x.dim() == 5:
                    # [B, S, C, H, W] -> [B, C, S, H, W] to match Conv3d channel order
                    x = x.permute(0, 2, 1, 3, 4)
                elif x.dim() == 4:
                    # [B, C, H, W] -> [B, C, 1, H, W]
                    x = x.unsqueeze(2)
                # Reduce channels + temporal convolution
                x = self.onset_reduce(x)
                x = self.onset_temporal(x)
                # Restore to [B, S, C, H, W]
                x = x.permute(0, 2, 1, 3, 4)
                # Squeeze H dimension, flatten to [B, S, C*W]
                x = x.mean(dim=3)
                x = x.flatten(start_dim=2)
                # Pass through FC to generate [B, S, T_out]
                onset_output = self.onset_fc(x)
            except Exception:
                onset_output = None
        
        if self.use_timbre_branch:
            # Predict timbre from latent
            # h shape: [B, num_stems, latent_channels, latent_height, latent_width]
            print("h shape : ", h_latent.shape)
            h_stem = h_latent[:, :self.num_stems, ...]            # 取前 S 個 stem + 保留 C = 128 !
            print("h_stem shape : ", h_stem.shape)
            # h_stem = h_stem.permute(0, 2, 1, 3, 4)
            print("h_stem shape : ", h_stem.shape)
            timbre_output = self.timbre_predictor(h_stem)
            # timbre_output shape: [B, num_stems, timbre_feature_dim]
        
        # Main output
        if self.predict_codebook_ids:
            main_output = self.id_predictor(h)
        else:
            main_output = self.out(h)
        
        # Return main output and multi-task outputs
        outputs = [main_output]
        if onset_output is not None:
            outputs.append(onset_output)
        if timbre_output is not None:
            outputs.append(timbre_output)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

####################################################
# For Multitask UNet Model - latent, Timbre, Onset #
####################################################

"""
The belowing are the multi-task learning branches.

1. Latent
2. Timbre
3. Onset
"""

class TimbreHeadPenult(nn.Module):
    def __init__(self, in_ch: int, stems: int = 5, timbre_dim: int = 7,
                 hidden: int = 128, stem_emb_dim: int = 16,
                 use_mean_std: bool = False, dropout: float = 0.1,
                 use_tanh: bool = True):
        super().__init__()
        self.stems = stems
        self.use_mean_std = use_mean_std
        self.use_tanh = use_tanh

        self.stem_emb = nn.Embedding(stems, stem_emb_dim)

        feat_in = in_ch * (2 if use_mean_std else 1) + stem_emb_dim
        layers = [
            nn.Linear(feat_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, timbre_dim),
        ]
        if use_tanh:
            layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):                       # h: (B, C, D, H, W)
        B, C, D, H, W = h.shape
        # Global aggregation: mean or mean+std
        g_mean = h.mean(dim=(2, 3, 4))          # (B, C)
        if self.use_mean_std:
            g_std = h.var(dim=(2, 3, 4), unbiased=False).sqrt()  # (B, C)
            g = torch.cat([g_mean, g_std], dim=1)                # (B, 2C)
        else:
            g = g_mean                                            # (B, C)

        # 擴展到每個 stem
        g = g.unsqueeze(1).expand(B, self.stems, g.shape[-1])     # (B, S, C or 2C)
        stem_ids = torch.arange(self.stems, device=h.device)
        e = self.stem_emb(stem_ids).unsqueeze(0).expand(B, -1, -1) # (B, S, E)
        x = torch.cat([g, e], dim=-1)                             # (B, S, C(+C)+E)
        y = self.mlp(x)                                           # (B, S, timbre_dim)
        return y

class TimbreHead(nn.Module):
    def __init__(self, cin_tot=640, stems=5, timbre_dim=7, hid=128):
        super().__init__()
        self.stems, self.cin = stems, cin_tot // stems
        self.fc = nn.Sequential(
            nn.Linear(self.cin, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, timbre_dim), nn.Sigmoid()
        )

    def forward(self, h):
        B, _, T, F, W = h.shape
        h = (h.view(B, self.stems, self.cin, T, F, W)
               .mean(dim=[3,4,5]))              # pool T,F,W → (B,S,Cin)
        return self.fc(h) 

#############
# New trial #
#############

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        norm = (lambda c: nn.GroupNorm(num_groups=min(32, c), num_channels=c)) if use_gn else (lambda c: nn.Identity())
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.n1   = norm(out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.n2   = norm(out_ch)
        self.act2 = nn.SiLU(inplace=True)

        # Kaiming init for 3D convs (SiLU-friendly)
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.up(x)
        x = self.act1(self.n1(self.conv1(x)))
        x = self.act2(self.n2(self.conv2(x)))
        return x


class Stem2DHead(nn.Module):
    """
    Single stem 2D head:
      in  : (B, D, T, W)  (D=4 as conv2d channel)
      out : (B, T)
    Structure: Conv2d×n → (temporal peak 5×1 optional) → depthwise(1×W) squeeze width → 1×1 out logits.
    """
    def __init__(self, d_in=4, w_target=64, mid=32, n_blocks=2, use_gn=True, use_temporal_peak=True):
        super().__init__()
        self.w_target = w_target
        self.use_temporal_peak = use_temporal_peak

        def norm2d(c):
            return nn.GroupNorm(num_groups=min(8, c), num_channels=c) if use_gn else nn.Identity()

        blocks = []
        c_in = d_in
        for _ in range(n_blocks):
            c_out = mid
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=(5,3), padding=(2,1), bias=False),
                norm2d(c_out),
                nn.SiLU(inplace=True),
            ]
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)

        if use_temporal_peak:
            # Only do 5×1 depthwise along time, promote peaks (can learn identity)
            self.temporal_peak2d = nn.Conv2d(mid, mid, kernel_size=(5,1), padding=(2,0),
                                             groups=mid, bias=True)

        # Squeeze W: depthwise across width, equivalent to learnable "weighted average"
        self.squeeze_w = nn.Conv2d(mid, mid, kernel_size=(1, w_target), padding=0,
                                   groups=mid, bias=True)

        # Project to single channel (logits)
        self.proj_out = nn.Conv2d(mid, 1, kernel_size=1, bias=True)

        # Initialize (SiLU-friendly)
        self.apply(self._init_kaiming)

    @staticmethod
    def _init_kaiming(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _logit(p, eps=1e-6):
        p = torch.as_tensor(p).clamp(eps, 1 - eps)
        return torch.log(p / (1.0 - p))

    def init_prior(self, p_prior=None):
        """Initialize: temporal_peak≈identity; squeeze_w≈average along W; proj_out.bias=logit(prior)."""
        with torch.no_grad():
            # Temporal peak: center=1, others=0
            if self.use_temporal_peak:
                self.temporal_peak2d.weight.zero_()
                self.temporal_peak2d.bias.zero_()
                # kernel shape: (mid,1,5,1); center tap at dim=2 index=2
                self.temporal_peak2d.weight[:, 0, 2, 0] = 1.0

            # Average along W
            self.squeeze_w.weight.zero_()
            self.squeeze_w.bias.zero_()
            self.squeeze_w.weight[..., 0:self.w_target] += 1.0 / float(self.w_target)

            # Final output bias
            if p_prior is not None:
                self.proj_out.bias.copy_(self._logit(p_prior).to(self.proj_out.bias.device, dtype=self.proj_out.bias.dtype))
            else:
                nn.init.zeros_(self.proj_out.bias)

    def forward(self, x2d):  # x2d: (B, D, T, W)
        h = self.blocks(x2d)                             # (B, mid, T, W)
        if self.use_temporal_peak:
            h = self.temporal_peak2d(h)                 # (B, mid, T, W)
        h = self.squeeze_w(h)                           # (B, mid, T, 1)
        h = self.proj_out(h)                            # (B, 1, T, 1)
        logits = h.squeeze(-1).squeeze(1)               # (B, T)
        return logits


class OnsetHeadUpsample3D_MultiHead(nn.Module):
    """
    After UNet's penultimate layer:
      in  : (B, 128, 4, 256, 16)
      up  : → (B, 16, 4, 1024, 64)
      stem: 1×1×1 → (B, S, 4, 1024, 64)
      head: Per stem 2D head to generate logits (B, T)
      out : (B, S, 1024)  —— Pure logits (no sigmoid)
    """
    def __init__(self, in_ch=128, stems=5, d_in=4,
                 t_out=1024, w_out=64,
                 hidden1=64, hidden2=32, hidden3=16,
                 use_gn=True,
                 head_mid=32, head_blocks=2, head_use_temporal_peak=True,
                 auto_init_prior=True,
                 default_priors=None):
        super().__init__()
        self.stems   = stems
        self.t_out   = t_out
        self.d_in    = d_in
        self.w_target = w_out

        # Upsample path (only expand H,W)
        self.proj = nn.Conv3d(in_ch, hidden1, kernel_size=1, bias=False)
        self.up1  = UpBlock3D(hidden1, hidden2, use_gn=use_gn)
        self.up2  = UpBlock3D(hidden2, hidden3, use_gn=use_gn)

        # 1×1×1 to each stem 3D map
        self.to_stems = nn.Conv3d(hidden3, stems, kernel_size=1, bias=True)

        # Per stem 2D head
        self.heads = nn.ModuleList([
            Stem2DHead(d_in=d_in, w_target=w_out, mid=head_mid,
                       n_blocks=head_blocks, use_gn=True,
                       use_temporal_peak=head_use_temporal_peak)
            for _ in range(stems)
        ])

        # Initialize head (with prior)
        if auto_init_prior:
            if default_priors is None:
                # Default: Kick, Snare, Toms, Hi-Hats, Cymbals
                default_priors = [0.001, 0.001, 0.001, 0.05, 0.001]
            self.init_head(default_priors)

        # Kaiming init for 3D convs
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.to_stems.weight, mode='fan_out', nonlinearity='relu')
        if self.to_stems.bias is not None:
            nn.init.zeros_(self.to_stems.bias)

    @staticmethod
    def _logit(p, eps=1e-6):
        p = torch.as_tensor(p).clamp(eps, 1 - eps)
        return torch.log(p / (1.0 - p))

    def init_head(self, p_prior_per_stem=None):
        """Clear to_stems.bias; set each head to average along width and prior bias."""
        with torch.no_grad():
            if self.to_stems.bias is not None:
                self.to_stems.bias.zero_()
        if p_prior_per_stem is None:
            for h in self.heads:
                h.init_prior(None)
        else:
            assert len(p_prior_per_stem) == self.stems, f"len(prior)={len(p_prior_per_stem)} != stems={self.stems}"
            for i, h in enumerate(self.heads):
                h.init_prior(p_prior_per_stem[i])

    def forward(self, h):  # h: (B, 128, 4, 256, 16)
        x = self.proj(h)               # (B, 64, 4, 256, 16)
        x = self.up1(x)                # (B, 32, 4, 512, 32)
        x = self.up2(x)                # (B, 16, 4, 1024, 64)

        # Size safety (only if mismatch)
        if x.shape[-3] != self.d_in or x.shape[-2] != self.t_out or x.shape[-1] != self.w_target:
            x = F.interpolate(x, size=(self.d_in, self.t_out, self.w_target),
                              mode='trilinear', align_corners=False)
        x = self.to_stems(x)           # (B, S, 4, 1024, 64)

        logits_per_stem = []
        # Per stem
        for s in range(self.stems):
            xs = x[:, s, ...]          # (B, 4, 1024, 64)
            logit_s = self.heads[s](xs)  # (B, 1024)
            logits_per_stem.append(logit_s)

        logits = torch.stack(logits_per_stem, dim=1)  # (B, S, 1024)
        return logits

class OnsetHead(nn.Module):
    """
    Simplified version of (B, Ctot, T, F, W) → (B, S, 1024)
    1. Average F, W
    2. 1×1 Conv1d projection + several Conv1d
    3. Linear interpolation to 1024
    """
    def __init__(self, cin_tot=640, num_stems=5,
                 t_out=1024, hid=64, n_layer=2):
        super().__init__()
        self.S    = num_stems
        self.tout = t_out
        self.cin  = cin_tot // num_stems

        # Cin → hid (Conv1d along the time dimension)
        layers = [nn.Conv1d(self.cin, hid, 1), nn.ReLU(inplace=True)]
        for _ in range(n_layer-1):
            layers += [nn.Conv1d(hid, hid, 3, padding=1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)

        # Last 1×1 to convert hid → 1 (probability)
        self.to_prob = nn.Conv1d(hid, 1, 1)

    def forward(self, h):                   # h : (B,Ctot,T,F,W)
        B, _, T, F, W = h.shape
        h = (h.view(B, self.S, self.cin, T, F, W)
               .mean(dim=[4,5])             # pool F,W → (B,S,Cin,T)
               .flatten(0,1))               # → (B·S,Cin,T)

        h = self.conv(h)                    # (B·S,hid,T)
        h = self.to_prob(h).squeeze(1)      # (B·S,T)

        # Linear interpolation/crop to 1024
        if h.shape[-1] != self.tout:
            h = torch.nn.functional.interpolate(
                    h.unsqueeze(1), size=self.tout, mode="linear",
                    align_corners=False).squeeze(1)    # (B·S,t_out)

        return h.view(B, self.S, self.tout)


class UNetModel_with_multitask(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        extra_film_condition_dim=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        extra_film_use_concat=False,  # If true, concatenate extrafilm condition with time embedding, else addition
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        no_condition=False,
        # ====== Multi-task branch parameters ======
        use_onset_branch=False,  # whether to use onset branch
        use_timbre_branch=False,  # whether to use timbre branch
        onset_branch_channels=64,  # onset branch channels
        timbre_branch_channels=64,  # timbre branch channels
        onset_output_frames=1024,  # onset output frames
        timbre_feature_dim=7,  # timbre feature dimension
        num_stems=5  # stems
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.extra_film_condition_dim = extra_film_condition_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.extra_film_use_concat = extra_film_use_concat
        time_embed_dim = model_channels * 4
        self.no_condition = no_condition
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        assert not (
            self.num_classes is not None and self.extra_film_condition_dim is not None
        ), "As for the condition of theh UNet model, you can only set using class label or an extra embedding vector (such as from CLAP). You cannot set both num_classes and extra_film_condition_dim."

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.use_extra_film_by_concat = (
            self.extra_film_condition_dim is not None and self.extra_film_use_concat
        )
        self.use_extra_film_by_addition = (
            self.extra_film_condition_dim is not None and not self.extra_film_use_concat
        )

        if self.extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)
            print(
                "+ Use extra condition on UNet channel using Film. Extra condition dimension is %s. "
                % self.extra_film_condition_dim
            )
            if self.use_extra_film_by_concat:
                print("\t By concatenation with time embedding")
            elif self.use_extra_film_by_addition:
                print("\t By addition with time embedding")

        if use_spatial_transformer and (
            self.use_extra_film_by_concat or self.use_extra_film_by_addition or self.no_condition
        ):
            print(
                "+ Spatial transformer will only be used as self-attention. Because you have choose to use film as your global condition."
            )
            spatial_transformer_no_context = True
        else:
            spatial_transformer_no_context = False

        if use_spatial_transformer and not spatial_transformer_no_context:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None and not spatial_transformer_no_context:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
        
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.downsample_layers = nn.ModuleList(                 # List to store corresponding Downsample layers
            [
                th.nn.Identity()
                #Downsample(self.model_channels, False, dims=dims) # Add Downsample layer corresponding to each added block
            ]


        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            no_context=spatial_transformer_no_context,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.downsample_layers.append(th.nn.Identity()
                    # Downsample(self.model_channels, False, dims=dims)
                    ) # Add Downsample layer corresponding to each added block
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.downsample_layers.append(Downsample(1, False, dims=dims))  # Add Downsample layer corresponding to each added block
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                no_context=spatial_transformer_no_context,
            ),
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # self.downsample_layers_after_middle_block = Downsample_one_dim(ch, conv_resample, dims=dims, target_dim = 2)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            no_context=spatial_transformer_no_context,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

        self.shape_reported = False
        
        # ====== Multi-task branch initialization ======
        self.use_onset_branch = use_onset_branch
        self.use_timbre_branch = use_timbre_branch
        self.onset_branch_channels = onset_branch_channels
        self.timbre_branch_channels = timbre_branch_channels
        self.onset_output_frames = onset_output_frames
        self.timbre_feature_dim = timbre_feature_dim
        self.num_stems = num_stems
        
        #######################################
        # Onset, Timbre branch initialization #
        #######################################

        stem_ch = model_channels * channel_mult[-1]
        cin     = stem_ch // num_stems   
        if use_timbre_branch:
            self.timbre_head = TimbreHead(cin_tot=640, stems=5, timbre_dim=7)
            self.timbre_head_penult = TimbreHeadPenult(
                in_ch=self.model_channels,
                stems=num_stems,
                timbre_dim=timbre_feature_dim,
                hidden=timbre_branch_channels,
                use_mean_std=False          
            )

        if self.use_onset_branch:

            self.onset_head = OnsetHeadUpsample3D_MultiHead(
                                    in_ch=128, stems=5, d_in=4, t_out=1024, w_out=64,
                                    head_mid=32, head_blocks=2, head_use_temporal_peak=True,
                                    auto_init_prior=True,
                                    default_priors=[0.001, 0.002, 0.003, 0.05, 0.002]
                                )
            self.onset_head.init_head(p_prior_per_stem=[0.06, 0.05, 0.02, 0.10, 0.03])

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional. an [N, extra_film_condition_dim] Tensor if film-embed conditional
        :return: an [N x C x ...] Tensor of outputs.
        """
        if not self.shape_reported:
            # debug disabled
            self.shape_reported = True

        assert (y is not None) == (
            self.num_classes is not None or self.extra_film_condition_dim is not None
        ), "must specify y if and only if the model is class-conditional or film embedding conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if self.use_extra_film_by_addition:
            # Ensure FiLM embedding matches time embedding dimensions
            film_emb = self.film_emb(y)  # [B, film_dim]
            if emb.dim() == 5:  # 5D: [B, C, D, H, W]
                film_emb = film_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, film_dim, 1, 1, 1]
            elif emb.dim() == 4:  # 4D: [B, C, H, W]
                film_emb = film_emb.unsqueeze(-1).unsqueeze(-1)  # [B, film_dim, 1, 1]
            emb = emb + film_emb
        elif self.use_extra_film_by_concat:
            # Ensure FiLM embedding matches time embedding dimensions
            film_emb = self.film_emb(y)  # [B, film_dim]
            if emb.dim() == 5:  # 5D: [B, C, D, H, W]
                film_emb = film_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, film_dim, 1, 1, 1]
            elif emb.dim() == 4:  # 4D: [B, C, H, W]
                film_emb = film_emb.unsqueeze(-1).unsqueeze(-1)  # [B, film_dim, 1, 1]
            emb = th.cat([emb, film_emb], dim=1)  # concatenate along the channel dimension

        h = x.type(self.dtype)
        h, mix = th.chunk(h, chunks=2, dim=2)
        
        mix_list = []
        mix = mix[:, 0:1, :, :, :]
        mix_list.append(mix)
        for i in range(len(self.downsample_layers)):
            mix = self.downsample_layers[i](mix)
            mix_list.append(mix)

        # for module in self.input_blocks:
        for i, module in enumerate(self.input_blocks):
            mix_i = mix_list[i]
            mix_to_add =  mix_i.repeat(1, h.shape[1], 1, 1, 1)
            h = h + mix_to_add
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        h_latent = h
        dec_penult = None
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)

            mix_i = mix_list[len(mix_list)-i-1]
            mix_to_add =  mix_i.repeat(1, h.shape[1], 1, 1, 1)
            h = h + mix_to_add
            h = module(h, emb, context)
            if i == len(self.output_blocks) - 2:
                dec_penult = h
        h = h.type(x.dtype)
        # ====== Multi-task branch prediction ======
        onset_output = None
        timbre_output = None

        # Let the gradient of the branch flow back to the main body: do not detach
        h_det = dec_penult

        if self.use_onset_branch:
            # Predict onset from latent
            # h shape: [B, num_stems, latent_channels, latent_height, latent_width]
            # onset_output = self.onset_head(h_latent)
            # onset_output shape: [B, num_stems, onset_output_frames]
            assert dec_penult is not None, "dec_penult cannot be obtained, please check the output_blocks loop index"
            onset_output = self.onset_head(h_det)  # -> (B, stems, 1024)
            # onset_output = self.onset_head(dec_penult)
            # debug disabled
        
        if self.use_timbre_branch:
            # Predict timbre from latent
            timbre_output = self.timbre_head_penult(h_det)  
            # timbre_output = self.timbre_head(h_latent)
            # timbre_output shape: [B, num_stems, timbre_feature_dim]
        
        # Main output
        if self.predict_codebook_ids:
            main_output = self.id_predictor(h)
        else:
            main_output = self.out(h)
        
        # Return main output and multi-task outputs
        outputs = [main_output]
        if onset_output is not None:
            outputs.append(onset_output)
        if timbre_output is not None:
            outputs.append(timbre_output)
        
        if len(outputs) == 1:
            return outputs[0]
        else:
            # debug disabled
            return tuple(outputs)
