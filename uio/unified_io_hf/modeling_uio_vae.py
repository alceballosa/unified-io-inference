import collections
import math
from dataclasses import dataclass
from itertools import repeat
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


@dataclass
class VAEConfig:
    embed_dim: int = 256
    n_embed: int = 1024
    double_z: bool = False
    z_channels: int = 256
    resolution: int = 256
    in_channels: int = 3
    out_ch: int = 3
    ch: int = 128
    ch_mult: Tuple[int, int, int, int, int] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int] = (16,)
    dropout: float = 0
    dtype: Any = torch.float32


def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function

    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
    ):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs
        )

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class AttnBlock(nn.Module):
    def __init__(self, n_in, dtype=torch.float32):
        super().__init__()
        self.n_in = n_in
        self.norm = nn.GroupNorm(num_groups=32, num_channels=n_in)
        self.q = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_in,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.k = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_in,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.v = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_in,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.proj_out = nn.Conv2d(
            in_channels=n_in,
            out_channels=n_in,
            kernel_size=1,
            bias=True,
            padding="same",
        )
        self.dtype = dtype

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        w_ = torch.einsum(
            "bcq,bck->bqk",
            torch.reshape(q, (b, c, h * w)),
            torch.reshape(k, (b, c, h * w)),
        )

        w_ = w_ * (c**-0.5)
        w_ = torch.nn.functional.softmax(w_, dim=-1).type(self.dtype)

        h_ = torch.einsum("bqk,bdk->bdq", w_, torch.reshape(v, (b, c, h * w)))

        h_ = torch.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)
        return x + h_


class Conv2dSame(torch.nn.Conv2d):
    """
    Implementation from:
    https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    """
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i_h, i_w = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=i_h, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=i_w, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Downsample(nn.Module):
    def __init__(self, n_in, dtype=torch.float32):
        super().__init__()
        self.conv = Conv2dSame(n_in, n_in, kernel_size=3, stride=2, bias=True)
        self.dtype = dtype

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, n_in, dtype=torch.float32):
        super().__init__()
        self.n_in = n_in
        self.dtype = dtype
        self.conv = nn.Conv2d(
            self.n_in, self.n_in, kernel_size=3, stride=1, padding="same", bias=True
        )

    def forward(self, x):
        B, H, W, C = x.shape
        #x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, dtype=torch.float32):
        super(ResBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dtype = dtype

        self.conv1 = nn.Conv2d(
            n_in, n_out, kernel_size=3, stride=1, padding="same", bias=True
        )
        self.conv2 = nn.Conv2d(
            n_out, n_out, kernel_size=3, stride=1, padding="same", bias=True
        )

        self.norm1 = nn.GroupNorm(32, n_out)
        self.norm2 = nn.GroupNorm(32, n_out)

        if n_in != n_out:
            self.nin_shortcut = nn.Conv2d(
                n_in, n_out, kernel_size=1, stride=1, padding="same", bias=True
            )
        else:
            self.nin_shortcut = None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class VAE_Encoder(nn.Module):
    """PyTorch implementation of Taming VAE encoder"""

    def __init__(self, config):
        super(VAE_Encoder, self).__init__()
        self.config = config
        curr_res = config.resolution
        self.num_resolutions = len(config.ch_mult)
        self.in_ch_mult = (1,) + tuple(config.ch_mult)

        self.conv_in = nn.Conv2d(
            in_channels=3,
            out_channels=1 * config.ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True,
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block_in = config.ch * self.in_ch_mult[i_level]

            block_out = config.ch * config.ch_mult[i_level]
            print(block_in, block_out)
            res_blocks_level = nn.ModuleList()
            attn_blocks_level = nn.ModuleList()

            for i_block in range(config.num_res_blocks):
                res_blocks_level.append(ResBlock(block_in, block_out, config.dtype))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn_blocks_level.append(AttnBlock(block_in))

            self.res_blocks.append(res_blocks_level)
            self.attn_blocks.append(attn_blocks_level)

            if i_level != self.num_resolutions - 1:
                self.downsamples.append(Downsample(block_in))

                curr_res = curr_res // 2

        self.mid_block_1 = ResBlock(block_in, block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(block_in, block_in)

        self.conv_out = nn.Conv2d(
            in_channels=block_in,
            out_channels=config.z_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=True,
        )

    def forward(self, x, training=False):

        curr_res = self.config.resolution
        hs = self.conv_in(x)

        for i_level in range(self.num_resolutions):
            res_blocks_level = self.res_blocks[i_level]
            attn_blocks_level = self.attn_blocks[i_level]

            block_in = self.config.ch * self.in_ch_mult[i_level]
            block_out = self.config.ch * self.config.ch_mult[i_level]

            for i_block in range(self.config.num_res_blocks):
                hs = res_blocks_level[i_block](hs)
                block_in = block_out
                if curr_res in self.config.attn_resolutions:
                    hs = attn_blocks_level[i_block](hs)

            if i_level != self.num_resolutions - 1:
                hs = self.downsamples[i_level](hs)
                curr_res = curr_res // 2
            print(hs.shape)
        hs = self.mid_block_1(hs)
        hs = self.mid_attn_1(hs)
        hs = self.mid_block_2(hs)
        hs = self.norm_out(hs)

        hs = nn.functional.relu(hs)
        hs = self.conv_out(hs)

        return hs


class VAE_Decoder(nn.Module):
    """PyTorch implementation of Taming VAE encoder"""

    def __init__(self, config):
        super(VAE_Decoder, self).__init__()
        self.config = config
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.num_resolutions = len(config.ch_mult)
        self.curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        self.block_in = config.ch * config.ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(
            in_channels=config.z_channels,
            out_channels=self.block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block_1 = ResBlock(self.block_in, self.block_in)
        self.mid_attn_1 = AttnBlock(self.block_in)
        self.mid_block_2 = ResBlock(self.block_in, self.block_in)

        self.up_layers = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            i_idx = self.num_resolutions - i_level - 1
            block_out = config.ch * config.ch_mult[i_level]
            up_res_blocks = nn.ModuleList()
            for i_block in range(config.num_res_blocks + 1):
                up_res_blocks.append(ResBlock(self.block_in, block_out))
                self.block_in = block_out
                if self.curr_res in config.attn_resolutions:
                    up_res_blocks.append(AttnBlock(self.block_in))
            self.up_layers.append(nn.Sequential(*up_res_blocks))
            if i_level != 0:
                self.up_layers.append(
                    Upsample(
                        self.block_in,
                    )
                )
                self.curr_res = self.curr_res * 2

        self.norm_out = nn.GroupNorm(
            num_groups=self.block_in,
            num_channels=self.block_in,
        )
        self.nonlinearity = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(
            in_channels=self.block_in,
            out_channels=config.out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)
        for layer in self.up_layers:
            h = layer(h)
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, embedding_init=None):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        if embedding_init is None:
            embedding_init = self.default_embed_init
        # TODO: check this dtype
        self.embedding = nn.Parameter(
            embedding_init((self.n_e, self.e_dim), dtype="float32"),
            requires_grad=True,
        )

    def default_embed_init(self, shape, dtype):
        bound = 1 / shape[1]
        # fix the below line
        # embedding = torch.zeros(shape, dtype=dtype)
        # TODO: validate
        embedding = torch.zeros(shape)
        nn.init.uniform_(embedding, -bound, bound)
        return embedding

    def get_codebook_entry(self, indices):
        min_encodings = nn.functional.one_hot(indices, self.n_e).to(
            self.embedding.device
        )
        z_q = torch.einsum("bqk,kd->bqd", min_encodings, self.embedding)
        return z_q

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2 * torch.einsum("ij,kj->ik", z_flattened, self.embedding)
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z.detach() + (z_q - z.detach())

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class DiscreteVAE(nn.Module):
    """Jax implementation of Taming VAE"""

    def __init__(self, config):
        super().__init__()
        self.config: VAEConfig = config
        cfg = self.config
        self.encoder = VAE_Encoder(cfg)
        self.quant_conv = nn.Conv2d(
            in_channels=cfg.ch_mult[-1] * cfg.ch,
            out_channels=cfg.z_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.quantize = VectorQuantizer(
            e_dim=cfg.embed_dim,
            n_e=cfg.n_embed,
            beta=0.25,
        )

        self.post_quant_conv = nn.Conv2d(
            in_channels=cfg.z_channels,
            out_channels=cfg.ch_mult[-1] * cfg.ch,
            kernel_size=(1, 1),
            bias=True,
        )

        self.decoder = VAE_Decoder(cfg)

    def encode(self, x, training=False):
        h = self.encoder(x, training)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h.permute(0, 2, 3, 1))
        return quant.permute(0, 3, 1, 2), emb_loss, info

    def decode(self, quant, training=False):
        quant = self.post_quant_conv(quant.permute(0, 2, 3, 1))
        dec = self.decoder(quant, training)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize._embedding.weight[code_b, :]
        bs, dim = quant_b.shape
        size = int(math.sqrt(dim))
        quant_b = quant_b.view(bs, size, size, -1)
        dec = self.decode(quant_b.permute(0, 3, 1, 2))
        return dec

    def get_codebook_indices(self, x, vae_decode=False, training=False):
        h = self.encoder(x, training)
        h = self.quant_conv(h)
        z, _, [_, _, indices] = self.quantize(h.permute(0, 2, 3, 1))

        if vae_decode:
            _ = self.decode(z.permute(0, 3, 1, 2), training)

        return indices.view(indices.shape[0], -1)

    def forward(self, x, training=False):
        quant, diff, _ = self.encode(x, training)
        dec = self.decode(quant, training)
        return dec
