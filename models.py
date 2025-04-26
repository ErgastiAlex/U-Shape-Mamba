# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_arch import VSSBlock
import einops
import torch.utils.checkpoint as checkpoint

from inspect import isfunction
if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"
print(f"attention mode is {ATTENTION_MODE}")


def modulate2d(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, text, mask=None):
        B, L, C = x.shape
        q = self.to_q(x)
        # text = default(text, x)
        k = self.to_k(text)
        v = self.to_v(text)

        q, k, v = map(
            lambda t: einops.rearrange(t, "B L (H D) -> B H L D", H=self.heads), (q, k, v)
        )  # B H L D
        if ATTENTION_MODE == "flash":
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "xformers":
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "B L H D -> B L (H D)", H=self.heads)
        elif ATTENTION_MODE == "math":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented
        return self.to_out(x)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

#################################################################################
#                                 Core USM Model                                #
#################################################################################

class USMBlock(nn.Module):
    """
    A USM block with adaptive layer norm zero (adaLN-Zero) conUSMioning.
    """
    def __init__(self, hidden_size, 
                       number=0, 
                       has_text=False, 
                       image_size=32,
                       skip_gate=False,
                       skip_conn=False,
                       use_checkpoint=True):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.has_text = has_text
        self.attn = VSSBlock(
                hidden_dim= hidden_size,
                drop_path=0,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=16,
                input_resolution=256,
                number=number,
                image_size=image_size,
                skip_gate=skip_gate)
        
        self.use_checkpoint = use_checkpoint
        adaln_number = (3*2 if self.has_text else 3)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, adaln_number * hidden_size, bias=True)
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if self.has_text:
            self.mca = CrossAttention(
                query_dim=hidden_size, context_dim=hidden_size, heads=8, dim_head=64, dropout=0.0
            )
            self.norm_mca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.skip_conn=skip_conn
        if skip_conn:
            self.skip = nn.Linear(2*hidden_size, hidden_size, bias=True)
            
            
    def forward(self, x, c, skip=None, text=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x, c, skip, text)
        else:
            return self._forward(x, c, skip, text)
    
    def _forward(self, x, c, skip=None, text=None):
        """
            x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        if skip is not None and self.skip_conn:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip(x)


        if not self.has_text:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
            x = x + gate_msa.unsqueeze(1).unsqueeze(1) * self.attn(modulate2d(self.norm1(x), shift_msa, scale_msa))
        else:
            shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca= self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1).unsqueeze(1) * self.attn(modulate2d(self.norm1(x), shift_msa, scale_msa))
            x = x.view(B, -1, C)
            x = x + gate_mca.unsqueeze(1) * self.mca(modulate(self.norm_mca(x), shift_mca, scale_mca), text)
            x = x.view(B, H, W, C)
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of USM.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
        


class USM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_checkpoint=True,
        learn_pos_emb = False,
        num_scans = 4,
        has_text = False,
        d_context = 768,
        skip_gate = False,
        skip_conn = False,
        use_convtranspose=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.learn_pos_emb = learn_pos_emb
        self.num_classes = num_classes
        self.has_text = has_text
        self.use_convtranspose = use_convtranspose
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        if self.num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = nn.Linear(d_context, hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=self.learn_pos_emb)

        self.enc_blocks = []
        enc_div = (depth//2)//3
        image_size = input_size
        for i in range(depth//2):
            self.enc_blocks.append(USMBlock(hidden_size, 
                    number=i, 
                    has_text=has_text,
                    skip_gate=skip_gate,
                    skip_conn=False,
                    image_size = image_size,
                    use_checkpoint=use_checkpoint))
            if (i+1) % enc_div == 0:
                self.enc_blocks.append(
                        nn.Conv2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0)
                )
                image_size //= 2

                
        
        self.enc_blocks = nn.ModuleList(self.enc_blocks)

        self.middle_block = USMBlock(hidden_size,
                    number=depth//2, 
                    has_text=has_text,
                    image_size=image_size,
                    skip_conn=False,
                    use_checkpoint=use_checkpoint,
                    )
    
        self.dec_blocks = []
        dec_div = (depth//2)//3
        for i in range(depth//2):
            if (i) % dec_div == 0:
                if self.use_convtranspose:
                    self.dec_blocks.append(
                        nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2, padding=0)
                    )
                else:
                    self.dec_blocks.append(UpsampleOneStep(2, hidden_size, hidden_size))
                image_size *=2

            self.dec_blocks.append(USMBlock(hidden_size, 
                    number=i+depth//2+1, 
                    has_text=has_text,
                    image_size=image_size,
                    skip_gate=skip_gate,
                    skip_conn=skip_conn,
                    use_checkpoint=use_checkpoint,
                    ))

        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.learn_pos_emb == False:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.num_classes != 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in USM blocks:
        for block in self.enc_blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.dec_blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.middle_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.middle_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of USM.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        if self.has_text:
            y = self.y_embedder(y)
            c = t + y.mean(1)
        elif self.num_classes > 0:
            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                                # (N, D)
        else:
            c = t

        B, L, D = x.shape

        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), D).contiguous()  # [B,H,W,C]

        xs=[]
        for block in self.enc_blocks:
            if not isinstance(block, nn.Conv2d):
                x = block(x, c, skip=None, text=y)
                xs.append(x)
            else:
                x = x.permute(0, 3, 1, 2).contiguous()
                x = block(x)
                x = x.permute(0, 2, 3, 1).contiguous()

        x = self.middle_block(x, c, skip=None, text=y)

        for block in self.dec_blocks:
            if not isinstance(block, nn.ConvTranspose2d):
                x = block(x, c, skip=xs.pop(), text=y)
            else:
                x = x.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]
                x = block(x)
                x = x.permute(0, 2, 3, 1).contiguous() # [B,H,W,C]

        x = x.view(B, L, D).contiguous()  # [B,H,W,C]
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of USM, but also batches the unconUSMional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        assert self.num_classes != 0, "Forward with cfg requires class labels."
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([half_eps,half_eps], dim=0)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   USM Configs                                  #
#################################################################################

def USM_XL_1(**kwargs):
    return USM(depth=57, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def USM_L_1(**kwargs):
    return USM(depth=49, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def USM_B_1(**kwargs):
    return USM(depth=25, hidden_size=768, patch_size=1, num_heads=16, **kwargs)

def USM_S_1(**kwargs):
    return USM(depth=13, hidden_size=384, patch_size=1, num_heads=16, **kwargs)

USM_models = {
    'USM-XL/1': USM_XL_1,
    'USM-L/1': USM_L_1,
    'USM-B/1': USM_B_1,
    'USM-S/1': USM_S_1,
}
