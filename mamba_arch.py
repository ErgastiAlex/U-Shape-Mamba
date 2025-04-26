import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import numpy as np
import einops

NEG_INF = -1000000

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

def get_horizontal_scan1(x):
    shuffle_x = x.clone()
    shuffle_x[1::2] = torch.flip(shuffle_x[1::2], dims=[1]) #flip column
    return shuffle_x.reshape(-1), shuffle_x.reshape(-1) #in horizontal scan, shuffle and reorder are same

def get_horizontal_scan2(x):
    shuffle_x = x.clone()
    shuffle_x[0::2] = torch.flip(shuffle_x[0::2], dims=[1]) #flip column
    return shuffle_x.reshape(-1), shuffle_x.reshape(-1)

def get_vertical_scan1(x):
    shuffle_x = x.clone()
    shuffle_x[:,1::2] = torch.flip(shuffle_x[:,1::2], dims=[0])
    shuffle_x = torch.transpose(shuffle_x, dim0=0, dim1=1)

    reorder_x = x.clone()
    reorder_x = torch.transpose(reorder_x, dim0=0, dim1=1)
    reorder_x[:,1::2] = torch.flip(reorder_x[:,1::2], dims=[0]) 
    return shuffle_x.reshape(-1), reorder_x.reshape(-1)

def get_vertical_scan2(x):
    shuffle_x = x.clone()
    shuffle_x[:,0::2] = torch.flip(shuffle_x[:,0::2], dims=[0])
    shuffle_x = torch.transpose(shuffle_x, dim0=0, dim1=1)

    reorder_x = x.clone()
    reorder_x = torch.transpose(reorder_x, dim0=0, dim1=1)
    reorder_x[:,0::2] = torch.flip(reorder_x[:,0::2], dims=[0])
    return shuffle_x.reshape(-1), reorder_x.reshape(-1)


def get_scan_index(image_size, number):
    index = torch.arange(image_size**2).reshape(image_size, image_size) # (H, W)

    if number % 8 == 0:
        shuffle_index, reorder_index = get_horizontal_scan1(index)
    elif number % 8 == 1:
        shuffle_index, reorder_index = get_horizontal_scan1(index)
        shuffle_index = torch.flip(shuffle_index, dims=[0])
        reorder_index = torch.flip(reorder_index, dims=[0])
    elif number % 8 == 2:
        shuffle_index, reorder_index = get_horizontal_scan2(index)
    elif number % 8 == 3:
        shuffle_index, reorder_index = get_horizontal_scan2(index)
        shuffle_index = torch.flip(shuffle_index, dims=[0])
        reorder_index = torch.flip(reorder_index, dims=[0])
    elif number % 8 == 4:
        shuffle_index, reorder_index = get_vertical_scan1(index)
    elif number % 8 == 5:
        shuffle_index, reorder_index = get_vertical_scan1(index)
        shuffle_index = torch.flip(shuffle_index, dims=[0])
        reorder_index = torch.flip(reorder_index, dims=[0])
    elif number % 8 == 6:
        shuffle_index, reorder_index = get_vertical_scan2(index)
    elif number % 8 == 7:
        shuffle_index, reorder_index = get_vertical_scan2(index)
        shuffle_index = torch.flip(shuffle_index, dims=[0])
        reorder_index = torch.flip(reorder_index, dims=[0])

    return shuffle_index, reorder_index
    

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            number=0,
            image_size=32,
            skip_gate=False,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.number=number

        self.skip_gate = skip_gate
        if self.skip_gate:
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = list([
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(1)
        ])
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = list([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(1)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.shuffle_index, self.reorder_index = get_scan_index(image_size, number)

        # self.in_proj_w  = nn.Parameter(torch.randn(self.d_inner, image_size**2))
        # self.out_proj_w = nn.Parameter(torch.randn(self.d_inner, image_size**2))


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def get_horizontal_scan1(self, x):
        x[:,:,1::2] = torch.flip(x[:,:,1::2], dims=[3]) #flip column
        return x
    
    def revert_horizontal_scan1(self, x):
        return self.get_horizontal_scan1(x)

    def get_horizontal_scan2(self, x):
        x[:,:,0::2] = torch.flip(x[:,:,0::2], dims=[3]) #flip column
        return x
    
    def revert_horizontal_scan2(self, x):
        return self.get_horizontal_scan2(x)

    def get_vertical_scan1(self, x):
        x[:,:,:,1::2] = torch.flip(x[:,:,:,1::2], dims=[2]) #flip row
        x = torch.transpose(x, dim0=2, dim1=3)
        return x

    def revert_vertical_scan1(self, x):
        x = torch.transpose(x, dim0=2, dim1=3) #change H and W
        x[:,:,:,1::2] = torch.flip(x[:,:,:,1::2], dims=[2]) #flip row
        return x

    def get_vertical_scan2(self, x):
        x[:,:,:,0::2] = torch.flip(x[:,:,:,0::2], dims=[2])
        x = torch.transpose(x, dim0=2, dim1=3)
        return x
    
    def revert_vertical_scan2(self, x):
        x = torch.transpose(x, dim0=2, dim1=3)
        x[:,:,:,0::2] = torch.flip(x[:,:,:,0::2], dims=[2])
        return x
    
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        # x = x.view(B, 1, C, -1) # B, 1, D, L

        if self.number % 8 == 0:
            x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L) # (1, 1, 192, 3136)
            xs = x_hw
        elif self.number % 8 == 1:
            x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L)
            xs = torch.flip(x_hw, dims=[-1])
        elif self.number % 8 == 2:
            x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
            xs = x_hw
        elif self.number % 8 == 3:
            x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
            xs = torch.flip(x_hw, dims=[-1])
        elif self.number % 8 == 4:
            x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
            xs = x_wh
        elif self.number % 8 == 5:
            x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
            xs = torch.flip(x_wh, dims=[-1])
        elif self.number % 8 == 6:
            x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
            xs = x_wh
        elif self.number % 8 == 7:
            x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
            xs = torch.flip(x_wh, dims=[-1])

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.contiguous().view(B, K, -1, L), self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().contiguous().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
    
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
         
        if self.number % 8 == 0:
            out = out_y[:, 0].view(B, -1, H, W)
            out = self.revert_horizontal_scan1(out)

        elif self.number % 8 == 1:
            out = out_y[:, 0].reshape(B, -1, L)
            out = torch.flip(out, dims=[-1]).view(B, -1, H, W)
            out = self.revert_horizontal_scan1(out)

        elif self.number % 8 == 2:
            out = out_y[:, 0].view(B, -1, H, W)
            out = self.revert_horizontal_scan2(out)
            
        elif self.number % 8 == 3:
            out = out_y[:, 0].reshape(B, -1, L)
            out = torch.flip(out, dims=[-1]).view(B, -1, H, W)
            out = self.revert_horizontal_scan2(out)

        elif self.number % 8 == 4:
            out = out_y[:, 0].view(B, -1, W, H)
            out = self.revert_vertical_scan1(out)

        elif self.number % 8 == 5:
            out = out_y[:, 0].reshape(B, -1, L)
            out = torch.flip(out, dims=[-1]).view(B, -1, W, H)
            out = self.revert_vertical_scan1(out)

        elif self.number % 8 == 6:
            out = out_y[:, 0].view(B, -1, W, H)
            out = self.revert_vertical_scan2(out)

        elif self.number % 8 == 7:
            out = out_y[:, 0].reshape(B, -1, L)
            out = torch.flip(out, dims=[-1]).view(B, -1, W, H)
            out = self.revert_vertical_scan2(out).view(B, -1, H, W)

        y = einops.rearrange(out, 'b c h w -> b c (h w)')
        return y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)

        if self.skip_gate:
            x = xz
        else:
            x, z = xz.chunk(2, dim=-1)
            
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        y = self.forward_core(x) # B C L
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        
        if not self.skip_gate:
            y = y * F.silu(z)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            image_size: int=32,
            number:int =0,
            skip_gate: bool=False,
            **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, 
                                   d_state=d_state, 
                                   expand=expand,dropout=attn_drop_rate, 
                                   number=number,
                                   image_size=image_size,
                                   skip_gate=skip_gate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))


    def forward(self, x):
        """ 
        x = B, H, W, C
        """
        x = x*self.skip_scale + self.drop_path(self.self_attention(x)) # [B,H,W,C]
        return x


