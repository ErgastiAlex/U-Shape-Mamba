# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for USM using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from torchvision.utils import save_image
import argparse
import logging
import os
from torchvision.utils import save_image


from models import USM_models
from diffusers.models import AutoencoderKL
from diffusion.rectified_flow import RectifiedFlow
from tqdm import tqdm

from accelerate import Accelerator

from functools import partial

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def out2img(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new USM model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    logger = create_logger("./")
    accelerator = Accelerator()
    device = accelerator.device
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    ema = USM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=False,
        use_checkpoint=args.use_ckpt,
        learn_pos_emb = args.learn_pos_emb,
        use_convtranspose=args.use_convtranspose,
        has_text=args.has_text,
        skip_gate=args.skip_gate,
        skip_conn=args.skip_conn)
    

    rectified_flow = RectifiedFlow(num_timesteps=25)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    logger.info(f"USM Parameters: {sum(p.numel() for p in ema.parameters()):,}")


    if args.num_classes != 0:
        sample_fn = partial(rectified_flow.sample, ema.forward_with_cfg)
    else:
        sample_fn = partial(rectified_flow.sample, ema.forward)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vae.to(device)

    ema.to(device)
    ema.load_state_dict(ckpt["ema"], strict=False)
    ema.eval()
    
    device = accelerator.device 

    os.makedirs('./samples', exist_ok=True)
    with torch.no_grad():
        for i in tqdm(range(args.num_samples//args.batch_size), desc="Sampling"):
            # Create sampling noise:
            n = args.batch_size
            z = torch.randn(n, 4, latent_size, latent_size, device=device)*rectified_flow.noise_scale

            model_kwargs = dict(y=0)
                
            # Sample images:
            samples = sample_fn(z, model_kwargs=model_kwargs, progress=False)
            samples = vae.decode(samples / 0.18215).sample

            for j,img in enumerate(samples):
                save_image((img+1)/2, f'./samples/sample_{i*args.batch_size+j}.png')
            


if __name__ == "__main__":
    # Default args here will train USM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--experiment_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(USM_models.keys()), default="USM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training

    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument('--learn_pos_emb', action='store_true')
    parser.add_argument('--skip_gate', action='store_true')
    parser.add_argument('--use_ckpt', action='store_true')
    parser.add_argument('--skip_conn', action='store_true')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--has_text', action='store_true')
    parser.add_argument('--use_convtranspose', action='store_true')

    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth")

    args = parser.parse_args()
    main(args)
