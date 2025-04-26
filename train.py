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
from copy import deepcopy
from torchvision.utils import save_image
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model
from my_metric import MyMetric

from models import USM_models
from diffusers.models import AutoencoderKL
from diffusion.rectified_flow import RectifiedFlow
import random

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

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

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace("/", "-")  # e.g., USM-XL/2 --> USM-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    try:
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [int(x.split("_")[-1]) for x in checkpoint_files]
        checkpoint_files.sort()
        checkpoint_file = checkpoint_files[-1]
    except:
        checkpoint_file = 0

    accelerator_project_config = ProjectConfiguration(project_dir=experiment_dir, automatic_checkpoint_naming=True, iteration = checkpoint_file+1, total_limit = 3)
    accelerator = Accelerator(project_config=accelerator_project_config)

    set_seed(args.global_seed)  # Set global seed for reproducibility
    
    device = accelerator.device
    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = USM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=False,
        use_checkpoint=args.use_ckpt,
        learn_pos_emb = args.learn_pos_emb,
        use_convtranspose=args.use_convtranspose,
        has_text=args.has_text,
        skip_gate=args.skip_gate,
        skip_conn=args.skip_conn)
    # Note that parameter initialization is done within the USM constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    rectified_flow = RectifiedFlow(num_timesteps=25, sampling=args.sampling)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    logger.info(f"USM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.has_text:
        from encoders import FrozenCLIPEmbedder
        text_emb = FrozenCLIPEmbedder()
        text_emb.to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.has_text:
        import torchvision
        train_dataset = torchvision.datasets.CocoCaptions(root = os.path.join(args.data_path, "train"),
                        annFile = os.path.join(args.data_path,"annotations","train.json"),
                        transform=transform)
        test_dataset = torchvision.datasets.CocoCaptions(root = os.path.join(args.data_path, "val"),
                        annFile = os.path.join(args.data_path, "annotations", "val.json"),
                        transform=transform)
        def collate_fn(batch):
            return tuple(zip(*batch))
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    else:
        train_dataset = ImageFolder(os.path.join(args.data_path,"train"), transform=transform)
        test_dataset = ImageFolder(os.path.join(args.data_path,"val"), transform=transform)

        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    logger.info(f"Train dataset contains {len(train_dataset)}")
    logger.info(f"Test dataset contains {len(test_dataset)}")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train() 
    ema.eval()  # EMA model should always be in eval mode

    if args.num_classes != 0:
        sample_fn = partial(rectified_flow.sample, ema.forward_with_cfg)
    else:
        sample_fn = partial(rectified_flow.sample, ema.forward)

    model, ema, opt, loader, test_loader = accelerator.prepare(model, ema, opt, loader, test_loader)

    vae.to(device)
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    try:
        accelerator.load_state()
        logger.info(f"Checkpoint found, starting from {checkpoint_file*args.ckpt_every} steps.")
        train_steps = checkpoint_file*args.ckpt_every
        log_steps = train_steps % args.log_every
    except:
        logger.info("No checkpoint found, starting from scratch.")
    
    device = accelerator.device 

    my_metrics = MyMetric()
    best_fid = 1e9
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        if train_steps >= 400000:
            break

        for x, y in loader:
            if args.has_text:
                captions=[]
                for cap in y:
                    captions.append(random.choice(cap))
                y = text_emb(random.choice(captions))
                x = torch.stack(x)
                
            if train_steps >= 400000:
                break


            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)
            loss_dict = rectified_flow.training_loss(model, x, model_kwargs)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)

            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save USM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                accelerator.save_state()
                logger.info(f"Checkpoint saved at step {train_steps}.")
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                with torch.no_grad():
                    my_metrics.reset()

                    i = 0
                    for x, y in test_loader:
                        if args.has_text:
                            captions=[]
                            for cap in y:
                                captions.append(random.choice(cap))
                            y = text_emb(random.choice(captions))
                            x = torch.stack(x)

                
                        # Create sampling noise:
                        n = x.shape[0]
                        z = torch.randn(n, 4, latent_size, latent_size, device=device)*rectified_flow.noise_scale

                        if args.num_classes != 0:
                            assert False, "Not implemented"
                            model_kwargs = dict(y=y, cfg_scale=2.0)
                        else:
                            model_kwargs = dict(y=y)

                        # Sample images:
                        samples = sample_fn(z, model_kwargs=model_kwargs, progress=False)

                        samples = vae.decode(samples / 0.18215).sample

                        x = out2img(x)
                        samples = out2img(samples)

                        my_metrics.update_real(x)
                        my_metrics.update_fake(samples)

                        i+=1
                        if i > args.samples and args.samples > 0:
                            break

                    _metric_dict = my_metrics.compute()
                    fid = _metric_dict["fid"]
                    _metric_dict = {f"eval/{k}": v for k, v in _metric_dict.items()}
                    logger.info(f"Eval FID at steps {train_steps}: {fid:.4f}, Best FID: {best_fid:.4f}")
                    
                    if fid < best_fid:
                        checkpoint_path = f"{experiment_dir}/best_{train_steps}.pt"
                        accelerator.save({
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "ema": accelerator.unwrap_model(ema).state_dict(),
                            "opt": opt.optimizer.state_dict(),
                            "args": args
                        }, checkpoint_path)
                        
                    best_fid = min(fid, best_fid)

                    


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    checkpoint_path = f"{checkpoint_dir}/last.pt"
    accelerator.save({
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": accelerator.unwrap_model(ema).state_dict(),
            "opt": opt.optimizer.state_dict(),
            "args": args
        }, checkpoint_path)
    logger.info(f"Saved last to {checkpoint_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train USM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
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
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=15000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument('--learn_pos_emb', action='store_true')
    parser.add_argument('--skip_gate', action='store_true')
    parser.add_argument('--use_ckpt', action='store_true')
    parser.add_argument('--skip_conn', action='store_true')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--has_text', action='store_true')
    parser.add_argument('--use_convtranspose', action='store_true')

    parser.add_argument("--sampling", type=str, choices=["log", "uniform"], default="log")
    args = parser.parse_args()
    main(args)
