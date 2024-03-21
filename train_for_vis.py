#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""
import configparser
import argparse
import logging
import math
import os
import cv2
import random
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from models.unet_3d_condition_gligen import UNet3DConditionModel
from mmengine.config import Config, ConfigDict
from mmengine import build_from_cfg
from dataset.youtube_loader import VISDataset
from mmtrack.registry import DATASETS

from pipelines.pipeline_text_to_video_synth import TextToVideoSDPipeline


class CombinedDataset(Dataset):

    def __init__(self, datasets, probabilities):
        self.datasets = datasets
        self.probabilities = probabilities

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        dataset_idx = torch.multinomial(torch.tensor(self.probabilities),
                                        1).item()
        dataset = self.datasets[dataset_idx]

        idx = random.randint(0, len(dataset) - 1)
        return dataset[idx]


classes_vis = [
    'airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog',
    'duck', 'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox', 'frog',
    'giant_panda', 'giraffe', 'horse', 'leopard', 'lizard', 'monkey',
    'motorbike', 'mouse', 'parrot', 'person', 'rabbit', 'shark', 'skateboard',
    'snake', 'snowboard', 'squirrel', 'surfboard', 'tennis_racket', 'tiger',
    'train', 'truck', 'turtle', 'whale', 'zebra'
]

classes_nus = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path,
                                   fourcc,
                                   fps=fps,
                                   frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=
        "Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=
        "Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=
        ("Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
         " `args.validation_prompt` multiple times: `args.num_validation_images`."
         ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=
        "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help=
        "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=
        ('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
         ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help=
        "Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=
        ("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
         ),
    )
    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=
        ("Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
         " remote repository specified with --pretrained_model_name_or_path."),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=
        ("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
         ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay",
                        type=float,
                        default=1e-2,
                        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=
        "The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=
        ("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=
        ("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
         ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
         ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=
        ("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
         " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=
        ("Whether training should be resumed from a previous checkpoint. Use a path saved by"
         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
         ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )

    parser.add_argument(
        "--cfg_name",
        type=str,
        default=None,
        help="dataset cfg",
    )

    parser.add_argument(
        "--geo_train",
        action='store_true',
        help="train geo layer only",
    )

    parser.add_argument(
        "--train_mode",
        type=str,
        help="temp or geo",
    )

    parser.add_argument(
        "--fg_reweight",
        type=float,
        default=1.0,
        help="Scale of foreground object loss",
    )
    parser.add_argument(
        "--combine_dataset",
        action="store_true",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (lambda image_url_or_path: load_image(image_url_or_path)
                      if urlparse(image_url_or_path).scheme else PIL.Image.
                      open(image_url_or_path).convert("RGB"))(url)
    return original_image


def main():
    args = parse_args()
    class_config = configparser.ConfigParser()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=
            ("Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
             " use `--variant=non_ema` instead."),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(
        args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id
                                  or Path(args.output_dir).name,
                                  exist_ok=True,
                                  token=args.hub_token).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                        subfolder="vae",
                                        revision=args.revision)
    unet = UNet3DConditionModel.from_pretrained(
        '/gligen2damo' if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=False,
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(),
                            model_cls=UNet3DConditionModel,
                            model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet3DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet3DConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate *
                              args.gradient_accumulation_steps *
                              args.per_gpu_batch_size *
                              accelerator.num_processes)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.geo_train:
        # Freeze unet blocks
        logger.info("Enable Geo Train. Train fusers only.")
        unet.requires_grad_(True)
        parameters_list = []
        if args.train_mode == 'geo':
            for name, para in unet.named_parameters():
                if 'fuser' in name or 'position_net' in name:
                    parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        if args.train_mode == 'temp':
            for name, para in unet.named_parameters():
                if 'temp' in name or 'tracklet' in name or 'instance_embedding' in name or 'transformer_in' in name or 'injector' in name:
                    parameters_list.append(para)
                    para.requires_grad = True
                else:
                    para.requires_grad = False
        optimizer = optimizer_cls(
            parameters_list,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    def tokenize_captions(captions):
        inputs = tokenizer(captions,
                           max_length=tokenizer.model_max_length,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        return inputs.input_ids

    # check para
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para_freeze.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    cfg = Config.fromfile(f'dataset/{args.cfg_name}')
    train_dataset = build_from_cfg(cfg.train_dataloader.dataset, DATASETS)
    if args.combine_dataset:
        cfg = Config.fromfile(f'dataset/got10k_cfg.py')
        train_dataset_2 = build_from_cfg(cfg.train_dataloader.dataset,
                                         DATASETS)
        cfg = Config.fromfile(f'dataset/lasot_cfg.py')
        train_dataset_3 = build_from_cfg(cfg.train_dataloader.dataset,
                                         DATASETS)
        # train_dataset = ConcatDataset(datasets=[train_dataset, train_dataset_2, train_dataset_3])
        train_dataset = CombinedDataset(
            [train_dataset, train_dataset_2, train_dataset_3], [0.6, 0.3, 0.1])
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Tracklet2video", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                pixel_values = batch["pixel_values"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True)
                latents = tensor_to_vae_latent(pixel_values, vae)
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)
                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(
                    tokenize_captions(batch["text_prompt"]).to(
                        accelerator.device))[0]

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz,
                                          device=latents.device,
                                          generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(
                        tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)

                # Concatenate the `original_image_embeds` with the `noisy_latents`.

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss

                bboxes_prompt = batch["bboxes"]
                fixed_num_objs = bboxes_prompt.shape[-2]
                cur_bs, _, num_frames = noisy_latents.shape[:3]
                if num_frames != args.num_frames:
                    print(
                        f"Expected frames: {args.num_frames}, but got: {num_frames}"
                    )

                text_embeddings = []
                for bid in range(cur_bs):
                    class_index = batch['seg_phrase'][bid].flatten()

                    if batch['dataset'][bid] == 'got10k':
                        class_config.clear()
                        class_config.read(
                            f"/GOT10K/train/{batch['video_name'][bid]}/meta_info.ini"
                        )
                        class_text = class_config["METAINFO"]["object_class"]
                        class_text = [
                            class_text,
                        ] * num_frames * fixed_num_objs
                    elif batch['dataset'][bid] == 'lasot':
                        class_text = batch['video_name'][bid]
                        class_text = [
                            class_text,
                        ] * num_frames * fixed_num_objs
                    elif batch['dataset'][bid] == 'nus':
                        class_text = [
                            classes_nus[int(_index)] for _index in class_index
                        ]
                    else:
                        class_text = [
                            classes_vis[int(_index)] for _index in class_index
                        ]
                    token_ids = tokenizer(
                        class_text,
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)

                    _text_embeddings = text_encoder(token_ids).pooler_output
                    _text_embeddings = _text_embeddings.view(
                        num_frames, fixed_num_objs, -1)
                    text_embeddings.append(_text_embeddings)
                text_embeddings = torch.stack(text_embeddings, dim=0)
                masks = batch['mask']

                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz,
                                          device=latents.device,
                                          generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_masks = torch.zeros_like(masks)
                    masks = torch.where(prompt_mask, null_masks, masks)

                cross_attention_kwargs = {}
                cross_attention_kwargs["gligen"] = {
                    "boxes": bboxes_prompt.flatten(1, 2),
                    "positive_embeddings": text_embeddings.flatten(1, 2),
                    "masks": masks.flatten(1, 2)
                }
                cross_attention_kwargs["fixed_objs"] = fixed_num_objs

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # Convert normalized bboxes to actual coordinates
                bboxes_actual = batch['bboxes'].clone().float()
                bboxes_actual[...,
                              [0, 2]] *= noisy_latents.size(4)  # x dimension
                bboxes_actual[...,
                              [1, 3]] *= noisy_latents.size(3)  # y dimension
                bboxes_actual = bboxes_actual.long(
                )  # Convert to long type for indexing

                # Step 1: Initialize the weights tensor with all ones
                re_weights = torch.ones_like(noisy_latents)
                # Step 2: Set higher weights for areas indicated by bboxes
                for i in range(cur_bs):
                    for j in range(num_frames):  # fig index
                        for k in range(fixed_num_objs):  # ins index
                            x1, y1, x2, y2 = bboxes_actual[i, j, k]
                            re_weights[i, :, j, y1:y2,
                                       x1:x2] = args.fg_reweight
                # Normalize the weights matrix so that its mean is 1
                re_weights = re_weights / re_weights.mean()
                loss = F.mse_loss(model_pred.float(),
                                  target.float(),
                                  reduction="none")
                loss = loss * re_weights
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item(
                ) / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints,
                                key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints
                                   ) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints
                                ) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[
                                    0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir,
                                                 f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if ((args.validation_prompt is not None)
                            and (global_step % args.validation_steps == 0)):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}.")
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(args.output_dir,
                                                    "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        edited_images = []
                        with torch.autocast(
                                str(accelerator.device).replace(":0", ""),
                                enabled=accelerator.mixed_precision == "fp16"):
                            boxes = batch['bboxes'][0]
                            video_cls_list = batch['seg_phrase'].view(
                                -1).tolist()
                            # Convert each integer to its corresponding class
                            video_cls_str_list = [
                                classes_vis[int_idx]
                                if int_idx < len(classes_vis) else ""
                                for int_idx in video_cls_list
                            ]
                            video_masks = batch['mask']
                            for val_img_idx in range(
                                    args.num_validation_images):
                                video_frames = pipeline(
                                    args.validation_prompt,
                                    seg_phrases=video_cls_str_list,
                                    video_masks=video_masks,
                                    num_frames=args.num_frames,
                                    width=args.width,
                                    height=args.height,
                                    bbox_prompt=boxes,
                                    num_inference_steps=50,
                                    guidance_scale=7,
                                    generator=generator,
                                ).frames
                                # draw bbox
                                frame_size = (args.width, args.height)
                                num_frames = args.num_frames

                                out_file = os.path.join(
                                    val_save_dir,
                                    f"step_{global_step}_val_img_{val_img_idx}.mp4",
                                )

                                for i in range(num_frames):
                                    img = Image.fromarray(video_frames[i])
                                    draw = ImageDraw.Draw(img)
                                    for bbox in batch['bboxes'][0][i].cpu(
                                    ).numpy()[:2]:
                                        x1, y1, x2, y2 = bbox
                                        top_left = (x1 * frame_size[0],
                                                    y1 * frame_size[1])
                                        bottom_right = (x2 * frame_size[0],
                                                        y2 * frame_size[1])
                                        draw.rectangle(
                                            [top_left, bottom_right],
                                            outline="red",
                                            width=1)
                                    video_frames[i] = np.array(img)
                                export_to_gif(video_frames, out_file, 8)

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = TextToVideoSDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
