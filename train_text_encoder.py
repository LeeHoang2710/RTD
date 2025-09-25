'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import json
import os
import pickle
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Set
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from loader import CIRRDataset, build_loader_RTD 
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
from models import Phi, build_text_encoders_RTD
from utils import extract_image_features, extract_pseudo_tokens_with_phi, contrastive_loss
from validate import cirr_compute_val_metrics

import sys
import transformers
from transformers import get_scheduler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger


logger = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--output_dir", default="trained_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--logging_dir", default="logs", type=str, help="tensorboard logs will saved here")
    parser.add_argument("--cache_dir", default="./hf_models", type=str,
                        help="Path to model cache folder")
    parser.add_argument("--report_to", default="tensorboard", type=str, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--clip_model_name", default="giga", type=str,
                        help="CLIP model to use, e.g 'large', 'giga'")
    parser.add_argument("--cirr_dataset_path", type=str, help="Path to CIRR dataset", required=True)
    parser.add_argument("--keywords_path", type=str, help="Path to keywords json file")
    parser.add_argument("--resume", default=None, type=str, help="Path to pretrained ckpt")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--max_train_steps", type=int, default=50000, help="Total number of training steps to perform")
    parser.add_argument("--phi_dropout", default=0.5, type=float, help="Dropout probability for the phi network")
    parser.add_argument("--l2_normalize", action="store_true", help="Whether or not to use l2 normalization")
    parser.add_argument("--batch_size", default=256, type=int, help="Training batch size for text encoder")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")
    parser.add_argument("--mixed_precision", default=None, type=str, choices=["no", "fp16", "bf16"], help="mixed precision")
    parser.add_argument("--validation_steps", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--checkpointing_steps", default=None, type=int, help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducibility")
    
    #### For RTD
    parser.add_argument("--phi_checkpoint", default=None, type=str, help="path for loading pre-trained phi checkpoint")
    parser.add_argument("--noise_scale", default=0.5, type=float, help="Scale of noise distribution")
    parser.add_argument("--tau", default=0.07, type=float, help="temperature")
    parser.add_argument("--caption_dir", default="/path/to/your_dataset/LLM_triplets.json", type=str,
                        help="The caption directory")
    ####
    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def save_text_encoder(name: str, cur_epoch: int, model_to_save, training_path: Path) -> None:
    """
    Save the weights of Phi during training
    """
    models_path = os.path.join(training_path, "checkpoints")
    os.makedirs(models_path, exist_ok=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, os.path.join(models_path, f'{name}.pt'))

def train_text_encoder(args):
    # We are going to use the pre-extracted clip image features. so we do not need image_encoder anymore.
    ### init accelerator here
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    
    #### RTD: init clip encoders 
    #### Initialize  (image_encoder, text_encoder) and another trainable text_encoder
    image_encoder, clip_preprocess, text_encoder, text_encoder_train, tokenizer = build_text_encoders_RTD(args)

    
    ### Define the phi model
    phi = Phi(input_dim=text_encoder.config.projection_dim,
                    hidden_dim=text_encoder.config.projection_dim * 4,
                    output_dim=text_encoder.config.hidden_size, dropout=args.phi_dropout)

    
    if args.phi_checkpoint:
        phi.load_state_dict(
                torch.load(args.phi_checkpoint, map_location=accelerator.device)[
                phi.__class__.__name__])
        
    if args.resume:
        text_encoder_train.load_state_dict(
                torch.load(args.resume, map_location=accelerator.device)[
                text_encoder_train.__class__.__name__])
        
    ### GPU handling
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    phi.to(accelerator.device, dtype=weight_dtype)
    
    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    phi.requires_grad_(False)
    phi.eval()


    ### Define the train datasets
    print('pytorch loader')
    train_dataset = build_loader_RTD(args, tokenizer, accelerator)

    ## evaluator
    if accelerator.is_main_process:
        ## Define CIRR validation set
        cirr_relative_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'relative', clip_preprocess)
        cirr_classic_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'classic', clip_preprocess)

        # Extract the features for the CIRR validation set
        cirr_val_index_features, cirr_val_index_names = extract_image_features(cirr_classic_val_dataset, image_encoder)

    # Define the optimizer, the loss and the grad scaler
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

    optimizer = optimizer_cls(text_encoder_train.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes,
    )

    text_encoder_train, optimizer, lr_scheduler, train_dataset = accelerator.prepare(
            text_encoder_train, optimizer, lr_scheduler, train_dataset
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("zeroshot-cir", config=vars(args))

    # Start with the training loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total steps = {args.max_train_steps}")

    text_encoder_train.train()

    train_loss = 0.0
    global_step = 0
    best_recall = -1

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    while True:
        for idx, (concat_tokens, target_tokens, reference_tokens, special_tokens) in enumerate(train_dataset):
            #### T_{r+c}
            concat_tokens = concat_tokens.to(accelerator.device)
            #### T_t
            target_tokens = target_tokens.to(accelerator.device)
            #### T_r:  special_tokens: a photo of [$],  reference_tokens: a photo of [T_r]
            special_tokens = special_tokens.to(accelerator.device)
            reference_tokens = reference_tokens.to(accelerator.device)
            
            #### Use refined batch sampling
            concat_tokens = torch.cat((concat_tokens, special_tokens), dim=0).to(accelerator.device)
            target_tokens = torch.cat((target_tokens, reference_tokens), dim=0).to(accelerator.device)
            
            #### extract target features
            target_features = text_encoder(input_ids=target_tokens)
            target_text_embeddings, _ = target_features.text_embeds, target_features.last_hidden_state
            
            #### For refined concatenatenation scheme, extract reference features using frozen text encoder
            reference_features = text_encoder(input_ids=reference_tokens)
            reference_text_embeddings, _ = reference_features.text_embeds, reference_features.last_hidden_state
            
            #### add noise and use phi for refined concatenatenation scheme.
            input_features = reference_text_embeddings.clone()
            input_features += args.noise_scale * torch.rand(input_features.shape[0], device=input_features.device).unsqueeze(-1) * torch.randn(input_features.shape, device=input_features.device)
            estimated_token_embeddings = phi(input_features)
            
            #### Consider refined batch sampling: T_{r+c} (a photo of [$] that [T_c]) and T_r (a photo of [$])
            estimated_token_embeddings = torch.cat((estimated_token_embeddings, estimated_token_embeddings), dim=0).to(accelerator.device)
            
            #### extract pseudo tokens [$]
            concat_text_embeddings, _ = encode_with_pseudo_tokens_HF(text_encoder_train.module, concat_tokens, estimated_token_embeddings, return_last_states=True)
            target_text_embeddings = target_text_embeddings.to(concat_text_embeddings.dtype)
            
            loss = contrastive_loss(target_text_embeddings,concat_text_embeddings,args.tau)
            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            
            # Backpropagation
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm is not None:
                accelerator.clip_grad_norm_(text_encoder_train.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train/train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                accelerator.log({'train/lr': lr_scheduler.get_last_lr()[0]}, step=global_step)
                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        logger.info(f"model saving... step: {global_step}")
                        save_text_encoder(f"text_encoder_{global_step:09}", global_step, accelerator.unwrap_model(text_encoder_train), args.output_dir)

                if global_step % args.validation_steps == 0 or global_step == 50:
                    if accelerator.is_main_process:
                        logger.info(f"evaluate model... step: {global_step}")

                        # Extract the pseudo tokens for the CIRR validation set using Phi
                        cirr_val_pseudo_tokens, cirr_val_ref_names_list = extract_pseudo_tokens_with_phi(image_encoder, phi,
                                                                                                         cirr_relative_val_dataset, args)
                        cirr_val_pseudo_tokens = cirr_val_pseudo_tokens.to(accelerator.device)

                        # Compute the CIRR validation metrics
                        cirr_results_dict = cirr_compute_val_metrics(cirr_relative_val_dataset, accelerator.unwrap_model(text_encoder_train).eval(),
                                                                     cirr_val_index_features, cirr_val_index_names,
                                                                     cirr_val_ref_names_list, cirr_val_pseudo_tokens)
                        check_list = ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_recall_at10', 'cirr_recall_at50' ] 

                        for check_key in check_list:
                            accelerator.log({f"validate/{check_key}": cirr_results_dict[check_key]}, step=global_step)
                        print(json.dumps(cirr_results_dict, indent=4))
                        
                        check_list = ['cirr_recall_at1', 'cirr_recall_at5', 'cirr_recall_at10', 'cirr_recall_at50', "cirr_group_recall_at1", "cirr_group_recall_at2", "cirr_group_recall_at3" ]
                        log_stats = {f"validate/{check_key}": cirr_results_dict[check_key] for check_key in check_list}

                        # Adding 'step' key with global_step value
                        log_stats['step'] = global_step

                        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                                f.write(json.dumps(log_stats) + "\n")

                        # Save the best model.
                        if args.checkpointing_steps:
                            if cirr_results_dict['cirr_recall_at1'] > best_recall:
                                best_recall = cirr_results_dict['cirr_recall_at1']
                                logger.info(f"best model saving... step: {global_step}")
                                save_text_encoder(f"text_encoder_best", global_step, accelerator.unwrap_model(text_encoder_train), args.output_dir)
                        
                        
            if global_step >= args.max_train_steps:
                    sys.exit("Reached the maximum number of training steps, exiting.")


if __name__ == '__main__':
    args = parse_args()
    train_text_encoder(args)
