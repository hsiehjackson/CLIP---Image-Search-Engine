import os
import argparse
import random

from dataset_yfcc100m import CLIPDataset, SampleDataset

import pytorch_lightning as pl
from pltrainer import PLTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.plugins import DeepSpeedPlugin

from tokenizer import ClipTokenizer
import torchvision.transforms as transforms
from PIL import Image

import clip

import warnings
warnings.filterwarnings("ignore")

def main(args):
    seed_everything(args.seed)
    
    # Tokenizer
    tokenizer = ClipTokenizer(
        bpe_path=args.bpe_path,
        context_length=77
    )
    print(f'Load Tokenizer with Vocab {tokenizer.vocab_size} Max Length {tokenizer.context_length} Complete')
    
    model, _ = clip.load(args.finetune_clip, jit=False)
    resolution = model.visual.input_resolution
        
    # Setting
    transform_sample = transforms.Compose([
            transforms.Resize(resolution, interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
    ])
    
    transform =  transforms.Compose([
            transform_sample,
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    
    # Dataset
    train_data = CLIPDataset(args.dataset_path, mode='train', transform=transform, tokenizer=tokenizer, 
                                max_text_length=tokenizer.context_length) 
    
    dev_data = CLIPDataset(args.dataset_path, mode='dev', transform=transform, tokenizer=tokenizer, 
                              max_text_length=tokenizer.context_length)
    
    train_sample = SampleDataset(args.dataset_path, mode='train', transform=transform, tokenizer=tokenizer, 
                                 max_text_length=tokenizer.context_length, 
                                sample_num=args.sample_num * args.num_gpus, transform_sample=transform_sample) 
    
    dev_sample = SampleDataset(args.dataset_path, mode='dev', transform=transform, tokenizer=tokenizer, 
                               max_text_length=tokenizer.context_length, 
                               sample_num=args.sample_num * args.num_gpus, transform_sample=transform_sample)
    
    print(f'Train: {len(train_data)} | Dev: {len(dev_data)} | Sample: {len(train_sample)}, {len(dev_sample)}')
    

    datasets = {
        'train': train_data,
        'valid': dev_data,
        'train_sample': train_sample,
        'valid_sample': dev_sample
    }
    

    # Logger
    tblogger = TensorBoardLogger(
        args.log_path, 
        default_hp_metric=False)
    
    
    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.model_path,
        filename= '{step:05d}-{valid_loss:.2f}',
        save_top_k=3,
        verbose=False,
        monitor='valid_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    pltrainer = PLTrainer(args, model, datasets)
    trainer = pl.Trainer(
        fast_dev_run=args.fast_dev_run,
        logger=[tblogger],
        max_steps=args.max_steps, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=1,
        gpus=args.num_gpus, 
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.25,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(pltrainer)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action='store_true')
        
    parser.add_argument('--dataset_path', default='dataset/yfcc100m', type=str, required=False)
    parser.add_argument('--bpe_path', default='src_openai_finetune/bpe_simple_vocab_16e6.txt.gz', type=str, required=False)
    parser.add_argument('--finetune_clip', default='ViT-B/32t', type=str, required=False, help='RN50, RN101, RN50x4, ViT-B/32')

    parser.add_argument('--model_path', default='models/', type=str, required=False)
    parser.add_argument('--log_path', default='logs/',type=str,required=False)
    
    parser.add_argument("--sample_num", default=1, type=int, required=False)
    parser.add_argument('--lr', default=5e-5, type=float, required=False)
    parser.add_argument('--warmup_steps', default=200, type=int, required=False)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, required=False)
    parser.add_argument("--accumulate_grad_batches",default=1, type=int, required=False)
    
    parser.add_argument("--num_workers", default=20, type=int, required=False)
    parser.add_argument("--num_gpus", default=1, type=int, required=False)
    parser.add_argument("--max_steps", default=10000, type=int, required=False)
    parser.add_argument("--batch_size", default=512, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)

    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path, exist_ok=True)
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
                        
    main(args)
