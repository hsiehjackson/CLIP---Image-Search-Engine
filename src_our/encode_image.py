# python -m torch.distributed.launch --nproc_per_node 4 main.py 
import os
import argparse
import pdb

from clip_our import CLIP
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from PIL import Image
import matplotlib.pyplot as plt

import pickle
import glob
import math
from tqdm import tqdm

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class CLIPDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.image_dir = dataset_path
        self.data = list(glob.glob(os.path.join(self.image_dir,'*.jpg')))
        print(f'Load {len(self.data)} images')
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'path': path}
    
    def collate_fn(self, batch):
        image = torch.stack([b['image'] for b in batch])
        path = [b['path'] for b in batch]
        return {'image': image, 'path': str(path)}
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
def main(args):

    set_seed(args)
    torch.distributed.init_process_group(backend='nccl')
    print(f"Use Device-{args.local_rank} GPU! | World size {dist.get_world_size()}")
    
    # Setting image transform and tokenizer
    config = resolve_data_config({}, model=timm.create_model(args.pretrain_image, pretrained=True))
    config['crop_pct'] = 1
    transform = create_transform(**config)
    
    # dataset
    dataset = CLIPDataset(args.dataset_path, transform)
    sampler = DistributedSampler(dataset)
    iterator = DataLoader(dataset, 
                          batch_size=args.batch_size, 
                          num_workers=args.num_workers, 
                          sampler=sampler, 
                          pin_memory=True, 
                          collate_fn=dataset.collate_fn)
    
    print(f'GPU-{args.local_rank} Load dataset complete')
    
    # Model
    model = CLIP(             
         pretrain_text = args.pretrain_text,
         pretrain_image = args.pretrain_image,
    )
    ckpt = torch.load(args.load_model)
    state_dict = {k.partition('model.')[2]: v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
            
    torch.cuda.set_device(args.local_rank)     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print(f'GPU-{args.local_rank} Load model complete')
    
    
    pbar = tqdm(iterator, desc="Encode")
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        paths = batch['path']
        with torch.no_grad():
            images_embeddings = model.module.encode_image(images).cpu()
            torch.save({
                'path': paths,
                'emb': images_embeddings,
            }, os.path.join(args.emb_path, f'{args.local_rank}-{step:03d}-{args.batch_size}.pt'))
    
if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default='./models/model.ckpt', required=False, help='Choose your model checkpoint')
    parser.add_argument('--pretrain_text', default= 'bert-base-uncased', type=str, required=False)
    parser.add_argument('--pretrain_image', default='resnet50' , type=str, required=False)
    parser.add_argument('--dataset_path', default='./dataset/yfcc100m/data', help='Image (.jpg) path')
    parser.add_argument('--emb_path', default='./dataset/yfcc100m/emb', help='Image embedding save path')
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--num_workers', default=20, type=int, required=False)
    parser.add_argument('--batch_size', default=512, type=int, required=False)
    parser.add_argument('--seed', default=42, type=int, required=False)
    
    args = parser.parse_args()
    
    os.makedirs(args.emb_path, exist_ok=True)
    
    main(args)