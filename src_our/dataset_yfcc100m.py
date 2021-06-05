import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

import ftfy
import html
import regex as re
import random


class CLIPDataset(Dataset):
    def __init__(self, dataset_dir, mode, max_text_length, transform=None, tokenizer=None, ):
        self.dataset_dir = dataset_dir
        df = pd.read_csv(os.path.join(self.dataset_dir, f'{mode}.tsv'), sep='\t')
        self.data = df[['pid', 'description']].values.tolist()
        self.transform = transform
        self.tokenizer = tokenizer
        self.padding_idx = 0
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        # Image
        image = Image.open(os.path.join(self.dataset_dir, 'data', f'{self.data[idx][0]}.jpg')).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Text
        text = self.data[idx][1]
        if self.tokenizer:
            text = self.tokenizer(text, return_tensors="pt")['input_ids'].squeeze().tolist()
            
        return {
            'image': image,
            'text': text
        }
    
    def collate_fn(self, batch):
        image = torch.stack([b['image'] for b in batch])
        max_length = min(max([len(b['text']) for b in batch]), self.max_text_length)
        token = torch.LongTensor([self.tokens_pad_to_len(b['text'], max_length) for b in batch])
        mask = torch.LongTensor([self.masks_pad_to_len(b['text'], max_length) for b in batch])
        return {
            'image': image,
            'text': {
                'input_ids': token,
                'attention_mask': mask
            }  
        }
        

    def tokens_pad_to_len(self, seq, to_len):
        return seq[:to_len] + [self.padding_idx] * max(0, to_len - len(seq))
    
    def masks_pad_to_len(self, seq, to_len):
        return [1] * len(seq[:to_len]) + [0] * max(0, to_len - len(seq))
        

class SampleDataset(Dataset):
    def __init__(self, dataset_dir, mode, max_text_length, 
                 transform=None, tokenizer=None, sample_num=None, transform_sample= None):
        self.dataset_dir = dataset_dir
        df = pd.read_csv(os.path.join(self.dataset_dir, f'{mode}.tsv'), sep='\t')
        self.data = random.sample(df[['pid', 'description']].values.tolist(), sample_num * 8)
        self.transform = transform
        self.transform_sample = transform_sample
        self.tokenizer = tokenizer
        self.padding_idx = 0
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        # Image
        image_raw = Image.open(os.path.join(self.dataset_dir, 'data', f'{self.data[idx][0]}.jpg')).convert('RGB')
        if self.transform:
            image = self.transform(image_raw)
        
        if self.transform_sample:
            image_raw = self.transform_sample(image_raw)
        
        # Text
        text_raw = self.data[idx][1]
        if self.tokenizer:
            text = self.tokenizer(text_raw, return_tensors="pt")['input_ids'].squeeze().tolist()
            
        return {
            'image': image,
            'text': text,
            'text_raw': text_raw,
            'image_raw': image_raw
        }
    
    def collate_fn(self, batch):
        image = torch.stack([b['image'] for b in batch])
        max_length = min(max([len(b['text']) for b in batch]), self.max_text_length)
        token = torch.LongTensor([self.tokens_pad_to_len(b['text'], max_length) for b in batch])
        mask = torch.LongTensor([self.masks_pad_to_len(b['text'], max_length) for b in batch])
        text_raw = [b['text_raw'] for b in batch]
        image_raw = torch.stack([b['image_raw'] for b in batch])
        return {
            'image': image,
            'text': {
                'input_ids': token,
                'attention_mask': mask
            },
            'text_raw': text_raw,
            'image_raw': image_raw
        }
        

    def tokens_pad_to_len(self, seq, to_len):
        return seq[:to_len] + [self.padding_idx] * max(0, to_len - len(seq))
    
    def masks_pad_to_len(self, seq, to_len):
        return [1] * len(seq[:to_len]) + [0] * max(0, to_len - len(seq))
        

if __name__ == '__main__':

    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model('resnet18', pretrained=True)
    config = resolve_data_config({}, model=model)
    config['crop_pct'] = 1
    transform = create_transform(**config)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/nsml/workspace/sharing/jutopia_for_tw_team_iu_shared_fs/models/bert-base-uncased')
    # tokenizer.add_special_tokens({'additional_special_tokens':['[SUM]', '[END]']})
    print(f'Load Tokenizer with Vocab {len(tokenizer)} Complete')
    
    dataset = StickerDataset(
        dataset_dir='/home/nsml/workspace/CLIP/src/dataset/yfcc100m',
        mode='train',
        max_text_length=tokenizer.model_max_length,
        transform=transform,
        tokenizer=tokenizer)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
    
    batch = next(iter(dataloader))
    print(batch['image'].shape)
    print(batch['text']['input_ids'].shape, batch['text']['attention_mask'].shape)