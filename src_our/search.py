import os
import argparse
import torch
import pdb

from clip_our import CLIP
from transformers import AutoTokenizer, AutoConfig

from PIL import Image
import matplotlib.pyplot as plt

import pickle
import glob
import math
from tqdm import tqdm
from ast import literal_eval

from searcher import Searcher


def encode_texts(text, model, tokenizer, device):
    text = tokenizer(text, return_tensors="pt")
    text = text.to(device)
    text_embedding = model.encode_text(text)
    return text_embedding

def readimage(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def plot(query, hits):
    num = len(hits)
    
    plt.figure(figsize=(20, 14))
    for i, hit in enumerate(hits):
        image = readimage(hit['path'])
        plt.subplot(5, math.ceil(num / 5), i+1)
        plt.imshow(image)
        plt.title(f"pid:{hit['pid']}\nscore:{float(hit['score']):.3f}", fontsize=12)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{query}.png',bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()

def main(args):
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Setting tokenizer
    config = AutoConfig.from_pretrained(args.pretrain_text)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_text)
    print(f'Load Tokenizer with Vocab {len(tokenizer)} Max Length {config.max_position_embeddings} Complete')
    
    
    # Model
    model = CLIP(             
         pretrain_text = args.pretrain_text,
         pretrain_image = args.pretrain_image,
    )
    ckpt = torch.load(args.load_model)
    state_dict = {k.partition('model.')[2]: v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    
    # Image Embedding
    paths = []
    embs = torch.tensor([])
    pbar = tqdm(list(glob.glob(os.path.join(args.emb_path, '*.pt')))[:50])
    for emb_file in pbar:
        ckpt = torch.load(emb_file)
        paths += literal_eval(ckpt['path'])
        embs = torch.cat((embs, ckpt['emb']),dim=0)
        pbar.set_postfix({'num_images': str(len(paths))})
    
    # Use GPU
    embs = embs.to(device)
    print('Path example:', paths[0])
    print('Embedding example:', embs[0].shape)
    
    searcher = Searcher(embs, paths)
    
    # Search
    while(True):
        word = input('Search words:')
        word = word.strip()
        if len(word) > 0:
            word_embedding = encode_texts(word, model, tokenizer, device)
            hits = searcher.search(word_embedding, top_k=[args.topk])[0]
            plot(word, hits)
            
    
if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default='models/model.ckpt', required=False)
    parser.add_argument('--pretrain_text', default= 'bert-base-uncased', type=str, required=False)
    parser.add_argument('--pretrain_image', default='resnet50' , type=str, required=False)
    parser.add_argument('--emb_path', default='./dataset/yfcc100m/emb')
    parser.add_argument('--topk', default=25, type=int)
    args = parser.parse_args()
    
    
    main(args)