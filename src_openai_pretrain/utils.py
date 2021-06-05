import json
import os
import io
import pickle
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint


import PIL.Image
from torchvision.transforms import ToTensor

import matplotlib as mpl
import matplotlib.pyplot as plt


import clip

def convert_16to32(model):
    for p in model.parameters():
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 
        

def convert_32to16(model):
    clip.model.convert_weights(model)

        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def plot_similarity(similarity, texts_raw, images_raw, filepath):
    
    similarity = similarity.astype(np.float32)
    images_raw = images_raw.float()
    count = len(texts_raw)
    
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), texts_raw, fontsize=12)
    plt.xticks([])
    for i, image in enumerate(images_raw):
        plt.imshow(image.permute(1, 2, 0), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])
    plt.show()
    plt.savefig(filepath, bbox_inches="tight")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image
    