# CLIP - Image Search Engine
In this project, you will go through CLIP model training and build a simple image search engine. Model traing can use original **CLIP** or **pretrained language model** and **pretrained imageNet models**.

![](https://i.imgur.com/xercGuw.png)

---
## Annoucement
* My framework is built with [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).

---
## Folder Structure
```
CLIP - Image Search Engine
├── dataset/ - dataset with images and descriptions
│   └── yfcc100m/
├── package/ - requirement scripts
├── models/ - model checkpoints after training
├── logs/ - tensorboard logs after training
├── src_openai_finetune/ - python code
├── src_openai_pretrain/ - python code
└── src_our/ - python code
```
---
## Install Requirements
* Create your own conda environment
```
conda create --name clip python=3.6
```
* Install Package (conda/pip, torch, apex...)
```
bash 0_install.sh
```
---
## Data Preparation
> YFCC100M Dataset
* Go to ```dataset/yfcc100m/```
* Follow ```README.md``` instructions
---
## Modeling
> OpenAI CLIP Finetune (option-1)
```
bash 1_finetune_openai.sh
```
* --dataset_path=dataset/yfcc100m
* --finetune_clip=ViT-B/32|RN50|RN101|RN50x4
> OpenAI CLIP Pretrain (option-2)
```
bash 1_pretrain_openai.sh
```
* --dataset_path=dataset/yfcc100m
* --pretrain_clip=ViT-B/32|RN50|RN101
* --text_length=77
* --image_resolution=224
> Our CLIP Pretrain (option-3)
```
bash 1_pretrain_our.sh
```
* --dataset_path=dataset/yfcc100m
* --pretrain_text=bert-base-uncased (models in [huggingface](https://huggingface.co/models))
* --pretrain_image=resnet50 (models in [pytorch-image-models](https://rwightman.github.io/pytorch-image-models/))
---
## Inference
> Encode image embedding
```
bash 2_inference_encode.sh

# image embedding folder structure
dataset/yfcc100m/emb/
├──gpu_id-step-batch_size.ckpt
├──0-000-512.ckpt
│  └──{'path': str ('list'), 'emb': torch.FloatTensor }
├──0-001-512.ckpt
└──0-002-512.ckpt
```
* encode_image.py
    * distributed (**multi-GPU**) encode images to embeddings
    * --dataset_path=dataset/yfcc100m/data
    * --load_model=models/model.ckpt
    * --emb_path=dataset/yfcc100m/emb
> Image Search Engine
```
bash 3_inference_search.sh

# Search Inferface
# Search: Hello World -> `Hello World.png`
# Search: iPad  -> `iPad.png`
```
* search.py
    * --emb_path=dataset/yfcc100m/emb
    * --load_model=models/model.ckpt
* searcher.py 
    * how to calculate cosine similarity to find topk
    * Return topk images
---
## Reference
* [OpenAI/CLIP](https://github.com/OpenAI/CLIP)
* [KeremTurgutlu/self_supervised](https://github.com/KeremTurgutlu/self_supervised/tree/fastai_update)
* [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [huggingface/transformers](https://github.com/huggingface/transformers)
* [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
