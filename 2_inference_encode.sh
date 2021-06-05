# Encode image with multi-GPU
python -m torch.distributed.launch --nproc_per_node 2 src_our/encode_image.py
# python -m torch.distributed.launch --nproc_per_node 2 src_openai_finetune/encode_image.py
# python -m torch.distributed.launch --nproc_per_node 2 src_openai_pretrain/encode_image.py