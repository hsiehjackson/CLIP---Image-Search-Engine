def vitb32_config(input_res, context_length, vocab_size):
    "ViT-B/32 configuration, uses 32x32 patches"
    return dict(
            embed_dim=512,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            image_resolution=input_res,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
    )

# Cell
def rn50_config(input_res, context_length, vocab_size):
    "ResNet50 configuration"
    return dict(
            embed_dim=512,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            image_resolution=input_res,
            vision_layers=(3,4,6,3),
            vision_width=64,
    )


def rn101_config(input_res, context_length, vocab_size):
    "ResNet101 configuration"
    return dict(
            embed_dim=512,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            image_resolution=input_res,
            vision_layers=(3,4,23,3),
            vision_width=64,
    )