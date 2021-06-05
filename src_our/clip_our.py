import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import re
from clip_module import AttentionPool2d

from transformers import AutoConfig, AutoModel, AutoTokenizer
import timm


pretrain_text_model = {
    'bert-base-chinese'
}

pretrain_image_model = {
    'resnet50',
    'efficientnet_v2s',
    'vit_base_patch16_224_in21k'
    # ...timm.list_models(pretrained=True)    
}
 
    
class CLIP(nn.Module):
    def __init__(self,
             pretrain_text: str = None,
             pretrain_image: str = None,
             ):
        super().__init__()
        text_model = PretrainText(pretrain_text)
        image_model = PretrainImage(pretrain_image)
            
        self.model = nn.ModuleDict({
            'text': text_model,
            'image': image_model
        })
        
    

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, text, image):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_text, logits_per_image
    
    def inference(self, text, image):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()

        return logits_per_text, logits_per_image    
    
    def encode_image(self, image):
        if type(self.model) == nn.ModuleDict:
            return self.model['image'](image)
        else:
            return self.model.encode_image(image)
    
    def encode_text(self, text):
        if type(self.model) == nn.ModuleDict:
            return self.model['text'](text)
        else:
            return self.model.encode_text(text)

        
class PretrainText(nn.Module):
    def __init__(self, pretrain_model_name, embed_dim=512):
        super().__init__()
        print(f'Use text pretrained model: {pretrain_model_name}')
        self.transformer_width = AutoConfig.from_pretrained(pretrain_model_name).hidden_size
        self.pretrain_model = AutoModel.from_pretrained(pretrain_model_name)
        self.ln_final = nn.LayerNorm(self.transformer_width)
        self.projection = nn.Parameter(torch.empty(self.transformer_width, embed_dim))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        if self.projection is not None:
            nn.init.normal_(self.projection, std=self.transformer_width ** -0.5)
            
    def forward(self, x):
        x = self.pretrain_model(**x).last_hidden_state
        # x.shape = [batch_size, n_ctx, transformer.width]
        x = self.ln_final(x)
        # take features from the cls embedding (cls is the smallest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.projection
        return x


class PretrainImage(nn.Module):
    def __init__(self, pretrain_model_name, embed_dim=512):
        super().__init__()
        print(f'Use image pretrained model: {pretrain_model_name}')
        model = timm.create_model(pretrain_model_name, pretrained=True)
        config = model.default_cfg
        image_resolution = config['input_size'][1]
        isImageTransformer = re.search(r'it_', pretrain_model_name)
        
        self.imagenet_width = list(model.children())[-1].in_features
        self.pretrain_model = nn.Sequential(*list(model.children())[:-2])
        if not isImageTransformer:
            self.pretrain_model.add_module(
                'attnpool',
                AttentionPool2d(image_resolution // 32, 
                                embed_dim=self.imagenet_width, 
                                num_heads=self.imagenet_width // 16, 
                                output_dim=self.imagenet_width)
            )
            
        self.ln_final = nn.LayerNorm(self.imagenet_width)
        self.projection = nn.Parameter(torch.empty(self.imagenet_width, embed_dim))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        if self.projection is not None:
            nn.init.normal_(self.projection, std=self.imagenet_width ** -0.5)
            
    def forward(self, x):
        x = self.pretrain_model(x)
        
        # ConvNet
        if x.size() == 4:
            x = x.flatten(start_dim=1)
        
        # Transformer
        if x.size() == 3: 
            x = x[:, 0, :]
        
        # x.shape = [batch_size, imagenet_width]
        x = self.ln_final(x)
        # take features from the cls embedding (eot_token is the highest number in each sequence)
        x = x @ self.projection
        return x
        
if __name__ == '__main__':
    from clip_openai import vitb32_config
        
    model = CLIP(             
         pretrain_text = 'bert-base-uncased' ,
         pretrain_image = 'resnet50',
    )
    
    
    
    batch_size = 5
    text = torch.randint(0, 20000, (batch_size, 77))
    image = torch.rand(batch_size, 3, 224, 224)
    logits_per_text, logits_per_image = model(text, image)
    print(logits_per_text.shape)
    print(logits_per_image.shape)
    