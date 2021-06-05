import os
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import plot_similarity

from deepspeed.ops.adam import FusedAdam

class PLTrainer(pl.LightningModule):
    def __init__(self, args, model, datasets):
        super().__init__()

        self.save_hyperparameters(args)
        
        self.model = model

        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']
        self.train_sample = datasets['train_sample']
        self.valid_sample = datasets['valid_sample']
    

    def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,  betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
        optimizer = FusedAdam(self.parameters(), lr=self.hparams.lr,  betas=(0.9,0.98), eps=1e-6, weight_decay=0.2)
        return {
            'optimizer': optimizer,
        }


    def criterion(self, logits_per_text, logits_per_image):
        label = torch.arange(len(logits_per_text)).to(logits_per_text.device)
        loss_text  = F.cross_entropy(logits_per_text, label)
        loss_image = F.cross_entropy(logits_per_image, label)
        acc_text = (torch.argmax(logits_per_text, dim=1) == label).sum().item() / self.hparams.batch_size
        acc_image = (torch.argmax(logits_per_image, dim=1) == label).sum().item() / self.hparams.batch_size 
        return loss_text, loss_image, acc_text, acc_image

        
    def training_step(self, batch, batch_idx):
        texts, images = batch['text'], batch['image']
        logits_per_text, logits_per_image = self.model(texts, images)
        loss_text, loss_image, acc_text, acc_image = self.criterion(logits_per_text, logits_per_image)
        loss = (loss_text + loss_image) / 2
        acc = (acc_text + acc_image) / 2
        
        logs_bar = {
            'loss': loss,
            'acc': acc
        }
        
        for k, v in logs_bar.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            
        logs_logger = {
            'train_loss_text': loss_text, 
            'train_loss_image': loss_image,
            'train_loss': loss,
            'train_acc_text': acc_text,
            'train_acc_image': acc_image,
            'train_acc': acc
        }

        
        for k, v in logs_logger.items():
            self.log(k, v, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            texts, images = batch['text'], batch['image']
            logits_per_text, logits_per_image = self.model(texts, images)
            loss_text, loss_image, acc_text, acc_image = self.criterion(logits_per_text, logits_per_image)
            
            output = {
                'loss_text': loss_text, 
                'loss_image': loss_image, 
                'acc_text': acc_text,
                'acc_image': acc_image,
            }

            return output
        
        elif dataloader_idx == 1 or dataloader_idx == 2:
            texts, images = batch['text'], batch['image']
            logits_per_text, _ = self.model.inference(texts, images)    
            logits_per_text = logits_per_text.cpu().numpy()
            filename = f'train-{self.global_rank}-{batch_idx}.png' if dataloader_idx == 1 else f'valid-{self.global_rank}-{batch_idx}.png'
            image = plot_similarity(logits_per_text, 
                            texts_raw=batch['text_raw'], 
                            images_raw=batch['image_raw'].cpu(),
                            filepath=os.path.join(self.hparams.log_path, filename))
            
            output = {
                'image': image
            }
                
            return output
    

    def validation_epoch_end(self, outputs):

        loss_text = torch.stack([output['loss_text'] for output in outputs[0]]).mean().item()
        loss_image = torch.stack([output['loss_image'] for output in outputs[0]]).mean().item()
        acc_text = np.mean([output['acc_text'] for output in outputs[0]])
        acc_image = np.mean([output['acc_image'] for output in outputs[0]])
        
        
        logs_logger = {
            'valid_loss_text': loss_text, 
            'valid_loss_image': loss_image, 
            'valid_loss': (loss_text + loss_image) / 2,
            'valid_acc_text': acc_text,
            'valid_acc_image': acc_image,
            'valid_acc':  (acc_text + acc_image) / 2,
        }
        

        for k, v in logs_logger.items():
            self.log(k, v, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        train_image = [output['image'] for output in outputs[1]]
        valid_image = [output['image'] for output in outputs[2]]
        
        for i, (t,v) in enumerate(zip(train_image, valid_image)):
            self.logger[0].experiment.add_image(f'train-{self.global_rank}-{i}', t, self.global_step)
            self.logger[0].experiment.add_image(f'valid-{self.global_rank}-{i}', v, self.global_step)

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.train_dataset.collate_fn)
    
    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.valid_dataset.collate_fn)
        train_sampler = DataLoader(self.train_sample, batch_size=8, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.train_sample.collate_fn)
        valid_sampler = DataLoader(self.valid_sample, batch_size=8, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=self.valid_sample.collate_fn)
        return [valid_dataloader, train_sampler, valid_sampler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items