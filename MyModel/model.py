import torch
import torch.nn as nn
import torch.nn.functional as F

class EELD(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x, length=None):
        x, length = self.encoder(x, length)
        preds_mask, preds_class = self.decoder(x)
        
        return preds_mask, preds_class, length
    
    @torch.inference_mode()
    def inference(self, x, length=None, theta=0.8):
        x, length = self.encoder(x, length)
        pred = self.decoder.inference(x, theta=theta)
        
        return pred, length
        