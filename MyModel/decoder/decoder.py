import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .position_encoding import PositionalEncoding1D

EPS = 1e-8

def check_nan(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MaskModule(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.mlp = MLP(d_model, d_model, d_model, num_layers)
        
    def forward(self, embedding, query):
        '''
        embedding: [B T C]
        query: [B N C]
        '''
        query = self.mlp(query).permute(0,2,1) # [B C N]
        logits = torch.matmul(embedding, query) # [B T N]
        
        return logits

class QueryModule(nn.Module):
    def __init__(self, d_model, ffn_mult, nhead, dropout, activation, normalize_before):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.self_attn = SelfAttentionLayer(d_model, nhead, dropout, activation, normalize_before)
        self.feed_forward = FFNLayer(d_model, d_model*ffn_mult, dropout, activation, normalize_before)
        
    def forward(self, embedding, embedding_pos, query, query_pos, attn_mask):
        # check_nan('before cross attn', query)
        x = self.cross_attn(query, embedding, memory_mask=attn_mask, memory_key_padding_mask=None, pos=embedding_pos, query_pos=query_pos)
        # check_nan('after cross attn', x)
        x = self.self_attn(x, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos)
        # check_nan('after self attn', x)
        x = self.feed_forward(x)
        # check_nan('after ffn', x)
        
        return x
    
class MaskedTransformerDecoder(nn.Module):
    def __init__(self, num_queries, num_mlp_layers, num_layers, 
                 ffn_mult, d_model, nhead, dropout, activation, normalize_before, apply_mask):
        super().__init__()
        self.pe_layer = PositionalEncoding1D(d_model)
        self.initial_norm = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList([QueryModule(d_model, ffn_mult, nhead, dropout, activation, normalize_before) for _ in range(num_layers)])
        self.mask_module = MaskModule(d_model, num_mlp_layers)
        self.classifier = nn.Linear(d_model, 1)
        
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, d_model)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        self.num_queries = num_queries
        self.nhead = nhead
        
        self.apply_mask = apply_mask
      
    # 추후 필요하면 구현  
    def resample_mask(self, mask):
        return mask
        
    def mask_logit_to_attn_mask(self, mask, threshold=0.5):
        
        mask_prob = mask.sigmoid().clamp(min=EPS, max=1-EPS)
        mask_binary = (mask_prob < threshold).bool()
        mask_binary = mask_binary.unsqueeze(1) # [B 1 T N]
        mask_binary = mask_binary.repeat(1, self.nhead, 1, 1) # [B nhead T N]
        mask_binary = mask_binary.flatten(0,1) # [B*nhead T N]
        attn_mask = mask_binary.transpose(1,2) # [B*nhead N T]
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # Safeguard for nan
        
        return attn_mask
    
    def forward(self, embedding):

        embedding_pos = self.pe_layer(embedding)
        
        B, _, _ = embedding.shape
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        query = self.initial_norm(query)
        mask = self.mask_module(embedding, query)
        resampled_mask = self.resample_mask(mask)
        cls = self.classifier(query)
        
        preds_mask = []
        preds_mask_resampled = []
        preds_class = []
        preds_mask.append(mask)
        preds_mask_resampled.append(resampled_mask)
        preds_class.append(cls)
        
        for decoder_layer in self.decoder_layers:
            if self.apply_mask:
                attn_mask = self.mask_logit_to_attn_mask(resampled_mask, threshold=0.5)
            else:
                attn_mask == None
            query = decoder_layer(embedding, embedding_pos, query, query_pos, attn_mask)
            query = self.initial_norm(query)
            mask = self.mask_module(embedding, query)
            cls = self.classifier(query)
            resampled_mask = self.resample_mask(mask)
            preds_mask.append(mask)
            preds_mask_resampled.append(resampled_mask)
            preds_class.append(cls)
        
        return preds_mask, preds_class
    
    @torch.inference_mode
    def inference(self, embedding, theta=0.8):
        """
        Args:
            embedding: (1, T, C)
            query: ignored (re-initialized inside)
            theta: threshold for selecting active speakers
        Returns:
            pred: (1, T, S) where S is the number of selected speakers
        """

        embedding_pos = self.pe_layer(embedding)  # (1, T, C)
        B, T, _ = embedding.shape
        assert B == 1, "This inference method assumes batch size == 1"

        # Initialize learnable queries
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)      # (1, N, C)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (1, N, C)

        # Initialize first mask
        query = self.initial_norm(query)
        mask = self.mask_module(embedding, query)
        resampled_mask = self.resample_mask(mask)

        for decoder_layer in self.decoder_layers:
            attn_mask = self.mask_logit_to_attn_mask(resampled_mask, threshold=0.5)
            query = decoder_layer(embedding, embedding_pos, query, query_pos, attn_mask)
            query = self.initial_norm(query)
            mask = self.mask_module(embedding, query)         # (1, T, N)
            cls = self.classifier(query)                      # (1, N, 1)
            resampled_mask = self.resample_mask(mask)         # (1, T, N)

        cls_prob = cls.squeeze().sigmoid().clamp(min=EPS, max=1-EPS)  # (N)

        pred = mask[:, :, cls_prob > theta]  # keep batch dim: (1, T, S)
        pred = pred.squeeze()

        return pred

def build_decoder(num_queries, num_mlp_layers, num_layers, ffn_mult, d_model, nhead, dropout, activation, normalize_before, apply_mask):
    return MaskedTransformerDecoder(num_queries, num_mlp_layers, num_layers, ffn_mult, d_model, nhead, dropout, activation, normalize_before, apply_mask)