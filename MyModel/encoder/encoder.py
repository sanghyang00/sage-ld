import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from .components import _get_feature_extractor, _get_encoder
from conformer import Conformer

class Resampler(nn.Module):
    def __init__(self, hidden_dim, resampling_factor, mode='down'):
        super().__init__()
        assert mode in ['down', 'up']
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)
        if mode == 'down':
            self.conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=resampling_factor * 2 + 1,
                stride=resampling_factor,
                padding=resampling_factor
            )
        else:  # mode == 'up'
            self.conv = nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=resampling_factor * 2,
                stride=resampling_factor,
                padding=resampling_factor // 2,
                output_padding=resampling_factor % 2
            )
    
    def forward(self, x): # [B T C]
        x = self.norm(x)
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        
        return x

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class AttentivePooling(Module):
    def __init__(self, input_dim, down_factor=10):
        super().__init__()
        self.down_factor = down_factor
        self.attn = nn.Linear(input_dim, 1)
        
    def forward(self, x, length=None):
        B, T, C = x.size()
        
        pad_len = (self.down_factor - (T % self.down_factor)) % self.down_factor 
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        T_new = T + pad_len
        length = (length + pad_len) // self.down_factor  # 정확한 길이 계산
        
        x = x.view(B, T_new // self.down_factor, self.down_factor, C)  # (B, T/10, 10, C)
        
        attn_logits = self.attn(x)  # (B, T/10, 10, 1)
        attn_scores = F.softmax(attn_logits, dim=2)  # (B, T/10, 10, 1)
        
        weighted = x * attn_scores 
        pooled = weighted.sum(dim=2) # (B, T/10, 10)
        
        return pooled, length

class Wav2Vec2Model(Module):

    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        is_conformer: bool,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.is_conformer = is_conformer

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x) if self.is_conformer else self.encoder(x, lengths)
        
        return x, lengths

class Wav2Vec2ModelWithPooling(Module):

    def __init__(
        self,
        feature_extractor: Module,
        downsampler: Module,
        encoder: Module,
        is_conformer: bool,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsampler = downsampler
        self.encoder = encoder
        self.is_conformer = is_conformer

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        x, lengths = self.feature_extractor(waveforms, lengths)
        x, lengths = self.downsampler(x, lengths)
        x = self.encoder(x) if self.is_conformer else self.encoder(x, lengths)
        
        return x, lengths

def wav2vec2_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
) -> Wav2Vec2Model:
    
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = _get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )
    encoder = _get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    
    return Wav2Vec2Model(feature_extractor, encoder, is_conformer=False)

def build_pretrained_encoder(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    num_parameters: str = '300m',
    ckpt_path: str = None,
) -> Wav2Vec2Model:
    if num_parameters == '300m':
        model = wav2vec2_model(
            extractor_mode="layer_norm",
            extractor_conv_layer_config=None,
            extractor_conv_bias=True,
            encoder_embed_dim=1024,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=128,
            encoder_pos_conv_groups=16,
            encoder_num_layers=24,
            encoder_num_heads=16,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=4096,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=True,
            encoder_layer_drop=encoder_layer_drop,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        
        return model
        
    elif num_parameters == '1b':
        model = wav2vec2_model(
            extractor_mode="layer_norm",
            extractor_conv_layer_config=None,
            extractor_conv_bias=True,
            encoder_embed_dim=1280,
            encoder_projection_dropout=encoder_projection_dropout,
            encoder_pos_conv_kernel=128,
            encoder_pos_conv_groups=16,
            encoder_num_layers=48,
            encoder_num_heads=16,
            encoder_attention_dropout=encoder_attention_dropout,
            encoder_ff_interm_features=5120,
            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
            encoder_dropout=encoder_dropout,
            encoder_layer_norm_first=True,
            encoder_layer_drop=encoder_layer_drop,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        
        return model
        
    else:
        raise ValueError()
    
def build_plain_encoder(d_model=1024, nhead=16, nlayers=6, ffn_mult=4, is_conformer=False):
    
    extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = _get_feature_extractor(
        "layer_norm", extractor_conv_layer_config, True
    )
    
    if not is_conformer:
        encoder = _get_encoder(in_features=512, 
                            embed_dim=d_model, 
                            dropout_input=0.0, 
                            pos_conv_kernel=d_model//nhead, 
                            pos_conv_groups=nhead,
                            num_layers=nlayers,
                            num_heads=nhead,
                            attention_dropout=0.0,
                            ff_interm_features=int(d_model*ffn_mult),
                            ff_interm_dropout=0.0,
                            dropout=0.0,
                            layer_norm_first=True,
                            layer_drop=0.0)
    
    else:
        encoder = nn.Sequential(nn.Linear(in_features=512, out_features=d_model),
                                Conformer(dim = d_model, 
                                        depth = nlayers, 
                                        dim_head = d_model//nhead, 
                                        heads = nhead, 
                                        ff_mult = ffn_mult, 
                                        conv_expansion_factor = 2, 
                                        conv_kernel_size = 31, 
                                        attn_dropout = 0., 
                                        ff_dropout = 0., 
                                        conv_dropout = 0.))
    
    return Wav2Vec2Model(feature_extractor, encoder, is_conformer=True)

def build_plain_encoder_with_pooling(d_model=1024, nhead=16, nlayers=6, ffn_mult=4, down_factor=10, is_conformer=False):
    
    extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = _get_feature_extractor(
        "layer_norm", extractor_conv_layer_config, True
    )
    
    downsampler = AttentivePooling(input_dim=512,
                                   down_factor=down_factor)
    
    if not is_conformer:
        encoder = _get_encoder(in_features=512, 
                            embed_dim=d_model, 
                            dropout_input=0.0, 
                            pos_conv_kernel=d_model//nhead, 
                            pos_conv_groups=nhead,
                            num_layers=nlayers,
                            num_heads=nhead,
                            attention_dropout=0.0,
                            ff_interm_features=int(d_model*ffn_mult),
                            ff_interm_dropout=0.0,
                            dropout=0.0,
                            layer_norm_first=True,
                            layer_drop=0.0)
    
    else:
        encoder = nn.Sequential(nn.Linear(in_features=512, out_features=d_model),
                                Conformer(dim = d_model, 
                                        depth = nlayers, 
                                        dim_head = d_model//nhead, 
                                        heads = nhead, 
                                        ff_mult = ffn_mult, 
                                        conv_expansion_factor = 2, 
                                        conv_kernel_size = 31, 
                                        attn_dropout = 0., 
                                        ff_dropout = 0., 
                                        conv_dropout = 0.))
    
    return Wav2Vec2ModelWithPooling(feature_extractor, downsampler, encoder, is_conformer=True)
