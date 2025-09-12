import os, re, sys, json, argparse, wandb
import numpy as np, pandas as pd
import librosa, torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm
from MyModel.encoder.encoder import build_plain_encoder
from MyModel.decoder.decoder import build_decoder
from MyModel.model import EELD
from MyModel.utils.utils import *
from dataset import AfricanSoapDataset, DISPLACEDataset

import pdb

MINIBATCH = 0
NUM_ACCUMULATION_STEPS = 0
CURRENT_STEP = 0
SPECIFIC_STEP = 0
NUM_TOTAL_STEPS = 5000
SAVE_DIR = None
torch.autograd.set_detect_anomaly(True)

def collate_fn(batch):
    inputs = []
    input_lengths = []
    lang_masks = []

    for (audio, lang_mask) in batch:
    
        if lang_mask.dim()==0:
            lang_mask = lang_mask.unsqueeze(0)

        inputs.append(audio)
        lang_masks.append(lang_mask)
        input_lengths.append(audio.shape[-1])

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    lang_masks = nn.utils.rnn.pad_sequence(lang_masks, batch_first=True)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    
    return inputs, lang_masks, input_lengths

def match_labels(pred: torch.Tensor, target: torch.Tensor, num_classes: int = None):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    if num_classes is None:
        num_classes = int(max(pred.max(), target.max())) + 1

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred_np, target_np):
        conf[p, t] += 1

    row_ind, col_ind = linear_sum_assignment(-conf)
    label_map = {int(row): int(col) for row, col in zip(row_ind, col_ind)}

    aligned_pred_np = np.vectorize(label_map.get)(pred_np)
    aligned_pred = torch.tensor(aligned_pred_np, device=pred.device)
    
    der_results = compute_der(aligned_pred, target)

    return der_results, aligned_pred, label_map

def main():

    global NUM_ACCUMULATION_STEPS
    global MINIBATCH
    global NUM_TOTAL_STEPS
    global SAVE_DIR
    global REMOVE_MUTE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32) # Modified
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--total_step', type=int, default=10000)
    parser.add_argument('--init_lr', type=float, default=5e-6)
    parser.add_argument('--peak_lr', type=float, default=1e-5)
    parser.add_argument('--final_lr', type=float, default=5e-6)
    parser.add_argument('--ft_dataset', type=str, default='africansoap')
    parser.add_argument('--ft_dataset_path', type=str, default='africansoap')
    parser.add_argument('--ft_dataset_sample_path', type=str, default='africansoap')
    parser.add_argument('--ckpt_path', type=str, default='finetuned_ckpts/best_model.pt')
    parser.add_argument('--is_conformer', type=str2bool, default='True')
    parser.add_argument('--masked_attn', type=str2bool, default='True')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_query', type=int, default=5)
    
    args = parser.parse_args()
    encoder = build_plain_encoder(d_model=args.hidden_dim, nhead=args.nhead, nlayers=args.num_layers, ffn_mult=4, is_conformer=args.is_conformer)
    decoder = build_decoder(num_queries=args.num_query, num_mlp_layers=3, num_layers=args.num_layers, ffn_mult=4, 
                            d_model=args.hidden_dim, nhead=args.nhead, dropout=0.0, activation='relu', normalize_before=True, apply_mask=args.masked_attn)
    model = EELD(encoder, decoder)
    
    print('Loading pretrained parameters...')
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 16
    
    if args.ft_dataset == 'displace_dev':
        val_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Dev', mode='test')
    elif args.ft_dataset == 'displace_eval':
        val_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Eval', mode='test')
    else:
        val_dataset = AfricanSoapDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, mode='test')

    test_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)
    
    model = model.to(device)

    data = []
    for i, (inputs, lang_masks, input_lengths) in tqdm(enumerate(test_loader), total=len(test_loader)):
  
        inputs, lang_masks, input_lengths = inputs.to(device), lang_masks.to(device), input_lengths.to(device)
        
        pred_logits, _ = model.inference(inputs, input_lengths, theta=0.8)
        if pred_logits.numel() != 0:
            pred_mask = pred_logits.argmax(dim=-1).squeeze()
        gt_mask = lang_masks.squeeze()

        if gt_mask.shape == pred_mask.shape:
            der, aligned, mapping = match_labels(pred_mask, gt_mask)

            error_rate, fa, miss, conf = der['DER'], der['FA'] / der['Total'], der['Miss'] / der['Total'], der['Confusion'] / der['Total']
            
            data.append([error_rate, fa, miss, conf])
        else:
            continue

    df = pd.DataFrame(data, columns=["DER", "False Alarm", "Miss", "Confusion"])

    df.to_csv('results.csv', index=False)
    
if __name__=='__main__':
    main()