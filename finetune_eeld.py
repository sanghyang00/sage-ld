import os, re, sys, json, argparse, wandb
import numpy as np, pandas as pd
import librosa, torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scheduler import TriStageLRScheduler
from MyModel.encoder.encoder import build_plain_encoder
from MyModel.decoder.decoder import build_decoder
from MyModel.model import EELD
from MyModel.utils.matcher import HungarianMatcher
from MyModel.utils.losses import EELDLoss
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

def train_and_validate(model, matcher, train_loader, val_loader, criterion, optimizer, device, scheduler=None):
    global NUM_ACCUMULATION_STEPS
    global CURRENT_STEP
    global SPECIFIC_STEP
    global NUM_TOTAL_STEPS
    global SAVE_DIR 
    global REMOVE_MUTE

    model.train()

    running_loss = 0.0
    running_loss_dia = 0.0
    running_loss_dice = 0.0
    running_loss_cls = 0.0
    step_loss = 0.0
    step_loss_dia = 0.0
    step_loss_dice = 0.0
    step_loss_cls = 0.0

    for i, (inputs, lang_masks, input_lengths) in enumerate(train_loader):
        inputs, lang_masks, input_lengths = inputs.to(device), lang_masks.to(device), input_lengths.to(device)
        
        preds_mask, preds_class, input_lengths = model(inputs, input_lengths)
        
        B = input_lengths.size(0)
        T = input_lengths.max()
        loss_seq = torch.arange(T, device=input_lengths.device)
        loss_masks = loss_seq.unsqueeze(0).expand(B, T)
        loss_masks = (loss_masks < input_lengths.unsqueeze(1)).float()

        total_loss = 0
        loss_dia = 0
        loss_dice = 0
        loss_cls = 0
        assert len(preds_mask) == len(preds_class)
        denom = len(preds_mask) * B
        lang_masks_ = lang_masks
        lang_masks = [sequence_to_mask(lang_mask, remove_mute=REMOVE_MUTE).transpose(0,1) for lang_mask in lang_masks]
        for interm_mask, interm_class in zip(preds_mask, preds_class):
            costs, indices = matcher(interm_mask, interm_class, lang_masks)
            for pred_msk, pred_cls, idx, lang_msk, loss_msk in zip(interm_mask, interm_class, indices, lang_masks, loss_masks):
                if lang_msk.numel() == 0:
                    continue
                T, N = pred_msk.shape
                full_query_idx = torch.arange(N, device=pred_msk.device)
                lang_idx, query_idx = idx
                rest_idx = full_query_idx[~torch.isin(full_query_idx, query_idx)]
                pred_mask_sorted = pred_msk[:, query_idx]
                pred_cls = pred_cls.squeeze()
                pred_class_sorted = torch.cat([pred_cls[query_idx], pred_cls[rest_idx]], dim=-1)
                cur_total, cur_dia, cur_dice, cur_cls = criterion(pred_mask_sorted, pred_class_sorted, lang_msk, mask=loss_msk)
                total_loss += cur_total / denom
                loss_dia += cur_dia / denom
                loss_dice += cur_dice / denom
                loss_cls += cur_cls / denom
        
        total_loss = total_loss / NUM_ACCUMULATION_STEPS
        loss_dia = loss_dia / NUM_ACCUMULATION_STEPS
        loss_dice = loss_dice / NUM_ACCUMULATION_STEPS
        loss_cls = loss_cls / NUM_ACCUMULATION_STEPS
        total_loss.backward()

        running_loss += total_loss.item()
        running_loss_dia += loss_dia.item()
        running_loss_dice += loss_dice.item()
        running_loss_cls += loss_cls.item()
        step_loss += total_loss.item()
        step_loss_dia += loss_dia.item()
        step_loss_dice += loss_dice.item()
        step_loss_cls += loss_cls.item()
        SPECIFIC_STEP += 1

        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_loader)):
            if scheduler is not None:
                scheduler.step()

            CURRENT_STEP += 1
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            optimizer.zero_grad()
            
            print(f'Step {CURRENT_STEP} - Diarization Loss: {step_loss_dia} | Dice Loss: {step_loss_dice} | Classification Loss: {step_loss_cls}')

            step_loss = 0.0
            step_loss_dia = 0.0
            step_loss_dice = 0.0
            step_loss_cls = 0.0

        if SPECIFIC_STEP % (NUM_ACCUMULATION_STEPS * 500) == 0:
            model.eval()
            val_loss = 0.0
            val_loss_dia = 0.0
            val_loss_dice = 0.0
            val_loss_cls = 0.0
            with torch.no_grad():
                for val_inputs, val_lang_masks, val_input_lengths in val_loader:
                    val_inputs, val_lang_masks, val_input_lengths = val_inputs.to(device), val_lang_masks.to(device), val_input_lengths.to(device)

                    preds_mask, preds_class, val_input_lengths = model(val_inputs, val_input_lengths)
                    
                    val_B = val_input_lengths.size(0)
                    val_T = val_input_lengths.max()
                    val_loss_seq = torch.arange(val_T, device=val_input_lengths.device)
                    val_loss_masks = val_loss_seq.unsqueeze(0).expand(val_B, val_T)
                    val_loss_masks = (val_loss_masks < val_input_lengths.unsqueeze(1)).float()
                    
                    assert len(preds_mask) == len(preds_class)
                    denom = len(preds_mask) * val_B

                    lang_masks = [sequence_to_mask(lang_mask, remove_mute=REMOVE_MUTE).transpose(0, 1) for lang_mask in val_lang_masks]

                    for interm_mask, interm_class in zip(preds_mask, preds_class):
                        costs, indices = matcher(interm_mask, interm_class, lang_masks)
                        for pred_msk, pred_cls, idx, lang_msk, loss_msk in zip(interm_mask, interm_class, indices, lang_masks, val_loss_masks):
                            if lang_msk.numel() == 0:
                                continue
                            T, N = pred_msk.shape
                            full_query_idx = torch.arange(N, device=pred_msk.device)
                            lang_idx, query_idx = idx
                            rest_idx = full_query_idx[~torch.isin(full_query_idx, query_idx)]
                            pred_mask_sorted = pred_msk[:, query_idx]
                            pred_cls = pred_cls.squeeze()
                            pred_class_sorted = torch.cat([pred_cls[query_idx], pred_cls[rest_idx]], dim=-1)
                            cur_total, cur_dia, cur_dice, cur_cls = criterion(pred_mask_sorted, pred_class_sorted, lang_msk, mask=loss_msk)
                            val_loss += cur_total / denom
                            val_loss_dia += cur_dia / denom
                            val_loss_dice += cur_dice / denom
                            val_loss_cls += cur_cls / denom
                    
            val_loss = val_loss / len(val_loader)

            model.train()
            
            save_path = os.path.join(SAVE_DIR, f'iter{CURRENT_STEP}.pt')
            torch.save(model.state_dict(), save_path)
            if CURRENT_STEP >= NUM_TOTAL_STEPS:
                sys.exit(0)

    epoch_loss = running_loss * NUM_ACCUMULATION_STEPS / len(train_loader)
    epoch_loss_dia = running_loss_dia * NUM_ACCUMULATION_STEPS / len(train_loader)
    epoch_loss_dice = running_loss_dice * NUM_ACCUMULATION_STEPS / len(train_loader)
    epoch_loss_cls = running_loss_cls * NUM_ACCUMULATION_STEPS / len(train_loader)

    return epoch_loss

def main():

    global NUM_ACCUMULATION_STEPS
    global MINIBATCH
    global NUM_TOTAL_STEPS
    global SAVE_DIR
    global REMOVE_MUTE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--minibatch', type=int, default=8)
    parser.add_argument('--total_step', type=int, default=10000)
    parser.add_argument('--init_lr', type=float, default=5e-6)
    parser.add_argument('--peak_lr', type=float, default=1e-5)
    parser.add_argument('--final_lr', type=float, default=5e-6)
    parser.add_argument('--ft_dataset', type=str, default='africansoap')
    parser.add_argument('--ft_dataset_path', type=str, default='africansoap')
    parser.add_argument('--ft_dataset_sample_path', type=str, default='africansoap')
    parser.add_argument('--vad', type=str2bool, default='True')
    parser.add_argument('--ckpt_path', type=str, default='pretrained_ckpts/best_model.pt')
    parser.add_argument('--is_conformer', type=str2bool, default='True')
    parser.add_argument('--masked_attn', type=str2bool, default='True')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_query', type=int, default=5)
    parser.add_argument('--weight', type=float, nargs=3, default=[1.0, 1.0, 3.0])
    parser.add_argument('--alpha', type=float, nargs=3, default=[0.5, 0.5, 0.3])
    parser.add_argument('--gamma', type=float, default=0.75)
    parser.add_argument('--loss_scale', type=float, nargs=3, default=[1.0, 1.0, 1.0])
    
    args = parser.parse_args()
        
    dia_weight, dice_weight, cls_weight = args.loss_scale
    
    assert args.batch % args.minibatch == 0
    
    MINIBATCH = args.minibatch
    NUM_ACCUMULATION_STEPS = args.batch / args.minibatch
    NUM_TOTAL_STEPS = args.total_step
    SAVE_DIR = f'ckpts_{args.ft_dataset}_ft'
    REMOVE_MUTE =  not args.vad
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder = build_plain_encoder(d_model=args.hidden_dim, nhead=args.nhead, nlayers=args.num_layers, ffn_mult=4, is_conformer=args.is_conformer)
    decoder = build_decoder(num_queries=args.num_query, num_mlp_layers=3, num_layers=args.num_layers, ffn_mult=4, 
                            d_model=args.hidden_dim, nhead=args.nhead, dropout=0.0, activation='relu', normalize_before=True, apply_mask=args.masked_attn)
    model = EELD(encoder, decoder)
    
    if args.ckpt_path is not None:
        print('Loading pretrained parameters...')
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
        
        print('Freezing feature extractor parameters...')
        for name, param in model.encoder.feature_extractor.named_parameters():
            param.requires_grad=False
            
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)
    num_warmup_steps = int(NUM_TOTAL_STEPS * 0.1)
    num_hold_steps = int(NUM_TOTAL_STEPS * 0.6)
    num_decay_steps = NUM_TOTAL_STEPS - num_warmup_steps - num_hold_steps        
    scheduler = TriStageLRScheduler(optimizer, init_lr=args.init_lr, peak_lr=args.peak_lr, final_lr=args.final_lr, 
                            warmup_steps=num_warmup_steps, hold_steps=num_hold_steps, decay_steps=num_decay_steps, total_steps=NUM_TOTAL_STEPS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 16
    criterion = EELDLoss(num_query=args.num_query, weight=args.weight, alpha=args.alpha, gamma=args.gamma, dia_weight=dia_weight, dice_weight=dice_weight, cls_weight=cls_weight, neg_cls_factor=0.2) # Modified
    matcher = HungarianMatcher(num_query=args.num_query, weight=args.weight, alpha=args.alpha, gamma=args.gamma, cost_dia=dia_weight, cost_dice=dice_weight, cost_cls=cls_weight)

    assert args.ft_dataset in ['displace_dev', 'displace_eval', 'africansoap']
    if args.ft_dataset == 'displace_dev':
        train_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Dev', mode='train')
        val_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Dev', mode='test')
    elif args.ft_dataset == 'displace_eval':
        train_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Eval', mode='train')
        val_dataset = DISPLACEDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, official_split='Eval', mode='test')
    else:
        train_dataset = AfricanSoapDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, mode='train')
        val_dataset = AfricanSoapDataset(args.ft_dataset_path, base_audio_dir=args.ft_dataset_sample_path, mode='test')
    
    train_loader = DataLoader(train_dataset,
                          batch_size=MINIBATCH,
                          shuffle=True,
                          num_workers=num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=MINIBATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)
    
    model = model.to(device)
    matcher = matcher.to(device)
    criterion = criterion.to(device)
    
    for i in range(1000):
        train_loss = train_and_validate(model, matcher, train_loader, val_loader, criterion, optimizer, device, scheduler)    
    
    wandb.finish()
    
if __name__=='__main__':
    main()