import json, torch, torchaudio, random
from collections import OrderedDict

def hf2torch(state_dict):
    assert isinstance(state_dict, OrderedDict), "Input must be an OrderedDict"

    state_dict.pop('masked_spec_embed', None)
    
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key

        if 'feature_projection' in new_key:
            new_key = 'encoder.' + new_key
            
        elif 'encoder' in new_key:
            new_key = new_key.replace('encoder', 'encoder.transformer')
        
        new_state_dict[new_key] = value

    return new_state_dict

def compute_output_length(length, kernel_sizes=[10,3,3,3,3,2,2], strides=[5,2,2,2,2,2,2]):
    for k, s in zip(kernel_sizes, strides):
        length = torch.div(length - k, s, rounding_mode="floor") + 1
        length = torch.max(torch.zeros_like(length), length)
    return length

def apply_lang_segments(flag_dict, mask, lang2id, return_embedding=True):
    for lang, segments in flag_dict.items():
        lang_id = lang2id[lang]
        for seg in segments:
            if return_embedding:
                start = compute_output_length(seg['start'])
                end = compute_output_length(seg['end'])
            else:
                start = seg['start']
                end = seg['end']
            mask[start:end] = lang_id
    
    return mask

def apply_lang_segments_ft(flag_dict, mask, return_embedding=True):
    for i, (lang, segments) in enumerate(flag_dict.items(), start=1):
        lang_id = i
        for seg in segments:
            if return_embedding:
                start = compute_output_length(seg['start'])
                end = compute_output_length(seg['end'])
            else:
                start = seg['start']
                end = seg['end']
            mask[start:end] = lang_id
            
    return mask

def apply_lang_segments_ft_vad(flag_dict, mask, return_embedding=True):
    for i, (lang, segments) in enumerate(flag_dict.items(), start=1):
        for seg in segments:
            if return_embedding:
                start = compute_output_length(seg['start'])
                end = compute_output_length(seg['end'])
            else:
                start = seg['start']
                end = seg['end']
            mask[start:end] = 0
            
    return mask

def apply_lang_masking(flag_dict, mask, lang2id, return_embedding=True, vad=False):
    for lang, segments in flag_dict.items():
        lang_id = lang2id[lang] 
        if vad:
            for seg in segments:
                if return_embedding:
                    start = compute_output_length(seg['start']) 
                    end = compute_output_length(seg['end']) 
                else:
                    start = seg['start']
                    end = seg['end']  

                mask[:, start:end] = 0
                mask[lang_id, start:end] = 1           

        else:
            for seg in segments:
                if return_embedding:
                    start = compute_output_length(seg['start']) 
                    end = compute_output_length(seg['end']) 
                else:
                    start = seg['start']
                    end = seg['end']    

                mask[lang_id, start:end] = 1
    
    return mask

def relabel_mask(tensor):
    non_zero = tensor[tensor != 0]

    if non_zero.numel() == 0:
        return tensor.clone()

    unique_vals, counts = non_zero.unique(return_counts=True)
    majority_val = unique_vals[torch.argmax(counts)]

    result = tensor.clone()
    result[result == majority_val] = 1 
    result[(result != 0) & (result != 1)] = 2

    return result

def sequence_to_mask(labels: torch.Tensor, remove_mute: bool):
    
    labels = labels.long()
    num_classes = int(labels.max().item() + 1)

    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # (T, N) or (B, T, N)

    if one_hot.ndim == 2:  # input was (T,)
        one_hot = one_hot.permute(1, 0)  # (N, T)
        if remove_mute:
            one_hot = one_hot[1:,:]
        return one_hot
    elif one_hot.ndim == 3:  # input was (B, T)
        one_hot = one_hot.permute(0, 2, 1)  # (B, N, T)
        if remove_mute:
            one_hot = one_hot[:,1:,:]
        return one_hot
    else:
        raise ValueError(f"Unexpected input shape: {labels.shape}")
    
def pad_lang_mask(lang_mask: torch.Tensor, target_dim: int = 3):
    
    T, L = lang_mask.shape
    if L < target_dim:
        padding = torch.zeros(T, target_dim - L, device=lang_mask.device, dtype=lang_mask.dtype)
        lang_mask = torch.cat([lang_mask, padding], dim=1)
        
    return lang_mask

def normalize_audio(x):
    max_value = torch.max(torch.abs(x))

    normalized_x = x / max_value
    
    return normalized_x

def apply_rir(audio, rir):
    reverb = torchaudio.functional.fftconvolve(audio, rir, mode='full')
    reverb = reverb[:, :audio.shape[1]]  # Trim to original length
    reverb = reverb / reverb.abs().max()  # Optional normalization
    
    return reverb

def add_noise(audio, noise=None, snr_db=10):
    
    if noise is None:
        noise = torch.randn_like(audio) # Gaussian noise
    
    if noise.shape[-1] > audio.shape[-1]:
        max_start = noise.shape[-1] - audio.shape[-1]
        start = random.randint(0, max_start)
        noise = noise[..., start:start + audio.shape[-1]]
    
    audio = torchaudio.functional.add_noise(audio, noise, torch.tensor([snr_db]))
    
    return audio

def load_json(path, reverse=False):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if reverse:
        data = {v: k for k, v in data.items()}
    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    
def compute_der(pred: torch.Tensor, ref: torch.Tensor):
    assert pred.shape == ref.shape, "Shape mismatch between pred and ref"

    total = ref.numel() 

    fa = ((ref == 0) & (pred != 0)).sum().item()

    miss = ((ref != 0) & (pred == 0)).sum().item()

    confusion = ((ref != 0) & (pred != 0) & (ref != pred)).sum().item()

    if total == 0:
        der = 0.0
    else:
        der = (fa + miss + confusion) / total

    return {
        'DER': der,
        'FA': fa,
        'Miss': miss,
        'Confusion': confusion,
        'Total': total
    }