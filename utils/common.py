import os , torch , random

# ================================================
# Data augmentation and pre-trained weight loading
# ================================================
class Augmentation():
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value

    def do(self, x_data, x_mask, y_target, y_mask, shift=40, drop_prob=0.2):
        """
        Apply Shift and Masking Augmentation.
        drop_prob: Probability to apply masking augmentation per batch.
        """
        # Spatial Shift
        max_shift = shift
        if max_shift > 0:
            delta_h = random.randint(-max_shift, max_shift)
            delta_w = random.randint(-max_shift, max_shift)
            x_data = self.shift(x_data, delta_h, delta_w, fill_value=self.fill_value)
            x_mask = self.shift(x_mask.float(), delta_h, delta_w, fill_value=0.0)
            y_target = self.shift(y_target, delta_h, delta_w, fill_value=0.0)
            y_mask = self.shift(y_mask.float(), delta_h, delta_w, fill_value=0.0)

        # Random Masking (Only on Input X)
        if random.random() < drop_prob:
            # Choose between 'pixel' (Bernoulli) or 'block' (Coarse) masking
            mode = 'pixel' if random.random() < 0.5 else 'block'
            x_data, x_mask = self.masking(x_data, x_mask, mode=mode)

        return x_data, x_mask, y_target, y_mask

    def masking(self, x, x_mask, mode='pixel', drop_rate=0.1):
        """
        Args:
            x: (B, T, C, H, W) or (B, C, H, W)
            x_mask: same shape as x
            mode: 'pixel' (random points) or 'block' (continuous regions)
        """
        if mode == 'pixel':
            # Your original logic (Bernoulli Dropout)
            keep_prob = 1.0 - drop_rate
            # Generate random mask (1=keep, 0=drop)
            rand_mask = torch.bernoulli(torch.full_like(x_mask, keep_prob)).to(x_mask.device)
            
            new_mask = x_mask * rand_mask
            x_new = x * new_mask if self.fill_value == 0.0 else torch.where(new_mask >= 0.5, x, self.fill_value)
            return x_new, new_mask

        elif mode == 'block':
            # Simulates large missing regions (more effective for artifacts)
            # Randomly zero out rectangular regions
            H, W = x.shape[-2:]
            new_mask = x_mask.clone()
            
            # Number of blocks to drop
            num_blocks = random.randint(2, 3) 
            
            for _ in range(num_blocks):
                # Random block size
                h_size = random.randint(H//16, H//8) # 50 ~ 100 px
                w_size = random.randint(W//8, W//4)  # 50 ~ 100 px
                
                h_start = random.randint(0, H - h_size)
                w_start = random.randint(0, W - w_size)
                
                # Apply to spatial dims (last 2 dims)
                new_mask[..., h_start:h_start+h_size, w_start:w_start+w_size] = 0.0
            
            x_new = x * new_mask if self.fill_value == 0.0 else torch.where(new_mask >= 0.5, x, self.fill_value)
            return x_new, new_mask
            
        return x, x_mask

    # Shift tensor with custom fill value. x: (..., H, W)
    def shift(self, x, shift_h, shift_w, fill_value=0.0):
        dtype, dev = x.dtype, x.device
        out = torch.zeros_like(x) if fill_value == 0.0 else torch.full_like(x, fill_value, device=dev, dtype=dtype)
        H, W = x.shape[-2:]
        
        src_h_start = max(0, -shift_h)
        src_h_end = min(H, H - shift_h)
        src_w_start = max(0, -shift_w)
        src_w_end = min(W, W - shift_w)
        
        dst_h_start = max(0, shift_h)
        dst_h_end = min(H, H + shift_h)
        dst_w_start = max(0, shift_w)
        dst_w_end = min(W, W + shift_w)
        
        out[..., dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
            x[..., src_h_start:src_h_end, src_w_start:src_w_end]
            
        return out

# ========================
# Load pre-trained weights
# ========================
def load_pretrained(model, pretrained_path, device):
    if not os.path.exists(pretrained_path):
        print("----- No pre-trained weights -----")
        return model
    print("----- Loading pre-trained weights -----")
    # Load pretrained state_dict and get current model state_dict
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys (mismatched shapes, specifically the final output layer)
    # Check if key exists in current model and the shape matches (crucial for C_out mismatch at dec_1)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    if len(filtered_dict)==0:
        print("pretrained model architecture is mismatched")
    else:
        # Analyze excluded layers (for verification)
        excluded_layers = set(pretrained_dict.keys()) - set(filtered_dict.keys())
        if len(excluded_layers) > 0:
            print(f"Excluded layers due to mismatch: {', '.join([layer for layer in excluded_layers])}")

        # Update and load current model weights
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    
    return model
