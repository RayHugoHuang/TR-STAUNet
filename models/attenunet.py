import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SCBlock, weights_init
from configs.config import cfg
STATIC_FEA = [fea for fea in cfg.STATIC_ABBRE if fea not in cfg.EXCLUDE_SET]
SEQ_LEN = cfg.SEQ_LEN
# ============================================================================
# >> Base UNet method
# ============================================================================
class BaseUNet(nn.Module):
    def __init__(self, layer_size=4, late_fusion=True):
        super().__init__()
        self.freeze_enc_norm = False
        self.layer_size = layer_size
        self.static_c = len(STATIC_FEA)
        self.late_fusion = late_fusion & (self.static_c > 0)

class Attention2D(nn.Module):       # Multi-Head 2D Self-Attention
    def __init__(self, channels):
        super().__init__()
        self.heads = 4 if channels % 4 == 0 else 1
        self.gn = nn.GroupNorm(16, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        q, k, v = self.qkv(self.gn(x)).chunk(3, dim=1)
        q = q.view(b, self.heads, c // self.heads, h * w).permute(0, 1, 3, 2).contiguous()
        k = k.view(b, self.heads, c // self.heads, h * w).permute(0, 1, 3, 2).contiguous()
        v = v.view(b, self.heads, c // self.heads, h * w).permute(0, 1, 3, 2).contiguous()

        # Fused Kernel (Auto-selects FlashAttention if available)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return self.dropout(self.out(out)) + x_in

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.sigmoid(self.psi(F.leaky_relu(g1 + x1, 0.2, True)))
        return x * psi

class TemporalFusion(nn.Module):
    """ Fuses temporal dimension using depthwise convolution to retain history and reduce VRAM. """
    def __init__(self, channels, t_len, bias=False):
        super().__init__()
        self.t_len = t_len
        if t_len > 1:
            self.fuse = nn.Conv3d(channels, channels, kernel_size=(t_len, 1, 1), groups=channels, bias=bias)
            self.fuse.apply(weights_init('kaiming'))

    def forward(self, x):
        if self.t_len > 1:
            x = self.fuse(x)
        return x.squeeze(2) # Output shape: (B, C, H, W)

class TransformerBlock(nn.Module):
    def __init__(self, channels, t_len):
        super().__init__()
        self.temp_fuse = TemporalFusion(channels, t_len)
        self.attn = Attention2D(channels)
        self.ffn = nn.Sequential(
            nn.GroupNorm(16 if channels % 16 == 0 else 1, channels),
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.temp_fuse(x)
        x = self.attn(x)
        x = x + self.ffn(x)
        return x

class AttenUNet(BaseUNet):
    def __init__(self, in_channels, out_channels=6, layer_size=4, late_fusion=False):
        super().__init__(layer_size, late_fusion)
        dynamic_channels = in_channels - self.static_c
        dims = [128, 128, 256, 256] + [256] * (layer_size - 4)
        self.out_channels = out_channels
        self.t_len = cfg.T_LEN

        # --- Encoder (Spatial conv and temporal fusion) ---
        self.encoder = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        in_dim = dynamic_channels if self.late_fusion else in_channels
        
        for i in range(layer_size):
            out_dim = dims[i] if i < len(dims)-1 else dims[-1]
            self.skip_fusions.append(TemporalFusion(in_dim, SEQ_LEN))
            self.encoder.append(nn.Sequential(
                SCBlock(in_dim, out_dim, 3, 2, 1),
                SCBlock(out_dim, out_dim, 3, 1, 1),
                nn.Dropout3d(0.1) if i>1 else nn.Identity()
            ))
            in_dim = out_dim

        # --- Bottleneck (Transformer) ---
        self.bottleneck = TransformerBlock(dims[-1], SEQ_LEN)

        # --- Decoder (Spatial only with Attention Gates) ---
        self.decoder = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        
        in_dim = dims[-1]
        for i in reversed(range(layer_size)):
            out_dim = dims[i]
            skip_dim = dynamic_channels if (self.late_fusion and i == 0) else (in_channels if i == 0 else dims[i-1])
            self.att_gates.append(AttentionGate(in_dim, skip_dim, max(16, out_dim // 2)))
            if i == 0:
                self.decoder.append(nn.Conv2d(in_dim + skip_dim, out_channels * self.t_len, 3, 1, 1))
            else:
                self.decoder.append(nn.Sequential(
                    SCBlock(in_dim + skip_dim, out_dim, 3, 1, 1, is_2d=True),
                    SCBlock(out_dim, out_dim, 3, 1, 1, is_2d=True),
                    nn.Dropout2d(0.1) if i>1 else nn.Identity()
                ))
            in_dim = out_dim

    def forward(self, x):
        x, static_x = self._prepare_input(x) # (B, T, C, H, W) -> (B, C, T, H, W)
        skips = []

        # ------------------------------ Encoder & Bottleneck ------------------------------
        for i, encoder in enumerate(self.encoder):
            skips.append(self.skip_fusions[i](x))
            x = encoder(x)
        x = self.bottleneck(x)

        # ------------------------------ Decoder ------------------------------
        for i in range(self.layer_size):
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            skip = self.att_gates[i](g=x, x=skip)
            cat_features = [x, skip, static_x] if (i == self.layer_size - 1 and self.late_fusion) else [x, skip]
            x = self.decoder[i](torch.cat(cat_features, dim=1))
        
        if not torch.is_grad_enabled(): x = F.relu(x, inplace=True)
        return x.reshape(x.shape[0], self.t_len, self.out_channels, x.shape[-2], x.shape[-1])
    
    def _prepare_input(self, x):
        if self.late_fusion:
            # Separate dynamic and static features
            dynamic_x = x[:, :, :-self.static_c, ...].permute(0, 2, 1, 3, 4).contiguous() # (B, C, T, H, W)
            static_x = x[:, -1, -self.static_c:, ...].contiguous() # (B, C, H, W)
            return dynamic_x, static_x
        return x.permute(0, 2, 1, 3, 4).contiguous(), None

if __name__ == "__main__":
    model = AttenUNet(30)
    for name, module in model.named_modules():
        print(name)