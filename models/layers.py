import math
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================================
# >> HELPER MODULES
# ========================================================================================
class SingleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3, 3), stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kt = kh = kw = kernel_size
        else:
            kt, kh, kw = kernel_size

        # Match PConv3D asymmetric temporal padding logic
        self.pad_t_front = (kt - 1) * dilation
        
        if isinstance(stride, int):
            stride_t, stride_h, stride_w = 1, stride, stride
        else:
            stride_t, stride_h, stride_w = 1, stride[1], stride[2]

        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            _, pad_h, pad_w = padding

        real_padding = (0, pad_h, pad_w)
        real_stride = (stride_t, stride_h, stride_w)
        
        self.input_conv = nn.Conv3d(in_channels, out_channels, (kt, kh, kw),
                                    real_stride, real_padding, dilation, groups, bias)
        self.input_conv.apply(weights_init('kaiming'))

    def forward(self, x, pad=True):
        if pad and self.pad_t_front > 0:
            # F.pad order: (W, H, T) -> (Left, Right, Top, Bottom, Front, Back)
            pad_tuple = (0, 0, 0, 0, self.pad_t_front, 0) 
            x = F.pad(x, pad_tuple, mode='constant', value=0.0)
        elif not x.is_contiguous():
            x = x.contiguous()
        
        return self.input_conv(x)

# Pseudo 3D Conv
class Block(nn.Module):
    def __init__(self, c_in, c_out, k_t=4, k_s=3, stride=1, bias=False, act=False, temporal=True):
        super().__init__()
        # Decompose (k_t, k_s, k_s) -> (1, k_s, k_s) + (k_t, 1, 1)
        self.spatial = SingleConv3D(c_in, c_out, (1, k_s, k_s), (1, stride, stride), padding=(0, k_s//2, k_s//2), bias=bias)
        if temporal:
            self.temporal = SingleConv3D(c_out, c_out, kernel_size=(k_t, 1, 1), stride=1, padding=(0, 0, 0), bias=bias)
        if act:
            self.norm_act = nn.Sequential(nn.GroupNorm(8 if c_out%8==0 else 1, c_out), nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, x, pad=True):
        x = self.spatial(x, pad=False)
        if hasattr(self, "norm_act"):
            x = self.norm_act(x)
        if hasattr(self, "temporal"):
            return self.temporal(x, pad)
        return x

class SCBlock(nn.Module):
    """ Convolution (2D or 3D Spatial) with Norm and activation. """
    def __init__(self, in_ch, out_ch, kz, stride=1, padding=0, bias=False, is_2d=False):
        super().__init__()
        if is_2d:
            self.conv = nn.Conv2d(in_ch, out_ch, kz, stride, padding, bias=bias)
        else:
            k_size = (1, kz, kz) if isinstance(kz, int) else kz
            s_size = (1, stride, stride) if isinstance(stride, int) else stride
            p_size = (0, padding, padding) if isinstance(padding, int) else padding
            self.conv = nn.Conv3d(in_ch, out_ch, k_size, s_size, p_size, bias=bias)
        self.norm_activation = nn.Sequential(nn.GroupNorm(8 if out_ch%8==0 else 1, out_ch), nn.LeakyReLU(0.2, inplace=True))
        self.conv.apply(weights_init('kaiming'))

    def forward(self, x):
        if not x.is_contiguous():   x = x.contiguous()
        x = self.conv(x)
        return self.norm_activation(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d(1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_out = self.fc(self.max(x))
        avg_out = self.fc(self.avg(x))
        return x + self.sigmoid(max_out + avg_out)

def weights_init(init_type='gaussian'):  # parameter weigts initialization
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun