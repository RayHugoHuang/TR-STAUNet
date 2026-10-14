import torch
import torch.nn as nn
# ===============================================
# Shared helper functions for Loss
# ===============================================
def _get_channel_weights(C, device, weights_if_6):
    return torch.tensor(weights_if_6 if C == 6 else [1.0] * C, device=device).unsqueeze(0)

def _reduce_masked_loss(loss, mask_sum, device):
    valid = mask_sum > 0
    return loss[valid].mean() if valid.any() else torch.tensor(float('nan'), device=device)

# ===============================================
# Revised from standard Loss function in torch.nn
# ===============================================
class CostSensitiveL1Loss(nn.Module):
    def __init__(self, sensitivity=1.0):
        super().__init__()
        self.alpha = sensitivity
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        if mask is None:    mask = torch.ones_like(target)
        mask_sum = mask.sum(dim=(2, 3)) # Calculate original mask sum for validity check

        # L1 error and weight
        l1_err = self.loss(pred, target)
        weight = 1.0 + self.alpha * torch.log1p(target.clamp(min=0)) # 1 + alpha * log(1 + target)
        loss = (l1_err * weight * mask).sum(dim=(2, 3)) / mask_sum.clamp(min=1)
        
        # Compensate for different scales, especially boosting CO and SO2
        loss = loss * _get_channel_weights(pred.shape[1], pred.device, [1.0] * 4 + [9.0, 6.0])
        return _reduce_masked_loss(loss, mask_sum, pred.device)
    
class MaskedL1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask):
        mask_sum = mask.sum(dim=(0, 2, 3))
        loss = ((pred - target).abs() * mask).sum(dim=(0, 2, 3)) / mask_sum.clamp(min=1e-8)
        return loss.mean() if self.reduction == "mean" else loss
    
# Regularization Loss (Customized)
def transmission_resistance_loss(pred, y_mask, uv_grid, uv_mask, alpha=5.0):
    # u_wind, v_wind, dsm_gradient: (B*T, 3, H, W) -> (B*T, 1, H, W)
    u_wind, v_wind, dsm_grad = uv_grid.chunk(3, dim=1)
    calc_mask = uv_mask[:, 0:1, :, :]

    # 1. Spatial gradients of predicted concentration (dx: Eastward gradient, dy: Northward gradient)
    dx = torch.nn.functional.pad(pred[:, :, :, 1:] - pred[:, :, :, :-1], (0, 1, 0, 0))
    dy = torch.nn.functional.pad(pred[:, :, 1:, :] - pred[:, :, :-1, :], (0, 0, 0, 1))

    # 2. Wind magnitude and directional components
    wind_mag_sq = u_wind.pow(2) + v_wind.pow(2)
    wind_magnitude = torch.sqrt(wind_mag_sq + 1e-4)
    u_norm = u_wind / wind_magnitude
    v_norm = v_wind / wind_magnitude

    # 3. Directional concentration derivatives (along & cross streamline)
    along_grad = dx * u_norm + dy * v_norm
    cross_grad = dx * v_norm - dy * u_norm

    # 4. Along: (wind_speed · ∇C)^2 + along_grad^2 & Cross: (wind_speed_norm × ∇C)^2 / (wind_speed + 1)
    along_loss = along_grad.pow(2) * (wind_mag_sq + 1.0)
    cross_loss = cross_grad.pow(2) / (wind_magnitude + 1.0)

    # 5. Topographic barrier & scale factor against station anchoring
    dsm_coef = 30 * torch.exp(-alpha * dsm_grad)
    loss_map = (along_loss + cross_loss + 1e-4) * dsm_coef

    # 6. Spatial reduction and channel weights
    norm = torch.sum(calc_mask, dim=(2, 3))
    spatial_loss = torch.sum(loss_map * calc_mask, dim=(2, 3)) / norm.clamp(min=1)

    channel_weights = _get_channel_weights(pred.shape[1], pred.device, [1.0] * 6)
    return (channel_weights * spatial_loss).mean()

def temporal_consistency_loss(pred, target, mask):
    """ Penalize temporal jitter by matching first-order derivatives. """
    if pred.dim() != 5 or pred.shape[1] <= 1:    return 0.0
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    valid_mask = mask[:, 1:] * mask[:, :-1]
    
    return (torch.abs(dp - dt) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
