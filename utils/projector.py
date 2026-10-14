import torch, h5py, os
import torch.nn as nn
import torch.nn.functional as F
from configs import cfg
STATIC_FEA = [fea for fea in cfg.STATIC_ABBRE if fea not in cfg.EXCLUDE_SET]

class SpatialProjector(nn.Module):
    """ Differentiable Spatial Projector layer. Maps sparse coordinates (N, 2) to dense grid (H, W). """
    def __init__(self, lat_range=cfg.GEO_AREA[1], lon_range=cfg.GEO_AREA[0], res=cfg.RESOLU, 
                 static_path=cfg.STATIC_PATH, interpolation_method=cfg.INTERPOLATION_METHOD):
        super().__init__()

        # Define Physics Boundaries as non-learnable buffers
        self.lat_min, self.lon_min, self.res = lat_range[0], lon_range[0], res
        self.H = int((lat_range[1] - lat_range[0]) / res) # H = 4.0 / 0.005 = 800
        self.W = int((lon_range[1] - lon_range[0]) / res) # W = 2.0 / 0.005 = 400
        self.fill_value = cfg.FILL_VALUE
        self.interpolation = False
        self.interpolation_method = interpolation_method

        self.has_static = os.path.exists(static_path) and bool(STATIC_FEA)
        if self.has_static:
            print(f"Loading static features from {static_path}")
            self.keep_idx = torch.tensor([cfg.STATIC_ABBRE.index(f) for f in STATIC_FEA], dtype=torch.long)
            with h5py.File(static_path, 'r') as f:
                val = torch.from_numpy(f['val'][self.keep_idx, :])   # Shape: (C_stat, H, W)
                mask = torch.from_numpy(f['mask'][self.keep_idx, :]) # Shape: (C_stat, H, W)
                val[mask < 0.5] = self.fill_value
            
            self.register_buffer('static_val', val.unsqueeze(0).unsqueeze(0))   # (C, H, W) -> (1, 1, C, H, W)
            self.register_buffer('static_mask', mask.unsqueeze(0).unsqueeze(0)) # (C, H, W) -> (1, 1, C, H, W)

    def project_tensor(self, vals, geos, param=None):
        """
        Strict Projection: Handles NaN in coords and NaN in values.
        Args:
            vals: (B, T, N, C) - Physical values
            geos: (B, T, N, 2) - Coordinates [Lon, Lat]
            param: (H, W, res) - Output target size and spatial resolution
        Returns:
            grid & mask: (B, T, C, H, W)
        """
        B, T, _, C = vals.shape
        device = vals.device
        H, W, res = param if param is not None else (self.H, self.W, self.res)
        res_lat, res_lon = res if isinstance(res, tuple) else (res, res)
        
        # Manifold Flattening (B, T, N, C) -> (B*T, N, C)
        vals_flat = vals.flatten(0, 1)
        geos_flat = geos.expand(B * T, -1, -1) if geos.ndim == 3 else geos.flatten(0, 1)
        
        # Coordinate Transformation
        lon_f = torch.round((geos_flat[..., 0] - self.lon_min) / res_lon)
        lat_f = torch.round((geos_flat[..., 1] - self.lat_min) / res_lat)
        spatial_mask = (lat_f >= 0) & (lat_f < H) & (lon_f >= 0) & (lon_f < W)
        
        # Final Valid Mask: Expand spatial mask to match vals_flat
        final_mask = spatial_mask.unsqueeze(-1) & ~torch.isnan(vals_flat)
        
        # Fast Sparse Indexing using torch.where to avoid massive tensor allocations
        b_idx, n_idx, c_idx = torch.where(final_mask)
        
        # Cast ONLY valid coordinates to integer safely
        y_idx = lat_f[b_idx, n_idx].to(torch.long)
        x_idx = lon_f[b_idx, n_idx].to(torch.long)
        
        # Scatter
        grid = torch.zeros(B * T, C, H, W, device=device, dtype=vals.dtype)
        mask = torch.zeros(B * T, C, H, W, device=device, dtype=vals.dtype)
        
        # Deterministic Write
        grid[b_idx, c_idx, y_idx, x_idx] = vals_flat[final_mask]
        mask[b_idx, c_idx, y_idx, x_idx] = 1.0
        
        return grid.view(B, T, C, H, W), mask.view(B, T, C, H, W)

    @torch.no_grad()
    def forward(self, gems_val, gems_geo, era5_val, era5_geo, hgt_val=None, hgt_geo=None):
        """
        Args:
            gems_val, gems_geo: Tensor (B, T, N_gems, C_gems), (B, T, N_gems, 2)
            era5_val, era5_geo: Tensor (B, T, N_era5, C_era5), (B, T, N_era, 2)
            hgt_val, hgt_geo: Tensor (B, T, N_hgt, C_hgt), (B, T, N_hgt, 2)
        Returns:
            out: (B, T, C_total, H, W)
        """
        grids, masks = [], []
        B, T = gems_val.shape[:2]
        kwargs = {'mode': self.interpolation_method, 'size': (self.H, self.W)}
        if self.interpolation_method in ["linear", 'bilinear', 'bicubic']:
            kwargs['align_corners'] = True

        for v, g, p in [(gems_val, gems_geo, (100, 25, (0.04, 0.08))), (era5_val, era5_geo, (40, 20, 0.1))]:
            if v.shape[-1] == 0:
                gr = torch.zeros(B, T, 0, self.H, self.W, device=v.device, dtype=v.dtype)
                ma = torch.zeros(B, T, 0, self.H, self.W, device=v.device, dtype=v.dtype)
            else:
                gr, ma = self.project_tensor(v, g, p if self.interpolation else None)
                if self.interpolation:
                    gr = F.interpolate(gr.flatten(0, 1), **kwargs).view(B, T, -1, self.H, self.W)
                    ma = F.interpolate(ma.flatten(0, 1), **kwargs).view(B, T, -1, self.H, self.W)
                if self.fill_value != 0.0:    gr = torch.where(ma >= 0.5, gr, self.fill_value)
                ma = (ma >= 0.5).to(ma.dtype)
            grids.append(gr)
            masks.append(ma)

        # Process Historical GT without interpolation
        if hgt_val is not None and hgt_geo is not None:
            gr, ma = self.project_tensor(hgt_val, hgt_geo)
            if self.fill_value != 0.0:    gr = torch.where(ma >= 0.5, gr, self.fill_value)
            grids.append(gr)
            masks.append(ma)
            
        if self.has_static:
            grids.append(self.static_val.expand(B, T, -1, -1, -1))
            masks.append(self.static_mask.expand(B, T, -1, -1, -1))

        return torch.cat(grids, dim=2), torch.cat(masks, dim=2) # Concatenation along Channel dimension
    
    @torch.no_grad()
    def project_gt(self, y_vals, y_geos, interpolation=False):
        """
        Projects Ground Truth data and generates a validity mask for Loss calculation.
        Args:
            y_vals: (B, T, N, C) - GT values (contain NaNs)
            y_geos: (B, T, N, 2) - GT coordinates [Lon, Lat]
        Returns:
            grid: (B, T, C, H, W) - Sparse GT grid (NaNs replaced by 0)
            mask: (B, T, C, H, W) - Binary mask (1=Valid, 0=Empty/NaN)
        """
        if interpolation: # for uv grid
            grid, mask = self.project_tensor(y_vals, y_geos, (40, 20, 0.1))
            B, T = grid.shape[:2]
            grid = F.interpolate(grid.flatten(0, 1), size=(self.H, self.W), mode='bilinear', align_corners=True)
            mask = F.interpolate(mask.flatten(0, 1), size=(self.H, self.W), mode='bilinear', align_corners=True)
            return grid.view(B, T, -1, self.H, self.W), mask.view(B, T, -1, self.H, self.W)
        
        return self.project_tensor(y_vals, y_geos)