import h5py, os, numpy as np
emission_inventory = "dataset_storage/raw/TEDS13.h5"
landuse_pop = "dataset_storage/raw/landuse_pop.h5"
dsm = "dataset_storage/raw/dsm_500m.h5"

def build_static_features():
    if os.path.exists("dataset_storage/processed/static_features.h5"):
        print("'static_features.h5' already exists.")
        return

    with h5py.File(emission_inventory, "r") as emi, h5py.File(landuse_pop, "r") as lapo, h5py.File(dsm, "r") as alt:
        # Emission inventory, shape: (800, 400, C1=8)
        x_emi = emi["TEDS"][:].astype(np.float32)
        emi_mask = emi["TEDS_mask"][:].astype(np.float32)
        
        # Population density, land use ; shape: (800, 400), (800, 400, C2=9)
        x_peo = lapo["pop_density"][:].astype(np.float32)[..., None]
        peo_mask = (np.nan_to_num(x_peo, nan=0.0)!=0).astype(np.float32)
        x_landuse = lapo["land_use"][:].astype(np.float32)
        landuse_mask = lapo["land_use_mask"][:].astype(np.float32)

        # DSM (convert m to km), shape: (800, 400)
        x_dsm = (alt["elevation"][:].astype(np.float32)/1000)[..., None]
        dsm_mask = (~np.isnan(x_dsm)).astype(np.float32)
        
        # Calculate Slope (Gradient, km) using numpy
        dsm_m = alt["elevation"][:].astype(np.float32) # Use original elevation in meters
        dsm_dx = np.pad(dsm_m[:, 1:] - dsm_m[:, :-1], ((0, 0), (1, 0)), mode='constant')
        dsm_dy = np.pad(dsm_m[1:, :] - dsm_m[:-1, :], ((1, 0), (0, 0)), mode='constant')
        dsm_grad = (np.sqrt(dsm_dx**2 + dsm_dy**2 + 1e-5) / 1000.0)[:, :, None]

        # Concatenate all spatial time static feature, shape: (C1+C2+3=20, 800, 400)
        x_static = np.concatenate((x_emi, x_landuse, x_peo, x_dsm, dsm_grad), axis=2)
        x_static = np.nan_to_num(x_static, nan=0.0).transpose(2, 0, 1)
        static_mask = np.concatenate((emi_mask, landuse_mask, peo_mask, dsm_mask, dsm_mask), axis=2).transpose(2, 0, 1)
        #print(f"static features max and min value: {np.nanmax(x_static)}, {np.nanmin(x_static)}")

    print(f"Saving static features shape: {x_static.shape}, mask shape: {static_mask.shape}")
    with h5py.File("dataset_storage/processed/static_features.h5", "w") as f:
        f["val"], f["mask"] = x_static, static_mask
