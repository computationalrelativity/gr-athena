import sys
sys.path.append("/home/mi58rip/pycompose") 
import shutil
import numpy as np
import h5py

from compose.eos import Metadata, Table

# 1. Define particle tracking metadata for SFHo
md = Metadata(
    pairs={
        0: ("e", "electron"),
        10: ("n", "neutron"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads={
        999: ("N", "average nucleus")
    }
)

# 2. Load the raw CompOSE files
eos = Table(md)
eos.read("/home/mi58rip/gr-athena/eos_tables/SFHo_raw/eos") 

# 3. Compute derived quantities, clean out unphysical points, and export
eos.compute_cs2(floor=1e-1)
eos.validate()
eos.shrink_to_valid_nb()

eos.add_Ye_from_yq(overwrite=True)
out_file = "/home/mi58rip/gr-athena/eos_tables/SFHo.h5"
eos.write_hdf5(out_file)

Ye_idx = np.argmin(np.abs(eos.yq - 0.5))
cold_eos_1d = (eos.slice_at_t_idx(0)).slice_at_y_idx(Ye_idx)
cold_eos_1d.add_coldslice(out_file)

# def create_h5_with_cold_slice(input_filename, output_filename):
#     # Safely duplicate the original file to preserve it
#     shutil.copy(input_filename, output_filename)
#     print(f"Copied {input_filename} to {output_filename}")
    
#     # Open the new copy using a context manager for safety
#     with h5py.File(output_filename, "a") as f:
#         if "cold_slice" in f:
#             del f["cold_slice"]
            
#         grp = f.create_group("cold_slice")

#         # Load 3D base arrays
#         nb = np.array(f['nb'])
#         yq = np.array(f['yq'])
#         t  = np.array(f['t'])
#         mn = f['mn'][()]

#         # Locate the beta-equilibrium slice at the lowest temperature (T index 0)
#         Q6 = np.array(f['Q6'])
#         Q6_cold = Q6[:, :, 0]
#         beta_eq_indices = np.argmin(Q6_cold, axis=1)
#         idx_nb = np.arange(len(nb))

#         # 1. Write the metadata datasets expected by GR-Athena++
#         grp.create_dataset("nb", data=nb)
#         grp.create_dataset("t", data=np.full_like(nb, t[0]))
#         grp.create_dataset("yq", data=yq[beta_eq_indices])
#         grp.create_dataset("mn", data=np.array([mn]))

#         # 2. Extract and write all thermodynamic datasets (Q1-Q9, cs2)
#         keys_to_extract = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'cs2']
        
#         for key in keys_to_extract:
#             if key in f:
#                 data_3d = np.array(f[key])
#                 data_1d = data_3d[idx_nb, beta_eq_indices, 0]
#                 grp.create_dataset(key, data=data_1d)

#     print(f"Success: 'cold_slice' group with 1D data appended to {output_filename}")

# create_h5_with_cold_slice("SFHo.h5", "SFHo_with_cold_slice.h5")

# def patch_hdf5(h5_filename):
#     with h5py.File(h5_filename, "a") as f:
#         grp = f["cold_slice"]
        
#         # Check if Y[e] already exists to avoid errors on multiple runs
#         if "Y[e]" in grp:
#             del grp["Y[e]"]
            
#         # Duplicate the 'yq' array as 'Y[e]'
#         grp.create_dataset("Y[e]", data=grp["yq"][:])
        
#     print(f"Success: 'Y[e]' dataset injected into {h5_filename}")

# patch_hdf5("SFHo_with_cold_slice.h5")

print(f"Success: {out_file} has been generated!")