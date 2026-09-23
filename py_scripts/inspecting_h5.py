import glob
import h5py
import numpy as np

# files = sorted(glob.glob("/home/mi58rip/gr-athena/runs/aic_test_SFHo/outputs/test_2/aic.out4.*.athdf"))
# file = "/home/mi58rip/gr-athena/eos_tables/SFHo.h5"
file = "/home/mi58rip/gr-athena/runs/aic_LS220_64_64_64/test_rewrite_ID_AMR/aic.out5.00054.athdf"

# with h5py.File(file, 'r') as f:
#     for name in f.keys():
#         print(name, ":", f[name][:].shape)
#         if "desc" in f[name].attrs:
#             print(name, ":", f[name].attrs["desc"])

with h5py.File(file, 'r') as f:
    print("Root-level attributes:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  {attr_name}: {attr_value}")
    
    print("\nHydro dataset attributes in detail:")
    hydro = f['hydro']
    for attr_name, attr_value in hydro.attrs.items():
        print(f"  {attr_name}:")
        if isinstance(attr_value, (list, np.ndarray)):
            if len(attr_value) <= 10:
                print(f"    {attr_value}")
            else:
                print(f"    {attr_value[:10]} ... (total {len(attr_value)} items)")
        else:
            print(f"    {attr_value}")