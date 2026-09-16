import glob
import h5py
import numpy as np

# files = sorted(glob.glob("/home/mi58rip/gr-athena/runs/aic_test_SFHo/outputs/test_2/aic.out4.*.athdf"))
# file = "/home/mi58rip/gr-athena/eos_tables/SFHo.h5"
file = "/home/mi58rip/gr-athena/eos_tables/LS220_240r_140t_50y_analmu_20120628_SVNr28_pycompose.h5"

# with h5py.File(file, 'r') as f:
#     for name in f.keys():
#         print(name, ":", f[name][:].shape)
#         if "desc" in f[name].attrs:
#             print(name, ":", f[name].attrs["desc"])

with h5py.File(file, 'r') as f:

    def inspect(name, obj):

        if isinstance(obj, h5py.Dataset):
            print(name, ":", obj.shape)

            if "desc" in obj.attrs:
                print("    desc:", obj.attrs["desc"])

        elif isinstance(obj, h5py.Group):
            print(name, ": GROUP")

    f.visititems(inspect)

    # print(f['Abar'][:])


    # mn = f['mn']
    # # print value of mn in MeV
    # print("mn:", mn[()], "MeV")
    # # in grams
    # print("mn:", mn[()] * 1.78266192e-27, "grams")
