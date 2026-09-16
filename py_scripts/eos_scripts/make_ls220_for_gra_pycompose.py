# Create a a file in the format for RNS code from the LS220 EOS table. Without implementing any functions but using all from the PyCompose code.

import sys
sys.path.append("/home/mi58rip/pycompose") 
import shutil
import numpy as np
import h5py

from compose.eos import Metadata, Table

mn_CGS = 1.674927370796472e-24
mn_MeV = mn_CGS * 5.609588603e26  # Convert grams to MeV/c^2

# LS220_table_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5"
LS220_table_path = "/home/mi58rip/gr-athena/eos_tables/LS220_240r_140t_50y_analmu_20120628_SVNr28.h5"

# md = Metadata(
#     pairs = {
#         0: ("e", "electron"),
#         10: ("n", "neutro"),
#         11: ("p", "proton"),
#         4002: ("He4", "alpha particle"),
#     },
# )

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

eos = Table(md)

# eos.read_from_stellarcollapse(LS220_table_path, mb=mn_MeV)
eos.read_from_stellarcollapse(LS220_table_path)

eos.compute_cs2(floor=1e-6)
eos.validate()
eos.shrink_to_valid_nb()

eos.add_Ye_from_yq(overwrite=True)

out_file = "/home/mi58rip/gr-athena/eos_tables/LS220_240r_140t_50y_analmu_20120628_SVNr28_pycompose.h5"

eos.write_hdf5(out_file)

Ye_idx = np.argmin(np.abs(eos.yq - 0.5))
print(f"Ye index for Ye=0.5: {Ye_idx}, Ye value: {eos.yq[Ye_idx]}")
cold_eos_1d = (eos.slice_at_t_idx(0)).slice_at_y_idx(Ye_idx)
cold_eos_1d.add_coldslice(out_file)

cold_eos_1d.write_rns("/home/mi58rip/gr-athena/eos_tables/LS220_240r_140t_50y_analmu_20120628_SVNr28_pycompose.rns", truncate=True)