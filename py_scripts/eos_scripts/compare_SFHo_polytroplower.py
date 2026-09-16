import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

# Plotting rho vs P from SFHo.h5 file

SFHo_table_path = "/home/mi58rip/gr-athena/eos_tables/SFHo_with_cold_slice.h5"
SFHo_1D_rns_path = "/home/mi58rip/gr-athena/eos_tables/SFHo_200_wd_1e-10.rns"



with h5py.File(SFHo_table_path, 'r') as f_SFHo:
    # for name in f_LS220.keys():
    #     print(name)
    #     if "desc" in f_LS220[name].attrs:
    #         print(name, ":", f_LS220[name].attrs["desc"])

    mn_MeV_SFHo = f_SFHo['mn']
    nb_fm3_SFHo = f_SFHo['nb'][:]
    Q1_MeV_SFHo = f_SFHo['Q1'][:]
    Ye_SFHo = f_SFHo['yq'][:]

    Temp_MeV_SFHo = f_SFHo['t'][:]
    P_MeVfm3_SFHo = Q1_MeV_SFHo*nb_fm3_SFHo[:, None, None]
    rho_MeVfm3_SFHo = mn_MeV_SFHo*nb_fm3_SFHo[:]

rns_table = np.loadtxt(SFHo_1D_rns_path, skiprows=1)
rho_rns_cgs = rns_table[:, 0]
P_rns_cgs = rns_table[:, 1]

# plot the cold slice of SFHo 1D rho vs P
fig, ax = plt.subplots(figsize=(5, 4))

# Find the index of the closest temperature and Ye in SFHo
temp = 0.1 # MeV
Ye = 0.5

print(Temp_MeV_SFHo.min())
temp_index_SFHo = np.argmin(np.abs(Temp_MeV_SFHo - temp))
Ye_index_SFHo = np.argmin(np.abs(Ye_SFHo - Ye))

print(
    f"Requested T={temp}, Ye={Ye} | "
    f"SFHo: T={Temp_MeV_SFHo[temp_index_SFHo]}, "
    f"Ye={Ye_SFHo[Ye_index_SFHo]} | "
)

#use same color but different line style for SFHo and LS220
ax.plot(rho_MeVfm3_SFHo*1.78266e12, P_MeVfm3_SFHo[:, Ye_index_SFHo, temp_index_SFHo]*1.60218e33, label='SFHo', linestyle='--', color='navy')
ax.plot(rho_rns_cgs, P_rns_cgs, label='SFHo with polytrope (<= 1e8 g/cm^3)', linestyle='-', color='darkgreen')

ax.set_xlabel(r'$\rho$ [gm/cm$^3$]', fontsize=14)
ax.set_ylabel(r'$P$ [dyne/cm$^2$]', fontsize=14)

ax.set_xscale('log')
ax.set_yscale('log')

# put x limit while the y axis adjusting according to the data
# ax.set_xlim(rho_MeVfm3_SFHo.min()*1.78266e12, rho_MeVfm3_SFHo.max()*1.78266e12)
# ax.set_ylim(1e15, 1e36)

ax.legend()

ax.set_title('T={} MeV, Ye={}'.format(temp, Ye))

plt.tight_layout()
plt.savefig("SFHo_Comparison_polytrope_attached_1e8_1e-10.png", dpi=600)
plt.close(fig)

# if __name__ == "__main__":
#     main()
    