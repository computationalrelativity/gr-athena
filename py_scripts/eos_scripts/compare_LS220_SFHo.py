import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

# Plotting rho vs P from SFHo.h5 file

SFHo_table_path = "/home/mi58rip/gr-athena/eos_tables/SFHo.h5"
LS220_table_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26.h5"


with h5py.File(SFHo_table_path, 'r') as f_SFHo, h5py.File(LS220_table_path, 'r') as f_LS220:
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

    Ye_LS220 = f_LS220['ye'][:]
    log_rho_LS220 = f_LS220['logrho'][:]
    log_P_LS220 = f_LS220['logpress'][:]
    log_Temp_LS220 = f_LS220['logtemp'][:]

# Temperature and Ye Pairs i want to plot for SFHo and LS220
pairs = [(0.1, 0.5), (1.0, 0.5), (10.0, 0.5), (100.0, 0.5)]

# plot the cold slice of SFHo 1D rho vs P
fig, ax = plt.subplots(figsize=(8, 6))

Colors = ['C0', 'C1', 'C2', 'C3']  # Define a list of colors for the pairs
for temp, Ye in pairs:
    # Find the index of the closest temperature and Ye in SFHo
    temp_index_SFHo = np.argmin(np.abs(Temp_MeV_SFHo - temp))
    Ye_index_SFHo = np.argmin(np.abs(Ye_SFHo - Ye))

    # Find the index of the closest temperature and Ye in LS220
    temp_index_LS220 = np.argmin(np.abs(10**log_Temp_LS220 - temp))
    Ye_index_LS220 = np.argmin(np.abs(Ye_LS220 - Ye))

    print(
        f"Requested T={temp}, Ye={Ye} | "
        f"SFHo: T={Temp_MeV_SFHo[temp_index_SFHo]}, "
        f"Ye={Ye_SFHo[Ye_index_SFHo]} | "
        f"LS220: T={10**log_Temp_LS220[temp_index_LS220]}, "
        f"Ye={Ye_LS220[Ye_index_LS220]}"
    )

    #use same color but different line style for SFHo and LS220
    ax.plot(rho_MeVfm3_SFHo*1.78266e12, P_MeVfm3_SFHo[:, Ye_index_SFHo, temp_index_SFHo]*1.60218e33, label='SFHo, T={} MeV, Ye={}'.format(temp, Ye), linestyle='--', color='C{}'.format(pairs.index((temp, Ye))))

    ax.plot(10**(log_rho_LS220), 10**(log_P_LS220[Ye_index_LS220, temp_index_LS220, :]), label='LS220, T={} MeV, Ye={}'.format(temp, Ye), linestyle='-', color='C{}'.format(pairs.index((temp, Ye))))

ax.set_xlabel(r'$\rho$ [gm/cm$^3$]', fontsize=14)
ax.set_ylabel(r'$P$ [dyne/cm$^2$]', fontsize=14)

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend()



plt.savefig("rho_vs_P_SFHo_LS220_Ye05.png", dpi=600)
plt.close(fig)

# if __name__ == "__main__":
#     main()
    