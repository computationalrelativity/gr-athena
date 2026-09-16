import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

table_1_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_2000pt.rns"
table_2_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_2000pt_poly_1.00e+08_1.00e-01.rns"
table_3_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_2000pt_poly_1.00e+08_1.00e-05.rns"
table_4_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_2000pt_poly_1.00e+08_1.00e-10.rns"
table_5_path = "/home/mi58rip/gr-athena/eos_tables/LS220_234r_136t_50y_analmu_20091212_SVNr26_pycompose_2000pt_poly_1.00e+08_1.00e+03.rns"
table_6_path = "/home/mi58rip/gr-athena/eos_tables/LS220_max.rns"
    
rns_table_1 = np.loadtxt(table_1_path, skiprows=1)
ed_rns_cgs_1 = rns_table_1[:, 0]
P_rns_cgs_1 = rns_table_1[:, 1]

rns_table_2 = np.loadtxt(table_2_path, skiprows=1)
ed_rns_cgs_2 = rns_table_2[:, 0]
P_rns_cgs_2 = rns_table_2[:, 1]

rns_table_3 = np.loadtxt(table_3_path, skiprows=1)
ed_rns_cgs_3 = rns_table_3[:, 0]
P_rns_cgs_3 = rns_table_3[:, 1]

rns_table_4 = np.loadtxt(table_4_path, skiprows=1)
ed_rns_cgs_4 = rns_table_4[:, 0]
P_rns_cgs_4 = rns_table_4[:, 1]

rns_table_5 = np.loadtxt(table_5_path, skiprows=1)
ed_rns_cgs_5 = rns_table_5[:, 0]
P_rns_cgs_5 = rns_table_5[:, 1]

rns_table_6 = np.loadtxt(table_6_path, skiprows=1)
ed_rns_cgs_6 = rns_table_6[:, 0]
P_rns_cgs_6 = rns_table_6[:, 1]

# plot the cold slice of SFHo 1D rho vs P
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(ed_rns_cgs_1, P_rns_cgs_1, label='LS220', linestyle='-', color='navy')
# ax.plot(ed_rns_cgs_2, P_rns_cgs_2, label='LS220 poly_1e8_1e-1', linestyle='-.', color='darkgreen')
# ax.plot(ed_rns_cgs_4, P_rns_cgs_4, label='LS220 poly_1e8_1e-5', linestyle='--', color='orange')
# ax.plot(ed_rns_cgs_3, P_rns_cgs_3, label='LS220 poly_1e8_1e-10', linestyle='-.', color='red')
ax.plot(ed_rns_cgs_5, P_rns_cgs_5, label='LS220 poly_1e8_1e+3', linestyle='--', color='purple')
ax.plot(ed_rns_cgs_6, P_rns_cgs_6, label='LS220 max', linestyle='-.', color='black')

ax.set_xlabel(r'$E$ [gm/cm$^3$]', fontsize=14)
ax.set_ylabel(r'$P$ [dyne/cm$^2$]', fontsize=14)

ax.set_xscale('log')
ax.set_yscale('log')

# ax.set_xbound(1e2, 1e17)

ax.legend()

ax.set_title('T=0.01 MeV, Ye=0.5')

plt.tight_layout()
plt.savefig("/home/mi58rip/gr-athena/plots/LS220_Comparison.png", dpi=600)
plt.close(fig)
    