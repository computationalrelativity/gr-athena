#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_)

@author: Boris Daszuta
@function: Parse GR-Athena++ scalar data

Note(s):
- Needs simtroller
- run from debug dir.
- utilize "z4c_tov_cx(vc).sh" from scripts dir
- ensure data dirs nested as in "sim_tags" below...
"""
###############################################################################
# python imports
import numpy as np
import matplotlib.pyplot as plt

# package imports
import simtroller as st
###############################################################################

st.extensions.plot_set_defaults()
# pool = st.core.parallel_pool_start() # spin up a worker pool

# pick colors consistently
col_s = [
  '#1f77b4', # blue    0
  '#ff7f0e', # orange  1
  '#2ca02c', # green   2
  '#d62728', # red     3
  '#9467bd', # purple  4
  '#8c564b', # brown   5
  'black',
  'cyan',
  'lime'
]

mkr_s = [
  '-',
  '--*',
  '--o',
  '--s',
  '--',
  '-d',
  '-',
  '-',
  '-'
]

plot_save_settings = {
  'dpi': 600,
  'pad_inches': 0,
  'bbox_inches': 'tight'
}

linewidth_thin = 0.6
linewidth_thick = 0.9
linewidth_vthick = 1.4
markersize = 0.7
# make style consistent
# plt.style.use('matplotlibrc')


# -----------------------------------------------------------------------------
# specify directory containing simulation output in athdf
dir_base = "${PWD}/../outputs/"
sim_tags = [
  # "z4c_cx/tov_cx_ninterp1",
  # "z4c_vc.n/tov_vc_ninterp1",
  # "z4c_vc/tov_vc_ninterp1.32.x2",
  # "z4c_vc/tov_vc_ninterp1.64.x2",
  # "z4c_vc/tov_vc_ninterp2_hybridinterp",
  "z4c_vc/tov_vc_ninterp2.64.32.X3",
]

fn_hst = "gr_tov.hst"

fig_A, ax_A = plt.subplots(3, sharex=True, figsize=(12,8))

data_diff = []

for tix, tag_sim in enumerate(sim_tags):
  labels, data = st.extensions.gra.hst_load([dir_base, tag_sim, fn_hst])

  ix_time    = labels.index("time")
  ix_mass    = labels.index("mass")
  ix_max_rho = labels.index("max-rho")
  ix_H       = labels.index("H-norm2")

  T, mass, max_rho = data[:, ix_time], data[:, ix_mass], data[:, ix_max_rho]

  Ham = np.sqrt(data[:, ix_H])

  ax_A[0].semilogy(T, np.abs(1 - mass / mass[0]),
                   mkr_s[tix],
                   label=tag_sim,
                   color=col_s[tix],
                   ms=markersize)

  ax_A[1].plot(T, max_rho - max_rho[0],
               mkr_s[tix],
               label=tag_sim,
               color=col_s[tix],
               ms=markersize)

  data_diff.append(max_rho - max_rho[0])

  # ax_A[2].semilogy(T, Ham,
  #                  mkr_s[tix],
  #                  label=tag_sim,
  #                  color=col_s[tix],
  #                  ms=markersize)

ax_A[2].semilogy(T, np.abs(data_diff[0] - data_diff[1]),
                 label="|32-64|")
data_diff.pop()
data_diff.pop()

# cross-branch ----------------------------------------------------------------
dir_base_MTE = "${PWD}/../../matter_tracker_extrema/outputs/"
sim_tags_MTE = [
  # "z4c_vc/tov_vc_ninterp1.32.x2",
  # "z4c_vc/tov_vc_ninterp1.64.x2",
  # "z4c_vc/tov_vc_ninterp2_hybridinterp",
  # "z4c_vc/tov_vc_ninterp2",
  "z4c_vc/tov_vc_ninterp2.64.32.X3",
]

for uix, tag_sim in enumerate(sim_tags_MTE):
  labels, data = st.extensions.gra.hst_load([dir_base_MTE, tag_sim, fn_hst])

  ix_time    = labels.index("time")
  ix_mass    = labels.index("mass")
  ix_max_rho = labels.index("max_rho")
  ix_H       = labels.index("H-norm2")

  T, mass, max_rho = data[:, ix_time], data[:, ix_mass], data[:, ix_max_rho]

  Ham = np.sqrt(data[:, ix_H])

  ax_A[0].semilogy(T, np.abs(1 - mass / mass[0]),
                   mkr_s[uix+tix+1],
                   label="MTE:" + tag_sim,
                   color=col_s[uix+tix+1],
                   ms=markersize)

  ax_A[1].plot(T, max_rho - max_rho[0],
               mkr_s[uix+tix+1],
               label="MTE:" + tag_sim,
               color=col_s[uix+tix+1],
               ms=markersize)

  data_diff.append(max_rho - max_rho[0])

  # ax_A[2].semilogy(T, Ham,
  #                  mkr_s[uix+tix+1],
  #                  label="MTE:" + tag_sim,
  #                  color=col_s[uix+tix+1],
  #                  ms=markersize)

ax_A[2].semilogy(T, np.abs(data_diff[0] - data_diff[1]),
                 label="MTE: |32-64|")
data_diff.pop()
data_diff.pop()


ax_A[0].legend(loc="lower right", fontsize="small")

ax_A[0].set_ylabel("$|1 - M(t) / M(0)|$")
ax_A[1].set_ylabel("$\\rho_c - \\rho_c(0)$")
# ax_A[2].set_ylabel("$\Vert \mathcal{H} \Vert$")
ax_A[1].set_ylim([-51e-5, 5e-5])

ax_A[2].set_ylabel("$||\\rho_c - \\rho_c(0)|_{32}-|\\rho_c - \\rho_c(0)|_{64}|$")
ax_A[2].set_xlim([0, 100])
ax_A[2].legend(loc="lower right", fontsize="small")
ax_A[2].set_xlabel("$t$")



fig_A.align_ylabels()
fig_A.tight_layout()

plt.savefig(
  "fig_cx_vc_vs_MTE_LO_diff.png",
  **plot_save_settings
)


plt.show()


#
# :D
#
