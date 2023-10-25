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
  '-*',
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
  "z4c_cx/tov_cx_ninterp2",
  "z4c_vc/tov_vc_ninterp2",
  "z4c_cx/tov_cx_ninterp2_hybridinterp",
  "z4c_vc/tov_vc_ninterp2_hybridinterp",
]

fn_hst = "gr_tov.hst"

fig_A, ax_A = plt.subplots(2, sharex=True, figsize=(6,8))

for tix, tag_sim in enumerate(sim_tags):
  labels, data = st.extensions.gra.hst_load([dir_base, tag_sim, fn_hst])

  ix_time    = labels.index("time")
  ix_mass    = labels.index("mass")
  ix_max_rho = labels.index("max-rho")

  T, mass, max_rho = data[:, ix_time], data[:, ix_mass], data[:, ix_max_rho]

  ax_A[0].semilogy(T, np.abs(1 - mass / mass[0]),
                   mkr_s[tix],
                   label=tag_sim,
                   color=col_s[tix],
                   ms=markersize)

  ax_A[1].plot(T, max_rho,
               mkr_s[tix],
               label=tag_sim,
               color=col_s[tix],
               ms=markersize)

  ax_A[0].legend(loc="upper left", fontsize="small")

  ax_A[0].set_ylabel("$|1 - M(t) / M(0)|$")
  ax_A[1].set_ylabel("$\\rho_c$")
  ax_A[1].set_xlabel("$t$")

  # ax_A.set_xlim([0, 12])
  # ax_A.set_ylim([1e-17, 1e-9])

  fig_A.align_ylabels()
  fig_A.tight_layout()


  plt.savefig(
    "fig_cx_vc.png",
    **plot_save_settings
  )


plt.show()


#
# :D
#
