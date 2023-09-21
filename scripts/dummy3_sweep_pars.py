#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_)

@author: Boris Daszuta
@function: Iterate grid configurations

Note:
Run from scripts folder after compiling with dummy3.sh
"""
# export PYTHONPATH=${repos}/numerical_relativity/simtroller:${PYTHONPATH}
import simtroller as st
import numpy as np
import matplotlib.pyplot as plt

import subprocess

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


# settings ====================================================================
dir_par="${PWD}/problems"
dir_base="${PWD}/../outputs/dummy_3d/cx"

fn_exec="dummy_cx_cx.x"
fn_tpl="dummy_3d.inp.tpl"  # template to use
fn_par="input.inp"         # filename to use after parsing
fn_scr="run.sh"            # script to generate and run

X_M = 1536
S_L = 10
S_R = 4
S_AB = 8

# for N_B = 16
N_M = np.array([32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
  208, 224, 240, 256])

# N_M = np.array([32,  48,  64, 80, 96, 112, 128])
N_B = 16

N_pars = {
  16: np.array([16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192,
       208, 224, 240, 256, 272, 288, 304, 320]),
  24: np.array([24,  48,  72,  96, 120, 144, 168, 192, 216, 240, 264, 288,
       312, 336]),
  32: np.array([32,  64,  96, 128, 160, 192, 224, 256, 288, 320])
}
# =============================================================================


# main loop -------------------------------------------------------------------
res = {}

ix = 0
for col, tup in zip(col_s, N_pars.items()):
  Num_MB = []

  N_B, N_M = tup

  # assemble commands & tpl replacements ======================================
  dir_base = st.core.io.path_to_absolute(dir_base)
  dir_par = st.core.io.path_to_absolute(dir_par)

  # load the template, perform replacements and write
  raw_tpl = st.core.io.raw_load([dir_par, fn_tpl])

  for N_M_ in N_M:
    print(f"N_M={N_M_}, N_B={N_B}")
    replacement_rules = {
      "N_M": N_M_,
      "N_B": N_B,
      "S_R": S_R,    # spherical radii
      "S_AB": S_AB,  # centeres at +-
      "S_L": S_L,    # level number
      "X_M": X_M
    }

    par_tpl = st.core.primitives.str_delimiter_replace(
      raw_tpl, replacement_rules, delimiters=("[[", "]]")
    )

    fn_par = st.core.io.path_to_absolute([dir_base, fn_par],
                                          verify=False)
    st.core.io.raw_dump(par_tpl, fn_par)

    # assemble base command and run
    out = subprocess.check_output([dir_base + "/" + fn_exec, '-i', fn_par])
    f_b = b"Number of MeshBlocks = "
    ix_L = out.rfind(f_b)
    ix_R = out[ix_L:].rfind(b";")

    # awful..
    Num_MB_ = int(out[ix_L + len(f_b):][:ix_R-len(f_b)])
    Num_MB.append(Num_MB_)


  res[ix] = Num_MB
  plt.semilogy(N_M, Num_MB, ".-",
               label=f"N_B={N_B}, S_R={replacement_rules['S_R']}",
               color=col)

  ix += 1

plt.title(f"X_M={X_M}, S_L={S_L}, S_AB={S_AB}")

plt.xlabel("N_M")
plt.ylabel("#MB")
plt.legend(loc="upper left")