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
import matplotlib

matplotlib.use('QtAgg')

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
fn_tpl="dummy_3d_weak.inp.tpl"  # template to use
fn_par="input.inp"              # filename to use after parsing
fn_scr="run.sh"                 # script to generate and run

X_M = 1536
S_L = 10
S_Rs = [4, 6]
S_AB = 8

N_M_max = 320
N_Bs = [16, 24, 32]
# =============================================================================

ix_c = 0

for S_R in S_Rs:

  for N_B in N_Bs:
    N_M_base = N_B

    Num_MB  = []
    Sum_N_M = []

    IX = np.array((1, 1, 1))

    i = 0
    while True:

      IX[np.mod(i, 3)] += 1
      N_M_ = (N_M_base * IX[0], N_M_base * IX[1], N_M_base * IX[2])

      if N_M_[0] > N_M_max:
        break

      i += 1

      print(np.sum(N_M_), N_M_)

      # assemble commands & tpl replacements ==================================
      dir_base = st.core.io.path_to_absolute(dir_base)
      dir_par = st.core.io.path_to_absolute(dir_par)

      # load the template, perform replacements and write
      raw_tpl = st.core.io.raw_load([dir_par, fn_tpl])

      print(f"N_M={N_M_}, N_B={N_B}")
      replacement_rules = {
        "N_M_1": N_M_[0],
        "N_M_2": N_M_[1],
        "N_M_3": N_M_[2],
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
      Sum_N_M.append(np.sum(N_M_))


    plt.semilogy(np.array(Sum_N_M) / 3, Num_MB, ".-",
                label=f"N_B={N_B}, S_R={replacement_rules['S_R']}",
                color=col_s[ix_c])
    ix_c += 1

print("done")


plt.title(f"X_M={X_M}, S_L={S_L}, S_AB={S_AB}")

plt.xlabel("sum(N_M_i, i) / 3")
plt.ylabel("#MB")
plt.legend(loc="upper left")

plt.savefig("weak.png")

print("done")
