#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_)

@author: Boris Daszuta
@function: Given spherically symmetric problem inspect fluxes

Note(s):
- Single block only.
"""

import numpy as np
import numpy.typing as npt
import typing as _t
import matplotlib.pyplot as plt

import simtroller as st

def get_dim_info(fn : str) -> tuple[int, ...]:
  """
  Extract dimension information from ascii dump.
  """
  to_replace = ("#", )

  with open(fn, "r") as fh:
    str_dim_info = fh.readline()

  str_dim_info = str_dim_info.strip("\n")
  for tr in to_replace:
    str_dim_info = str_dim_info.replace(tr, "")

  return tuple([int(d) for d in str_dim_info.split(" ") if len(d) > 0])

def load_data(fn : str) -> npt.NDArray[_t.Any]:
  """
  Load data from text dump.
  """
  return np.loadtxt(
    fn, comments="#"
  )

# comparison ------------------------------------------------------------------

# grids
dir_base = "${PWD}/../outputs/"
fns = [
  "z4c_vc/tov_vc_ninterp1/z4c_x1",
  "z4c_vc/tov_vc_ninterp1/z4c_x2",
  "z4c_vc/tov_vc_ninterp1/z4c_x3",
]
x = []
for fn in fns:
  fn_x = st.core.io.path_to_absolute([dir_base, fn])

  dim_x = get_dim_info(fn_x)[::-1]
  dat_x = load_data(fn_x)
  x.append(dat_x)

dir_base = "${PWD}/../outputs/"
# dir_base = "${PWD}/../../matter_tracker_extrema/outputs/"
fns = [
  "z4c_vc/tov_vc_ninterp1/data.mtask.flux_0.tov",
  "z4c_vc/tov_vc_ninterp1/data.mtask.flux_1.tov",
  "z4c_vc/tov_vc_ninterp1/data.mtask.flux_2.tov"
]

fl_dat_A = []

for fn in fns:
  fn_A = st.core.io.path_to_absolute([dir_base, fn])

  dim_A = get_dim_info(fn_A)[::-1]
  # dim_A = get_dim_info(fn_A)
  dat_A = load_data(fn_A)

  # allow multiple dumps in single file
  sh_A = dat_A.shape
  if len(sh_A) == 1:
    sh_A = [1, dat_A.size]

  dim_A = tuple(
    [d for d in
     st.core.primitives.sequence_generator_flatten((sh_A[0], dim_A))]
  )

  fl_dat_A.append(dat_A.reshape(dim_A))

# transpose to match flux[0]: (ix_dat, ix_hyd, ix_z, ix_y, ix_x)
ix_dat = 0
tr_flux_0 = fl_dat_A[0][ix_dat]
tr_flux_1 = np.transpose(fl_dat_A[1][ix_dat], (0, 3, 1, 2))
tr_flux_2 = np.transpose(fl_dat_A[2][ix_dat], (0, 3, 2, 1))

# 0 : (z,y,x) -> (z,y,x)
# 1 : (z,y,x) -> (x,z,y)
# 2 : (z,y,x) -> (x,y,z)

# scalar fields:
print("scalar abs_diff")
for six in (0, 4):
  print(f"hyd[{six}]: max|tr(flux[0]) - tr(flux[1])|:",
        np.abs(tr_flux_0[six] - tr_flux_1[six]).max())

  print(f"hyd[{six}]: max|tr(flux[0]) - tr(flux[2])|:",
        np.abs(tr_flux_0[six] - tr_flux_2[six]).max())

# permuted vectors:

ix_O = 1
# (0,x) vs (1,y)
diff_01_A = np.abs(tr_flux_0[ix_O+0] - tr_flux_1[ix_O+1]).max()
# (0,y) vs (1,z)
diff_01_B = np.abs(tr_flux_0[ix_O+1] - tr_flux_1[ix_O+2]).max()
# (0,z) vs (1,x)
diff_01_C = np.abs(tr_flux_0[ix_O+2] - tr_flux_1[ix_O+0]).max()

# (0,x) vs (2,z)
diff_02_A = np.abs(tr_flux_0[ix_O+0] - tr_flux_2[ix_O+2]).max()
# (0,y) vs (2,y)
diff_02_B = np.abs(tr_flux_0[ix_O+1] - tr_flux_2[ix_O+1]).max()
# (0,z) vs (2,x)
diff_02_C = np.abs(tr_flux_0[ix_O+2] - tr_flux_2[ix_O+0]).max()

print("vectorial abs_diff + perm")
print("hyd: (0,x) vs (1,y):", diff_01_A)
print("hyd: (0,y) vs (1,z):", diff_01_B)
print("hyd: (0,z) vs (1,x):", diff_01_C)

print("hyd: (0,x) vs (2,z):", diff_02_A)
print("hyd: (0,y) vs (2,y):", diff_02_B)
print("hyd: (0,z) vs (2,x):", diff_02_C)


#
# :D
#
