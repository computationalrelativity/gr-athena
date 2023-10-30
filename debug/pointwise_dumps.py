#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_)

@author: Boris Daszuta
@function: ...

Note(s):
- ...
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
fns = [
  # "z4c_vc/tov_vc_ninterp1/data.mtask.w_pf.tov",
  "z4c_vc/tov_vc_ninterp1/data.mtask.w.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolve.flux_l.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.gamma_3.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.pgas_l_first.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.prim_r.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.flux_2.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.storage.u.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.storage.mat.tov",
]

for fn in fns:
  fn_A = st.core.io.path_to_absolute([dir_base, fn])

  dim_A = get_dim_info(fn_A)[::-1]
  dat_A = load_data(fn_A)

  # allow multiple dumps in single file
  sh_A = dat_A.shape
  if len(sh_A) == 1:
    sh_A = [1, dat_A.size]

  dim_A = tuple(
    [d for d in
     st.core.primitives.sequence_generator_flatten((sh_A[0], dim_A))]
  )
  dat_A = dat_A.reshape(dim_A)

dir_base_MTE = "${PWD}/../../matter_tracker_extrema/outputs/"
fns = [
  # "z4c_vc/tov_vc_ninterp1/data.mtask.w_pf.tov",
  "z4c_vc/tov_vc_ninterp1/data.mtask.w.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolve.flux_l.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.gamma_3.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.pgas_l_first.tov",
  # "z4c_vc/tov_vc_ninterp1/data.rsolver.prim_r.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.flux_2.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.storage.u.tov",
  # "z4c_vc/tov_vc_ninterp1/data.mtask.storage.mat.tov",
]

for fn in fns:
  fn_B = st.core.io.path_to_absolute([dir_base_MTE, fn])

  # d-set dims
  dim_B = get_dim_info(fn_B)[::-1]
  dat_B = load_data(fn_B)

  # allow multiple dumps in single file
  sh_B = dat_B.shape
  if len(sh_B) == 1:
    sh_B = [1, dat_B.size]

  dim_B = tuple(
    [d for d in
     st.core.primitives.sequence_generator_flatten((sh_B[0], dim_B))]
  )
  dat_B = dat_B.reshape(dim_B)


print("max|dat_A - dat_B|:", np.abs((dat_A - dat_B).flatten()).max())

#
# :D
#
