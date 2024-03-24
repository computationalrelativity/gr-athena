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


fns = [
  # "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_0.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_1.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_2.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_0.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_1.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateHydroFlux.flux_2.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Primitives.u.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Primitives.w.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Primitives.storage.u.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Primitives.storage.adm.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.Primitives.u.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.Primitives.w.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.Primitives.storage.u.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.Primitives.storage.adm.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.u.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.u.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.rhs.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.rhs.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.mat.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.CalculateZ4cRHS.storage.mat.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.UpdateSource.storage.adm.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.UpdateSource.storage.mat.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.UpdateSource.phydro.w.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.UpdateSource.storage.adm.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.UpdateSource.storage.mat.post.tov",
  "z4c_vc/tov_vc_ninterp1/data.UpdateSource.phydro.w.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Z4cToADM.storage.u.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Z4cToADM.storage.adm.pre.tov",
  # "z4c_vc/tov_vc_ninterp1/data.Z4cToADM.storage.u.post.tov",
  # "z4c_vc/tov_vc_ninterp1/data.EnforceAlgConstr.storage.u.pre.tov",
  "z4c_vc/tov_vc_ninterp1/data.EnforceAlgConstr.storage.u.post.tov",
]

for fn in fns:

  dir_base = "${PWD}/../outputs/"

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

  print(f"{np.abs((dat_A - dat_B).flatten()).max():1.3e} @ {fn}")
  print(f"{np.abs((dat_A - dat_B)[0,:,4:-4,4:-4,4:-4].flatten()).max():1.3e} @ {fn}")

#
# :D
#
