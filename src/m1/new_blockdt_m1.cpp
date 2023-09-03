//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt_z4c.cpp
//  \brief computes timestep using CFL condition on a MeshBlock

// C/C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "m1.hpp"
#include "../z4c/z4c.hpp"

//! \fn Real Z4c::NewBlockTimeStep(void)
//  \brief calculate the minimum timestep within a MeshBlock
Real M1::NewBlockTimeStep(void) {
  MeshBlock * pmb = pmy_block;
  int tid = 0;

  Real min_dt = (FLT_MAX);

  // Just slice dt1_, dt2_, dt3_ directly
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  //NB have changed from leq mbi.ku to lt mbi.ku - CenterWidth etc live on 
  //cell centered grid, which have 1 less grid point
  for (int k = mbi.kl; k < mbi.ku; ++k) {
    for (int j = mbi.jl; j < mbi.ju; ++j) {

      // Note: these should be uniform for Cart. and equiv. to dxn
      pmb->pcoord->CenterWidth1(k, j, mbi.il, mbi.iu, dt1);
      pmb->pcoord->CenterWidth2(k, j, mbi.il, mbi.iu, dt2);
      pmb->pcoord->CenterWidth3(k, j, mbi.il, mbi.iu, dt3);

      for (int i = mbi.il; i < mbi.iu; ++i) {
        // Real & dt_1 = dt1(i);
        Real & dt_1 = pmb->pcoord->dx1f(i);
        min_dt = std::min(min_dt, dt_1);
      }
      if (pmb->block_size.nx2 > 1) {
        for (int i = mbi.il; i < mbi.iu; ++i) {
//        Real & dt_2 = dt2(i);
          Real & dt_2 = pmb->pcoord->dx2f(j);
          min_dt = std::min(min_dt, dt_2);
        }
      }
      if (pmb->block_size.nx3 > 1) {
        for (int i = mbi.il; i < mbi.iu; ++i) {
//        Real & dt_3 = dt3(i);
          Real & dt_3 = pmb->pcoord->dx3f(k);
          min_dt = std::min(min_dt, dt_3);
        }
      }
    }
  }

  // FIXME: this should only set the new dt if it is smaller than that already
  // set by the hydro and spacetime modules
  min_dt *= pmb->pmy_mesh->cfl_number;
  pmb->new_block_dt_ = min_dt;

  return min_dt;
}
