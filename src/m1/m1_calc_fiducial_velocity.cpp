//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_fiducial_velocity.cpp
//  \brief computes the fiducial velocity (CC) on a mesh block 

// C++ standard headers
//#include <cmath> 
#include <sstream>
#include <iostream>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/linear_algebra.hpp"
#include "m1.hpp"
#include "m1_macro.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

void M1::CalcFiducialVelocity()
{
  using namespace LinearAlgebra;
  MeshBlock * pmb = pmy_block;

  switch (opt.fiducial_velocity)
  {
    case opt_fiducial_velocity::fluid:
    {
      M1_GLOOP2(k,j)
      for (int a=0; a<M1_NDIM; ++a)
      {
        M1_GLOOP1(i)
        {
          fidu.sp_v_u(a,k,j,i) = hydro.sp_w_util_u(a,k,j,i);
        }
      }
      break;
    }
    case opt_fiducial_velocity::mixed:
    {
      M1_GLOOP2(k,j)
      for (int a=0; a<M1_NDIM; ++a)
      {
        M1_GLOOP1(i)
        {
          Real const rho = hydro.sc_w_rho(k,j,i);
          Real const fac = 1.0/std::max(rho, opt.fiducial_velocity_rho_fluid);
        	fidu.sp_v_u(a,k,j,i) = hydro.sp_w_util_u(a,k,j,i) * rho * fac;
        }
      }

      break;
    }
    case opt_fiducial_velocity::zero:
    {
      fidu.sp_v_u.ZeroClear();
      return;
    }
    case opt_fiducial_velocity::none:
    {
      return;
    }
  }

  // Have: fidu.sp_v_u = utilde^i = W v^i
  // Want: fidu.sp_v_u = v^i
  M1_GLOOP3(k,j,i)
  {
    Real const oo_W = 1.0 / hydro.sc_W(k,j,i);
    for(int a = 0; a < NDIM; ++a)
    {
      fidu.sp_v_u(a,k,j,i) = oo_W * fidu.sp_v_u(a,k,j,i);
    }
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//