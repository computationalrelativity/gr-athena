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

void M1::CalcFiducialVelocity()
{
  using namespace LinearAlgebra;
  MeshBlock * pmb = pmy_block;

  switch (opt.fiducial_velocity)
  {
    case opt_fiducial_velocity::fluid:
    {
      M1_GLOOP2(k,j)
      for (int a=0; a<N; ++a)
      {
        M1_GLOOP1(i)
        {
          fidu.vel_u(a,k,j,i) = pmb->phydro->w(IVX+a,k,j,i);
        }
      }
      break;
    }
    case opt_fiducial_velocity::mixed:
    {
      M1_GLOOP2(k,j)
      for (int a=0; a<N; ++a)
      {
        M1_GLOOP1(i)
        {
          Real const rho = pmb->phydro->w(IDN,k,j,i);
          Real const fac = 1.0/std::max(rho, opt.fiducial_velocity_rho_fluid);
        	fidu.vel_u(a,k,j,i) = pmb->phydro->w(IVX+a,k,j,i) * rho * fac;
        }
      }

      break;
    }
    case opt_fiducial_velocity::zero:
    {
      fidu.vel_u.ZeroClear();
      fidu.Wlorentz.Fill(1.0);
      return;
    }
    case opt_fiducial_velocity::none:
    {
      return;
    }
  }

  // Have: fidu.vel_u = utilde^i = W v^i
  // Want: fidu.vel_u = v^i
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities -------------------------------------------
  AT_N_sym sl_adm_gamma_dd(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_vec sl_w_util_u(    pmb->phydro->w, IVX);

  // various scratches --------------------------------------------------------
  AT_N_sym adm_gamma_dd_(mbi.nn1); // gamma_{ij}
  AT_N_sca W_(       mbi.iu+1);
  AT_N_vec w_util_u_(mbi.iu+1);

  M1_GLOOP2(k,j)
  {
    // Note: internally maps geometric sampling to matter sampling (& M1)
    pmy_coord->GetGeometricFieldCC(adm_gamma_dd_, sl_adm_gamma_dd, k, j);

    // prepare scratch
    for (int a=0; a<NDIM; ++a)
    for (int i=mbi.il; i<=mbi.iu; ++i)
    {
      w_util_u_(a,i) = sl_w_util_u(a,k,j,i);
    }

    for (int i=mbi.il; i<=mbi.iu; ++i)
    {
      const Real norm2_util = InnerProductSlicedVec3Metric(
        w_util_u_, adm_gamma_dd_, i
      );

      fidu.Wlorentz(k,j,i) = std::sqrt(1. + norm2_util);

      Real const oo_W = 1.0 / fidu.Wlorentz(k,j,i);
      for(int a = 0; a < NDIM; ++a)
      {
        fidu.vel_u(a,k,j,i) = oo_W * fidu.vel_u(a,k,j,i);
      }
    }
  }
}
