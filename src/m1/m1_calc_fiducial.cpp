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
#include "m1_utils.hpp"

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
      // In the case that we have fluid then we can slice auxiliaries
#if FLUID_ENABLED
      M1_GLOOP3(k,j,i)
      {
        fidu.sc_W(k,j,i) = pmb->phydro->derived_ms(IX_LOR,k,j,i);
      }

      M1_GLOOP2(k,j)
      for (int a=0; a<M1_NDIM; ++a)
      {
        M1_GLOOP1(i)
        {
          fidu.sp_v_u(a,k,j,i) = hydro.sp_w_util_u(a,k,j,i) / fidu.sc_W(k,j,i);
        }
      }

      // map to form
      fidu.sp_v_d.ZeroClear();

      M1_GLOOP2(k,j)
      {
        for (int a=0; a<N; ++a)
        for (int b=0; b<N; ++b)
        M1_GLOOP1(i)
        {
          fidu.sp_v_d(a,k,j,i) += geom.sp_g_dd(a,b,k,j,i) * fidu.sp_v_u(b,k,j,i);
        }
      }

      return;
#else
      // computation of, and rescaling by, W below
      M1_GLOOP2(k,j)
      for (int a=0; a<M1_NDIM; ++a)
      {
        M1_GLOOP1(i)
        {
          fidu.sp_v_u(a,k,j,i) = hydro.sp_w_util_u(a,k,j,i);
        }
      }
      break;
#endif // FLUID_ENABLED
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
      fidu.sc_W.Fill(1.);
      fidu.sp_v_u.ZeroClear();
      fidu.sp_v_d.ZeroClear();
      return;
    }
    case opt_fiducial_velocity::none:
    {
      return;
    }
  }

  // Have: fidu.sp_v_u = utilde^i = W v^i
  // Want: fidu.sp_v_u = v^i
  //
  // Fiducial velocity does not necessarily coincide with hydro, recompute
  // Lorentz factor.

  M1_GLOOP3(k,j,i)
  {
    const Real norm2_util = InnerProductVecMetric(
      fidu.sp_v_u, geom.sp_g_dd,
      k,j,i
    );
    fidu.sc_W(k,j,i) = std::sqrt(1. + norm2_util);
  }


  // rescale fluid velocity
  M1_GLOOP3(k,j,i)
  {
    Real const oo_W = OO(fidu.sc_W(k,j,i));
    // Real const oo_alpha = OO(geom.sc_alpha(k,j,i));

    for(int a=0; a<N; ++a)
    {
      fidu.sp_v_u(a,k,j,i) = (
        oo_W * fidu.sp_v_u(a,k,j,i) //  + oo_alpha * geom.sp_beta_u(a,k,j,i)
      );
    }
  }


  // map to form
  fidu.sp_v_d.ZeroClear();

  M1_GLOOP2(k,j)
  {
    for (int a=0; a<N; ++a)
    for (int b=0; b<N; ++b)
    M1_GLOOP1(i)
    {
      fidu.sp_v_d(a,k,j,i) += geom.sp_g_dd(a,b,k,j,i) * fidu.sp_v_u(b,k,j,i);
    }
  }
}

// ----------------------------------------------------------------------------
void M1::CalcFiducialFrame(AthenaArray<Real> & u)
{
  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U_n);

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_C_sca & sc_chi  = lab_aux.sc_chi(ix_g,ix_s);

    AT_C_sca & sc_nG   = U_n.sc_nG(ix_g,ix_s);

    AT_C_sca & sc_J   = rad.sc_J(  ix_g,ix_s);
    AT_D_vec & st_H_u = rad.st_H_u(ix_g, ix_s);
    AT_C_sca & sc_n   = rad.sc_n(  ix_g,ix_s);

    AT_C_sca & sc_avg_nrg = radmat.sc_avg_nrg(ix_g,ix_s);

    M1_GLOOP3(k, j, i)
    {
      Assemble::Frames::ToFiducial(
        *pm1,
        sc_J, st_H_u, sc_n,
        sc_chi,
        sc_E, sp_F_d, sc_nG,
        k, j, i
      );

      // Fiducial frame expression
      // sc_avg_nrg(k,j,i) = (sc_n(k,j,i) > 0)
      //   ? sc_J(k,j,i) / sc_n(k,j,i)
      //   : 0.0;

      // Eulerian expression
      // Real dotFv (0.0);
      // for (int a=0; a<N; ++a)
      // {
      //   dotFv += sp_F_d(a,k,j,i) * fidu.sp_v_u(a,k,j,i);
      // }
      // const Real W = pm1->fidu.sc_W(k,j,i);
      // sc_avg_nrg(k,j,i) = W / sc_nG(k,j,i) * (sc_E(k,j,i) - dotFv);
    }

  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//