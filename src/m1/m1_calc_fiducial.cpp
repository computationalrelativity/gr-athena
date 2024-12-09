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
  // If v is spatial then:
  // v^a = (0, 1 / W u_til^i + 1 / alpha * beta^i)
  M1_GLOOP3(k,j,i)
  {
    Real const oo_W     = OO(fidu.sc_W(k,j,i));
    Real const oo_alpha = OO(geom.sc_alpha(k,j,i));

    for(int a=0; a<N; ++a)
    {
      fidu.sp_v_u(a,k,j,i) = (
        oo_W * fidu.sp_v_u(a,k,j,i) // + oo_alpha * geom.sp_beta_u(a,k,j,i)
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

  // geometric quantities
  const AT_C_sca & sc_alpha = geom.sc_alpha;
  // const AT_N_vec & sp_beta_u = geom.sp_beta_u;

  // required fiducial quantities
  AT_C_sca & sc_W   = fidu.sc_W;
  // AT_N_vec & sp_v_u = fidu.sp_v_u;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_C_sca & sc_chi  = lab_aux.sc_chi(ix_g,ix_s);

    AT_C_sca & sc_nG   = U_n.sc_nG(ix_g,ix_s);

    AT_C_sca & sc_J   = rad.sc_J(  ix_g,ix_s);
    AT_C_sca & sc_n   = rad.sc_n(  ix_g,ix_s);

    AT_D_vec & st_H_u = rad.st_H_u(ix_g, ix_s);

    M1_GLOOP3(k, j, i)
    {
      // We write:
      // sc_J = J_0
      // st_H = H_n n^alpha + H_v v^alpha + H_F F^alpha
      Real J_0, H_n, H_v, H_F;

      Assemble::Frames::ToFiducialExpansionCoefficients(
        *this,
        J_0, H_n, H_v, H_F,
        sc_chi, sc_E, sp_F_d,
        k, j, i
      );

      /*
      // Ensure we do not encounter zero-division
      J_0 = std::max(J_0, this->opt.fl_J);

      const Real W = sc_W(k,j,i);
      const Real oo_J_0 = OO(J_0);

      const Real sc_G__ = (
        W + oo_J_0 * H_n  // note sign switch due to n proj
      );

      sc_J(k,j,i) = J_0;
      sc_n(k,j,i) = sc_nG(k,j,i) / sc_G__;
      */


      const Real W = sc_W(k,j,i);

      const Real sc_G__ = (J_0 > 0)
        ? (W + H_n / J_0)
        : W;

      // J_0 = std::max(J_0, this->opt.fl_J);

      sc_J(k,j,i) = J_0;
      sc_n(k,j,i) = sc_nG(k,j,i) / sc_G__;

      /*
      const Real alpha = sc_alpha(k,j,i);
      const Real oo_alpha = OO(alpha);

      st_H_u(0,k,j,i) = H_n * oo_alpha;
      */
    }

  }

}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//