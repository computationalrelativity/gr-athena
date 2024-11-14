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
      fidu.st_v_u.ZeroClear();
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

  /*
  // Lorentz factor (if fid not fluid derived)
  if (opt.fiducial_velocity == opt_fiducial_velocity::fluid)
  {
    M1_GLOOP3(k,j,i)
    {
      fidu.sc_W(k,j,i) = hydro.sc_W(k,j,i);
    }
  }
  else
  {
    M1_GLOOP2(k,j)
    {
      for (int i=mbi.il; i<=mbi.iu; ++i)
      {
        const Real norm2_util = InnerProductVecMetric(
          fidu.sp_v_u, geom.sp_g_dd,
          k,j,i
        );
        fidu.sc_W(k,j,i) = std::sqrt(1. + norm2_util);
      }
    }
  }
  */

  // rescale fluid velocity
  M1_GLOOP3(k,j,i)
  {
    Real const oo_W = 1.0 / fidu.sc_W(k,j,i);
    for(int a=0; a<N; ++a)
    {
      fidu.sp_v_u(a,k,j,i) = oo_W * fidu.sp_v_u(a,k,j,i);
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

  // space-time extension
  fidu.st_v_u.ZeroClear();

  AT_D_sym & st_g_uu_ = scratch.st_g_uu_;

  M1_GLOOP2(k,j)
  {
    Assemble::st_g_uu_(this, st_g_uu_, k, j, 0, mbi.nn1-1);

    scratch.st_vec_.ZeroClear();

    for(int a=0; a<N; ++a)
    M1_GLOOP1(i)
    {
      scratch.st_vec_(0,i) += fidu.sp_v_u(a,k,j,i) * geom.sp_beta_d(a,k,j,i);
    }

    for(int a=0; a<N; ++a)
    for(int b=0; b<N; ++b)
    M1_GLOOP1(i)
    {
      scratch.st_vec_(1+a,i) += geom.sp_g_dd(a,b,k,j,i) * fidu.sp_v_u(b,k,j,i);
    }

    for(int a=0; a<D; ++a)
    for(int b=0; b<D; ++b)
    M1_GLOOP1(i)
    {
      fidu.st_v_u(a,k,j,i) += st_g_uu_(a,b,i) * scratch.st_vec_(a,i);
    }
  }

}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
//
// This isn't needed for the core algorithm but may be for data-dumping
void M1::CalcFiducialFrame(AthenaArray<Real> & u)
{
  vars_Lab U_n { {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS}, {N_GRPS,N_SPCS} };
  SetVarAliasesLab(u, U_n);

  AT_C_sca & sc_W = fidu.sc_W;
  AT_N_vec & sp_v_u = fidu.sp_v_u;
  AT_N_vec & sp_v_d = fidu.sp_v_d;

  // point to scratches
  AT_C_sca & dotFv_ = scratch.sc_A_;
  AT_C_sca & sc_W2_ = scratch.sc_B_;

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    AT_C_sca & sc_E    = U_n.sc_E(  ix_g,ix_s);
    AT_N_vec & sp_F_d  = U_n.sp_F_d(ix_g,ix_s);
    AT_N_sym & sp_P_dd = lab_aux.sp_P_dd(ix_g,ix_s);

    AT_C_sca & sc_nG   = U_n.sc_nG(ix_g,ix_s);

    AT_C_sca & sc_H_t = rad.sc_H_t(ix_g,ix_s);
    AT_N_vec & sp_H_d = rad.sp_H_d(ix_g,ix_s);
    AT_C_sca & sc_J   = rad.sc_J(  ix_g,ix_s);
    AT_C_sca & sc_n   = rad.sc_n(  ix_g,ix_s);

    M1_GLOOP2(k,j)
    {
      dotFv_.ZeroClear();

      for (int a=0; a<N; ++a)
      M1_GLOOP1(i)
      {
        dotFv_(i) += sp_F_d(a,k,j,i) * sp_v_u(a,k,j,i);
      }

      M1_GLOOP1(i)
      {
        sc_W2_(i) = SQR(sc_W(k,j,i));
      }

      // assemble J
      M1_GLOOP1(i)
      {
        sc_J(k,j,i) = Assemble::sc_J__(sc_W2_(i),
                                       dotFv_(i),
                                       sc_E,
                                       sp_v_u,
                                       sp_P_dd,
                                       k, j, i);
      }

      // assemble (H_t, H_d)
      M1_GLOOP1(i)
      {
        const Real W = sc_W(k, j, i);
        sc_H_t(k,j,i) = Assemble::sc_H_t__(W, dotFv_(i), sc_E, sc_J, k, j, i);
        Assemble::sp_H_d__(sp_H_d, W, sc_J, sp_F_d, sp_v_d, sp_v_u, sp_P_dd,
                           k, j, i);
      }

      // assemble sc_n
      M1_GLOOP1(i)
      {
        const Real Gamma =  Assemble::sc_G__(
          fidu.sc_W(k,j,i), sc_E(k,j,i), sc_J(k,j,i), dotFv_(i),
          opt.fl_E, opt.fl_J, opt.eps_E
        );

        sc_n(k,j,i) = sc_nG(k,j,i) / Gamma;
      }
    }
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//