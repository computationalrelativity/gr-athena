//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_tov.cpp
//  \brief Problem generator for single TOV star in Cowling approximation

// C headers

// C++ headers
#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <fstream>    // ifstream
#include <iomanip>
#include <iostream>   // endl, ostream
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../z4c/z4c.hpp"
// #include "../z4c/z4c_amr.hpp"
#include "../trackers/extrema_tracker.hpp"
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput
#include "../utils/linear_algebra.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>

namespace {
  int RefinementCondition(MeshBlock *pmb);
  Real Maxrho(MeshBlock *pmb, int iout);
  Real linf_H(MeshBlock *pmb, int iout);
}

namespace matter_debug {

using namespace LinearAlgebra;

// params
Real ID_EoS_K;
Real ID_EoS_Gamma;

Real ID_pert_K_A_diag;
Real ID_pert_K_A;
Real ID_pert_K_phi;
Real ID_pert_K_f2;

Real ID_pert_g_A_diag;
Real ID_pert_g_A;
Real ID_pert_g_phi;
Real ID_pert_g_f2;

Real ID_pert_util_A;
Real ID_pert_util_phi;
Real ID_pert_util_f2;

Real ID_pert_util_0;
Real ID_pert_util_1;
Real ID_pert_util_2;

Real ID_pert_rho_min;
Real ID_pert_rho_A;
Real ID_pert_rho_phi;
Real ID_pert_rho_f2;

Real ID_pert_p_min;
Real ID_pert_p_A;
Real ID_pert_p_phi;
Real ID_pert_p_f2;


// for readability
const int D = NDIM + 1;
const int N = NDIM;

typedef AthenaArray< Real>                         AA;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;
// typedef AthenaTensor<Real, TensorSymm::SYM2, D, 2> AT_D_sym;
// typedef AthenaTensor<Real, TensorSymm::NONE, D, 1> AT_D_vec;

// symmetric tensor derivatives
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 3> AT_N_D1sym;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 4> AT_N_D2sym;

void SetNAN_Fields(MeshBlock * pmb)
{
  pmb->phydro->w_init.Fill(   NAN);
  pmb->phydro->w.Fill(        NAN);
  pmb->phydro->w1.Fill(       NAN);
  pmb->pz4c->storage.u.Fill(  NAN);
  pmb->pz4c->storage.u1.Fill( NAN);
  pmb->pz4c->storage.adm.Fill(NAN);
  pmb->pz4c->storage.mat.Fill(NAN);
}

void SetZero_Fields(MeshBlock * pmb)
{
  pmb->phydro->w_init.Fill(   0.);
  pmb->phydro->w.Fill(        0.);
  pmb->phydro->w1.Fill(       0.);
  pmb->pz4c->storage.u.Fill(  0.);
  pmb->pz4c->storage.u1.Fill( 0.);
  pmb->pz4c->storage.adm.Fill(0.);
  pmb->pz4c->storage.mat.Fill(0.);
}

void Trig_Functions(
  MeshBlock * pmb,
  const int il, const int iu,
  const int j,
  const int k,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3,
  AthenaArray<Real> & d0_T,
  const Real A,
  const Real f2,
  const Real phi)
{
  for (int a=0; a<NDIM; ++a)
  for (int b=0; b<NDIM; ++b)
  {
    const Real arg_1 = f2 * x2(j) * M_PI - (a+b) * M_PI / 2.0 + phi;
    const Real arg_2 = f2 * x3(k) * M_PI - (a+b) * M_PI / 2.0 + phi;

    const Real T1 = std::sin(arg_1);
    const Real T2 = std::sin(arg_2);

    for (int i=il; i<=iu; ++i)
    {
      const Real arg_0 = f2 * x1(i) * M_PI - (a+b) * M_PI / 2.0 + phi;

      const Real T0 = std::sin(arg_0);

      d0_T(a,b,i) = A * T0 * T1 * T2;
    }
  }

}

void Trig_Functions(
  MeshBlock * pmb,
  const int il, const int iu,
  const int j,
  const int k,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3,
  AthenaArray<Real> & d0_T,
  AthenaArray<Real> & d1_T,
  const Real A,
  const Real f2,
  const Real phi)
{
  for (int a=0; a<NDIM; ++a)
  for (int b=0; b<NDIM; ++b)
  {
    const Real arg_1 = f2 * x2(j) * M_PI - (a+b) * M_PI / 2.0 + phi;
    const Real arg_2 = f2 * x3(k) * M_PI - (a+b) * M_PI / 2.0 + phi;

    const Real d1_arg_1 = f2 * M_PI;
    const Real d1_arg_2 = f2 * M_PI;

    const Real T1 = std::sin(arg_1);
    const Real T2 = std::sin(arg_2);

    const Real d1_T1 = std::cos(arg_1) * d1_arg_1;
    const Real d1_T2 = std::cos(arg_2) * d1_arg_2;

    for (int i=il; i<=iu; ++i)
    {
      const Real arg_0 = f2 * x1(i) * M_PI - (a+b) * M_PI / 2.0 + phi;
      const Real d1_arg_0 = f2 * M_PI;

      const Real d2_arg_0 = 0.;

      const Real T0 = std::sin(arg_0);

      const Real d1_T0 = std::cos(arg_0) * d1_arg_0;

      d0_T(a,b,i) = A * T0 * T1 * T2;

      d1_T(0,a,b,i) = A * d1_T0 * T1 * T2;
      d1_T(1,a,b,i) = A * T0 * d1_T1 * T2;
      d1_T(2,a,b,i) = A * T0 * T1 * d1_T2;
    }
  }

}

void Trig_Functions(
  MeshBlock * pmb,
  const int il, const int iu,
  const int j,
  const int k,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3,
  AthenaArray<Real> & d0_T,
  AthenaArray<Real> & d1_T,
  AthenaArray<Real> & d2_T,
  const Real A,
  const Real f2,
  const Real phi)
{
  for (int a=0; a<NDIM; ++a)
  for (int b=0; b<NDIM; ++b)
  {
    const Real arg_1 = f2 * x2(j) * M_PI - (a+b) * M_PI / 2.0 + phi;
    const Real arg_2 = f2 * x3(k) * M_PI - (a+b) * M_PI / 2.0 + phi;

    const Real d1_arg_1 = f2 * M_PI;
    const Real d1_arg_2 = f2 * M_PI;

    const Real d2_arg_1 = 0.;
    const Real d2_arg_2 = 0.;

    const Real T1 = std::sin(arg_1);
    const Real T2 = std::sin(arg_2);

    const Real d1_T1 = std::cos(arg_1) * d1_arg_1;
    const Real d1_T2 = std::cos(arg_2) * d1_arg_2;

    const Real d2_T1 = (-std::sin(arg_1) * std::pow(d1_arg_1, 2.) +
                        std::cos(arg_1) * d2_arg_1);
    const Real d2_T2 = (-std::sin(arg_2) * std::pow(d1_arg_2, 2.) +
                        std::cos(arg_2) * d2_arg_2);

    for (int i=il; i<=iu; ++i)
    {
      const Real arg_0 = f2 * x1(i) * M_PI - (a+b) * M_PI / 2.0 + phi;
      const Real d1_arg_0 = f2 * M_PI;

      const Real d2_arg_0 = 0.;

      const Real T0 = std::sin(arg_0);

      const Real d1_T0 = std::cos(arg_0) * d1_arg_0;

      const Real d2_T0 = (-std::sin(arg_0) * std::pow(d1_arg_0, 2.) +
                          std::cos(arg_0) * d2_arg_0);

      d0_T(a,b,i) = A * T0 * T1 * T2;

      d1_T(0,a,b,i) = A * d1_T0 * T1 * T2;
      d1_T(1,a,b,i) = A * T0 * d1_T1 * T2;
      d1_T(2,a,b,i) = A * T0 * T1 * d1_T2;

      d2_T(0,0,a,b,i) = A * d2_T0 * T1 * T2;
      d2_T(0,1,a,b,i) = A * d1_T0 * d1_T1 * T2;
      d2_T(0,2,a,b,i) = A * d1_T0 * T1 * d1_T2;

      d2_T(1,0,a,b,i) = A * d1_T0 * d1_T1 * T2;
      d2_T(1,1,a,b,i) = A * T0 * d2_T1 * T2;
      d2_T(1,2,a,b,i) = A * T0 * d1_T1 * d1_T2;

      d2_T(2,0,a,b,i) = A * d1_T0 * T1 * d1_T2;
      d2_T(2,1,a,b,i) = A * T0 * d1_T1 * d1_T2;
      d2_T(2,2,a,b,i) = A * T0 * T1 * d2_T2;
    }
  }

}

// geometry -------------------------------------------------------------------

void Set_Geom_Gauge(MeshBlock * pmb)
{
  Z4c * pz4c = pmb->pz4c;
  pz4c->GaugeGeodesic(pz4c->storage.u);
  // pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);
}

void Set_Geom_g(MeshBlock * pmb,
                const int il, const int iu,
                const int j,
                const int k,
                const AthenaArray<Real> & x1,
                const AthenaArray<Real> & x2,
                const AthenaArray<Real> & x3,
                AT_N_sca   & psi4,
                AT_N_sym   & g_dd,
                AT_N_D1sym & d1g_ddd,
                AT_N_D2sym & d2g_dddd)
{
  const Real A = ID_pert_g_A;
  const Real f2 = ID_pert_g_f2;
  const Real phi = ID_pert_g_phi;

  // Array of trig functions
  AthenaArray<Real> d0_T(3, 3, iu+1);
  AthenaArray<Real> d1_T(3, 3, 3, iu+1);
  AthenaArray<Real> d2_T(3, 3, 3, 3, iu+1);

  Trig_Functions(pmb, il, iu, j, k, x1, x2, x3, d0_T, d1_T, d2_T, A, f2, phi);

  // populate metric coefficients
  for (int a=0; a<NDIM; ++a)
  for (int b=a; b<NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    g_dd(a,b,i) = ID_pert_g_A_diag * (a == b) + d0_T(a,b,i);
  }

  for (int c=0; c<NDIM; ++c)
  for (int a=0; a<NDIM; ++a)
  for (int b=a; b<NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    d1g_ddd(c,a,b,i) = d1_T(c,a,b,i);
  }

  for (int c=0; c<NDIM; ++c)
  for (int d=0; d<NDIM; ++d)
  for (int a=0; a<NDIM; ++a)
  for (int b=a; b<NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    d2g_dddd(c,d,a,b,i) = d2_T(c,d,a,b,i);
  }

  // Conformal factor
  for (int i=il; i<=iu; ++i)
  {
    psi4(i) = std::pow(Det3Metric(g_dd, i), -1./3.);
  }

}

void Set_Geom_K(MeshBlock * pmb,
                const int il, const int iu,
                const int j,
                const int k,
                const AthenaArray<Real> & x1,
                const AthenaArray<Real> & x2,
                const AthenaArray<Real> & x3,
                AT_N_sym   & K_dd,
                AT_N_D1sym & d1K_ddd)
{
  const Real A = ID_pert_K_A;
  const Real f2 = ID_pert_K_f2;
  const Real phi = ID_pert_K_phi;

  // Array of trig functions
  AthenaArray<Real> d0_T(3, 3, iu+1);
  AthenaArray<Real> d1_T(3, 3, 3, iu+1);

  Trig_Functions(pmb, il, iu, j, k, x1, x2, x3, d0_T, d1_T, A, f2, phi);

  for (int a=0; a<NDIM; ++a)
  for (int b=a; b<NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    K_dd(a,b,i) = ID_pert_K_A_diag * (a==b) + d0_T(a,b,i);
  }

  for (int c=0; c<NDIM; ++c)
  for (int a=0; a<NDIM; ++a)
  for (int b=a; b<NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    d1K_ddd(c,a,b,i) = d1_T(c,a,b,i);
  }

}

void Set_ADM_Constraints_Vacuum(MeshBlock * pmb,
                                const int il, const int iu,
                                const int j,
                                const int k,
                                AT_N_sym   & g_dd_,
                                AT_N_D1sym & d1g_ddd_,
                                AT_N_D2sym & d2g_dddd_,
                                AT_N_sym   & K_dd_,
                                AT_N_D1sym & d1K_ddd_)
{
  Z4c * pz4c = pmb->pz4c;
  MB_info* mbi = &(pz4c->mbi);

  // scratch for fields
  AT_N_sca   oo_detg_(  iu+1);
  AT_N_sym   g_uu_(     iu+1);
  AT_N_D1sym Gamma_ddd_(iu+1);
  AT_N_D1sym Gamma_udd_(iu+1);
  AT_N_sym   R_dd_(     iu+1);
  AT_N_sca   R_(        iu+1);
  AT_N_sym   K_ud_(     iu+1);
  AT_N_sca   K_(        iu+1);
  AT_N_sca   KK_(       iu+1);

  AT_N_D1sym D1K_udd_(  iu+1);
  AT_N_D1sym D1K_ddd_(  iu+1);

  AT_N_vec   M_u_(      iu+1);

  // slice storage
  AT_N_sca sl_H(  pz4c->storage.con, Z4c::I_CON_H);
  AT_N_vec sl_M_d(pz4c->storage.con, Z4c::I_CON_Mx);

  // prepare inverse metric
  for (int i=il; i<=iu; ++i)
  {
    oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
    Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
  }

  // Christoffel
  for (int c = 0; c < NDIM; ++c)
  for (int a = 0; a < NDIM; ++a)
  for (int b = a; b < NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    Gamma_ddd_(c,a,b,i) = 0.5*(d1g_ddd_(a,b,c,i) +
                               d1g_ddd_(b,a,c,i) - d1g_ddd_(c,a,b,i));
  }

  Gamma_udd_.ZeroClear();
  for (int c = 0; c < NDIM; ++c)
  for (int a = 0; a < NDIM; ++a)
  for (int b = a; b < NDIM; ++b)
  for (int d = 0; d < NDIM; ++d)
  for (int i=il; i<=iu; ++i)
  {
    Gamma_udd_(c,a,b,i) += g_uu_(c,d,i)*Gamma_ddd_(d,a,b,i);
  }

  // Ricci
  R_.ZeroClear();
  R_dd_.ZeroClear();

  for(int a = 0; a < NDIM; ++a)
  for(int b = a; b < NDIM; ++b)
  {
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d)
    {
      // Part with the Christoffel symbols
      for(int e = 0; e < NDIM; ++e)
      for (int i=il; i<=iu; ++i)
      {
        R_dd_(a,b,i) += g_uu_(c,d,i) *
                        Gamma_udd_(e,a,c,i) * Gamma_ddd_(e,b,d,i);
        R_dd_(a,b,i) -= g_uu_(c,d,i) *
                        Gamma_udd_(e,a,b,i) * Gamma_ddd_(e,c,d,i);
      }

      // Wave operator part of the Ricci
      for (int i=il; i<=iu; ++i)
      {
        R_dd_(a,b,i) += 0.5 * g_uu_(c,d,i) * (
                        - d2g_dddd_(c,d,a,b,i) -
                          d2g_dddd_(a,b,c,d,i) +
                          d2g_dddd_(a,c,b,d,i) +
                          d2g_dddd_(b,c,a,d,i)
        );
      }
    }
    for (int i=il; i<=iu; ++i)
    {
      R_(i) += g_uu_(a,b,i) * R_dd_(a,b,i);
    }
  }

  // Extrinsic curvature: traces and derivatives
  K_.ZeroClear();
  K_ud_.ZeroClear();
  KK_.ZeroClear();
  D1K_udd_.ZeroClear();

  for(int a = 0; a < NDIM; ++a)
  {
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for (int i=il; i<=iu; ++i)
    {
      K_ud_(a,b,i) += g_uu_(a,c,i) * K_dd_(c,b,i);
    }

    for (int i=il; i<=iu; ++i)
    {
      K_(i) += K_ud_(a,a,i);
    }
  }

  for (int a = 0; a < NDIM; ++a)
  for (int b = 0; b < NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    KK_(i) += K_ud_(a,b,i) * K_ud_(b,a,i);
  }

  // Covariant derivative of K
  for (int a = 0; a < NDIM; ++a)
  for (int b = 0; b < NDIM; ++b)
  for (int c = b; c < NDIM; ++c)
  {
    for (int i=il; i<=iu; ++i)
    {
      D1K_ddd_(a,b,c,i) = d1K_ddd_(a,b,c,i);
    }

    for(int d = 0; d < NDIM; ++d)
    for (int i=il; i<=iu; ++i)
    {
      D1K_ddd_(a,b,c,i) -= Gamma_udd_(d,a,b,i) * K_dd_(d,c,i);
      D1K_ddd_(a,b,c,i) -= Gamma_udd_(d,a,c,i) * K_dd_(b,d,i);
    }
  }

  for (int a = 0; a < NDIM; ++a)
  for (int b = 0; b < NDIM; ++b)
  for (int c = b; c < NDIM; ++c)
  for (int d = 0; d < NDIM; ++d)
  for (int i=il; i<=iu; ++i)
  {
    D1K_udd_(a,b,c,i) += g_uu_(a,d,i) * D1K_ddd_(d,b,c,i);
  }


  // Hamiltonian constraint (Vacuum)
  for (int i=il; i<=iu; ++i)
  {
    sl_H(k,j,i) = R_(i) + SQR(K_(i)) - KK_(i);  // - 16*M_PI * mat.rho(k,j,i);
  }

  // Momentum constraint (contravariant & vacuum)
  //
  M_u_.ZeroClear();
  for (int a = 0; a < NDIM; ++a)
  for (int b = 0; b < NDIM; ++b)
  {
    // for (int i=il; i<=iu; ++i)
    // {
    //   M_u_(a,i) -= 8*M_PI * g_uu(a,b,i) * mat.S_d(b,k,j,i);
    // }

    for (int c = 0; c < NDIM; ++c)
    for (int i=il; i<=iu; ++i)
    {
        M_u_(a,i) += g_uu_(a,b,i) * D1K_udd_(c,b,c,i);
        M_u_(a,i) -= g_uu_(b,c,i) * D1K_udd_(a,b,c,i);
    }
  }

  // Momentum constraint (covariant)
  for (int a = 0; a < NDIM; ++a)
  for (int i=il; i<=iu; ++i)
  {
    sl_M_d(a,k,j,i) = 0;
  }

  for (int a = 0; a < NDIM; ++a)
  for (int b = 0; b < NDIM; ++b)
  for (int i=il; i<=iu; ++i)
  {
    sl_M_d(a,k,j,i) += g_dd_(a,b,i) * M_u_(b,i);
  }

}

void Set_Geometry(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  Z4c * pz4c = pmb->pz4c;
  MB_info* mbi = &(pz4c->mbi);

  // scratch for fields
  AT_N_sca   psi4_(   iu+1);
  AT_N_sym   g_dd_(   iu+1);
  AT_N_sym   g_uu_(   iu+1);
  AT_N_sym   K_dd_(   iu+1);

  AT_N_D1sym d1K_ddd_( iu+1);
  AT_N_D1sym d1g_ddd_( iu+1);
  AT_N_D2sym d2g_dddd_(iu+1);

  // slice storage
  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    Set_Geom_g(pmb, il, iu, j, k, x1, x2, x3,
      psi4_, g_dd_, d1g_ddd_, d2g_dddd_);

    Set_Geom_K(pmb, il, iu, j, k, x1, x2, x3,
      K_dd_, d1K_ddd_);

    // populate geometry in z4c class
    for( int a=0; a<NDIM; ++a)
    for( int b=a; b<NDIM; ++b)
    {
      for (int i=il; i<=iu; ++i)
      {
        sl_g_dd(a,b,k,j,i) = g_dd_(a,b,i);
        sl_K_dd(a,b,k,j,i) = K_dd_(a,b,i);
      }
    }

    for (int i=il; i<=iu; ++i)
    {
      sl_psi4(k,j,i) = psi4_(i);
    }

    Set_ADM_Constraints_Vacuum(pmb, il, iu, j, k,
      g_dd_, d1g_ddd_, d2g_dddd_,
      K_dd_, d1K_ddd_);

  }

  // Gauge & Z4c construction
  Set_Geom_Gauge(pmb);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
}

// ----------------------------------------------------------------------------

void Set_Hydro_rho(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  // Pick some initial hydro density
  AT_N_sca sl_w_rho(pmb->phydro->w, IDN);

  const Real A0  = ID_pert_rho_min;
  const Real A   = ID_pert_rho_A;
  const Real f2  = ID_pert_rho_f2;
  const Real phi = ID_pert_rho_phi;

  // Array of trig functions
  AthenaArray<Real> d0_T(3, 3, iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {

    Trig_Functions(pmb, il, iu, j, k, x1, x2, x3,
                   d0_T, A, f2, phi);

    for (int i=il; i<=iu; ++i)
    {
      sl_w_rho(k,j,i) = A0 + d0_T(0,0,i);
    }
  }
}

void Set_Hydro_p(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  // Pick some initial hydro pressure
  AT_N_sca sl_w_p(pmb->phydro->w, IPR);

  const Real A0  = ID_pert_p_min;
  const Real A   = ID_pert_p_A;
  const Real f2  = ID_pert_p_f2;
  const Real phi = ID_pert_p_phi;

  // Array of trig functions
  AthenaArray<Real> d0_T(3, 3, iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {

    Trig_Functions(pmb, il, iu, j, k, x1, x2, x3,
                   d0_T, A, f2, phi);

    for (int i=il; i<=iu; ++i)
    {
      sl_w_p(k,j,i) = A0 + d0_T(0,0,i);
    }
  }
}

void Set_Hydro_velocity(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  // Pick some initial hydro velocity
  AT_N_vec sl_w_util_u(pmb->phydro->w, IVX);

  const Real A   = ID_pert_util_A;
  const Real f2  = ID_pert_util_f2;
  const Real phi = ID_pert_util_phi;

  // Array of trig functions
  AthenaArray<Real> d0_T(3, 3, iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    Trig_Functions(pmb, il, iu, j, k, x1, x2, x3,
                   d0_T, A, f2, phi);

    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(0,k,j,i) = ID_pert_util_0 + d0_T(0,0,i);
      sl_w_util_u(1,k,j,i) = ID_pert_util_1 + d0_T(0,1,i);
      sl_w_util_u(2,k,j,i) = ID_pert_util_2 + d0_T(0,2,i);
    }
  }

}

void Set_Hydro_velocity_With_ADM_Constraints(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  Z4c * pz4c = pmb->pz4c;
  Hydro * phydro = pmb->phydro;

  // slice fields
  AT_N_sca sl_w_rho(   phydro->w, IDN);
  AT_N_sca sl_w_p(     phydro->w, IPR);
  AT_N_vec sl_w_util_u(phydro->w, IVX);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);
  AT_N_vec S_d_(      iu+1);
  AT_N_vec S_u_(      iu+1);

  AT_N_sca W_(             iu+1);
  AT_N_sca w_utilde_norm2_(iu+1);

  // debug
  Real W_max = -std::numeric_limits<Real>::infinity();
  Real W_min = std::numeric_limits<Real>::infinity();

  Real W_diff = 0;

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {

    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }

      for (int i=il; i<=iu; ++i)
      {
        S_d_(a,i) = sl_S_d(a,k,j,i);
      }
    }

    // prepare inverse metric
    for (int i=il; i<=iu; ++i)
    {
      oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
    }

    // raise idx
    LinearAlgebra::SlicedVecMet3Contraction(
      S_u_, S_d_, g_uu_,
      il, iu
    );

    // Lorentz factor
    for (int i=il; i<=iu; ++i)
    {
      const Real Eos_TGam = ID_EoS_Gamma / (ID_EoS_Gamma - 1.0);

      W_(i) = std::sqrt(
        (sl_A_rho(k,j,i) + sl_w_p(k,j,i)) / (
          sl_w_rho(k,j,i) + sl_w_p(k,j,i) * Eos_TGam
        )
      );
    }

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      const Real Eos_TGam = ID_EoS_Gamma / (ID_EoS_Gamma - 1.0);
      const Real den = W_(i) * (
        sl_w_rho(k,j,i) + Eos_TGam * sl_w_p(k,j,i)
      );

      sl_w_util_u(a,k,j,i) = S_u_(a,i) / den;

      w_util_d_(a,i) = S_d_(a,i) / den;

    }

    // check consistency of W inference
    for (int i=il; i<=iu; ++i)
    {
      for (int a=0; a<NDIM; ++a)
      {
        w_util_u_(a,i) = sl_w_util_u(a,k,j,i);
      }

      // Real const norm2_utilde = InnerProductSlicedVec3Metric(w_util_u_,
      //                                                        sl_g_dd,
      //                                                        k, j, i);

      Real const norm2_utilde = InnerProductSlicedVec3Metric(w_util_d_,
                                                             g_uu_,
                                                             i);

      const Real W_rec = std::sqrt(1.0 + norm2_utilde);

      W_diff += std::abs(W_(i) - W_rec);
    }

    // debug
    for (int i=il; i<=iu; ++i)
    {
      if (W_(i) < W_min)
        W_min = W_(i);

      if (W_(i) > W_max)
        W_max = W_(i);
    }

  }

  std::cout << "(W_min, W_max, W_diff): ";
  std::cout << W_min << ", ";
  std::cout << W_max << ", ";
  std::cout << W_diff << std::endl;


}

void Set_Hydro_velocity_rho_With_ADM_Constraints_Hydro_p(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  Z4c * pz4c = pmb->pz4c;
  Hydro * phydro = pmb->phydro;

  // slice fields
  AT_N_sca sl_w_rho(   phydro->w, IDN);
  AT_N_sca sl_w_p(     phydro->w, IPR);
  AT_N_vec sl_w_util_u(phydro->w, IVX);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);
  AT_N_vec S_d_(      iu+1);
  AT_N_vec S_u_(      iu+1);
  AT_N_vec v_d_(      iu+1);
  AT_N_vec v_u_(      iu+1);

  AT_N_sca W_(             iu+1);
  AT_N_sca w_utilde_norm2_(iu+1);

  const Real Rat_ID_EoS_Gam = ID_EoS_Gamma / (ID_EoS_Gamma - 1.0);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }
    }

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      v_d_(a,i) = sl_S_d(a,k,j,i) / (sl_A_rho(k,j,i) + sl_w_p(k,j,i));
    }

    for (int i=il; i<=iu; ++i)
    {
      // prepare inverse metric
      oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);

      // data for Lorentz factor
      Real const norm2_v = InnerProductSlicedVec3Metric(v_d_, g_uu_, i);
      W_(i) = 1. / std::sqrt(1.-norm2_v);
    }

    // raise idx
    LinearAlgebra::SlicedVecMet3Contraction(
      v_u_, v_d_, g_uu_,
      il, iu
    );

    // now seed fluid velocity
    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(a,k,j,i) = v_u_(a,i) * W_(i);
    }

    // seed also density
    for (int i=il; i<=iu; ++i)
    {
      const Real W2 = SQR(W_(i));
      sl_w_rho(k,j,i) = 1. / W2 * (sl_A_rho(k,j,i) +
                        sl_w_p(k,j,i) * (1. - Rat_ID_EoS_Gam * W2));

    }

  }

  // sl_w_p.array().print_all("%.1e");
  // std::exit(0);

  /*
  AT_N_sca K_inferred(ku+1,ju+1,iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    K_inferred(k,j,i) = sl_w_p(k,j,i) / std::pow(sl_w_rho(k,j,i),
                                                 ID_EoS_Gamma);
  }
  K_inferred.array().print_all("%.2e");

  std::cout << "max" << std::endl;
  std::cout << K_inferred.array().num_max() << std::endl;
  std::exit(0);
  */
}

void Set_Hydro_velocity_p_With_ADM_Constraints_Hydro_rho(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  Z4c * pz4c = pmb->pz4c;
  Hydro * phydro = pmb->phydro;

  // slice fields
  AT_N_sca sl_w_rho(   phydro->w, IDN);
  AT_N_sca sl_w_p(     phydro->w, IPR);
  AT_N_vec sl_w_util_u(phydro->w, IVX);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);
  AT_N_vec S_d_(      iu+1);
  AT_N_vec S_u_(      iu+1);
  AT_N_vec v_d_(      iu+1);
  AT_N_vec v_u_(      iu+1);

  AT_N_sca W_(             iu+1);
  AT_N_sca w_utilde_norm2_(iu+1);

  const Real Rat_ID_EoS_Gam = ID_EoS_Gamma / (ID_EoS_Gamma - 1.0);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }
    }

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      const Real denom = (
        sl_A_rho(k,j,i) + ID_EoS_K * std::pow(sl_w_rho(k,j,i), ID_EoS_Gamma)
      );
      v_d_(a,i) = sl_S_d(a,k,j,i) / denom;
    }

    for (int i=il; i<=iu; ++i)
    {
      // prepare inverse metric
      oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);

      // data for Lorentz factor
      Real const norm2_v = InnerProductSlicedVec3Metric(v_d_, g_uu_, i);
      W_(i) = 1. / std::sqrt(1.-norm2_v);
    }

    // raise idx
    LinearAlgebra::SlicedVecMet3Contraction(
      v_u_, v_d_, g_uu_,
      il, iu
    );

    // now seed fluid velocity
    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(a,k,j,i) = v_u_(a,i) * W_(i);
    }

    // seed also pressure
    for (int i=il; i<=iu; ++i)
    {
      const Real W2 = SQR(W_(i));
      // sl_w_rho(k,j,i) = 1. / W2 * (sl_A_rho(k,j,i) +
      //                   sl_w_p(k,j,i) * (1. - Rat_ID_EoS_Gam * W2));

      const Real num = sl_A_rho(k,j,i) - sl_w_rho(k,j,i) * W2;
      const Real den = Rat_ID_EoS_Gam * W2 - 1.;
      sl_w_p(k,j,i) = num / den;

      // sl_w_p(k,j,i) = ID_EoS_K * std::pow(sl_w_rho(k,j,i), ID_EoS_Gamma);
    }

    if (k == 4 && j == 3)
    {
      const int i = 7;

      std::cout << std::setprecision(24) << std::endl;

      std::cout << ID_EoS_K << std::endl;

      std::cout << "ADM: rho, S" << std::endl;
      std::cout << sl_A_rho(k,j,i) << std::endl;
      std::cout << sl_S_d(0,k,j,i) << std::endl;
      std::cout << sl_S_d(1,k,j,i) << std::endl;
      std::cout << sl_S_d(2,k,j,i) << std::endl;

      std::cout << "w: rho, p" << std::endl;
      std::cout << sl_w_rho(k,j,i) << std::endl;
      std::cout << sl_w_p(k,j,i) << std::endl;

      std::cout << "g_uu_" << std::endl;
      std::cout << g_uu_(0,0,k,j,i) << std::endl;
      std::cout << g_uu_(0,1,k,j,i) << std::endl;
      std::cout << g_uu_(0,2,k,j,i) << std::endl;

      std::cout << g_uu_(1,0,k,j,i) << std::endl;
      std::cout << g_uu_(1,1,k,j,i) << std::endl;
      std::cout << g_uu_(1,2,k,j,i) << std::endl;

      std::cout << g_uu_(2,0,k,j,i) << std::endl;
      std::cout << g_uu_(2,1,k,j,i) << std::endl;
      std::cout << g_uu_(2,2,k,j,i) << std::endl;


      std::cout << "v_d_" << std::endl;
      std::cout << v_d_(0,i) << std::endl;
      std::cout << v_d_(1,i) << std::endl;
      std::cout << v_d_(2,i) << std::endl;

      std::exit(0);
    }

  }

  // sl_w_p.array().print_all("%.1e");
  // std::exit(0);

  /*
  AT_N_sca K_inferred(ku+1,ju+1,iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    K_inferred(k,j,i) = sl_w_p(k,j,i) / std::pow(sl_w_rho(k,j,i),
                                                 ID_EoS_Gamma);
  }
  K_inferred.array().print_all("%.2e");

  std::cout << "max" << std::endl;
  std::cout << K_inferred.array().num_max() << std::endl;
  std::exit(0);
  */
}


// root-finding fcn -----------------------------------------------------------
struct sys_Hydro_rparams
{
  Real Gamma;
  Real K_adi;

  Real guu_00;
  Real guu_01;
  Real guu_02;

  Real guu_10;
  Real guu_11;
  Real guu_12;

  Real guu_20;
  Real guu_21;
  Real guu_22;

  Real A_rho;
  Real A_Sd_0;
  Real A_Sd_1;
  Real A_Sd_2;
};


int sys_Hydro(
  const gsl_vector * x,
  void *params,
  gsl_vector * f)
{
  const Real Gamma = ((struct sys_Hydro_rparams *) params)->Gamma;
  const Real K_adi = ((struct sys_Hydro_rparams *) params)->K_adi;

  const Real Rat_ID_EoS_Gam = Gamma / (Gamma - 1.0);

  const Real guu_00 = ((struct sys_Hydro_rparams *) params)->guu_00;
  const Real guu_01 = ((struct sys_Hydro_rparams *) params)->guu_01;
  const Real guu_02 = ((struct sys_Hydro_rparams *) params)->guu_02;

  const Real guu_10 = ((struct sys_Hydro_rparams *) params)->guu_10;
  const Real guu_11 = ((struct sys_Hydro_rparams *) params)->guu_11;
  const Real guu_12 = ((struct sys_Hydro_rparams *) params)->guu_12;

  const Real guu_20 = ((struct sys_Hydro_rparams *) params)->guu_20;
  const Real guu_21 = ((struct sys_Hydro_rparams *) params)->guu_21;
  const Real guu_22 = ((struct sys_Hydro_rparams *) params)->guu_22;

  const Real A_rho  = ((struct sys_Hydro_rparams *) params)->A_rho;

  const Real A_Sd_0 = ((struct sys_Hydro_rparams *) params)->A_Sd_0;
  const Real A_Sd_1 = ((struct sys_Hydro_rparams *) params)->A_Sd_1;
  const Real A_Sd_2 = ((struct sys_Hydro_rparams *) params)->A_Sd_2;

  const Real F_rho = gsl_vector_get(   x, 0);
  const Real F_p   = gsl_vector_get(   x, 1);
  const Real F_vd_0   = gsl_vector_get(x, 2);
  const Real F_vd_1   = gsl_vector_get(x, 3);
  const Real F_vd_2   = gsl_vector_get(x, 4);

  const Real oo_W2 = 1. - (
    F_vd_0 * F_vd_0 * guu_00 + F_vd_0 * F_vd_1 * guu_01 + F_vd_0 * F_vd_2 * guu_02 +
    F_vd_1 * F_vd_0 * guu_10 + F_vd_1 * F_vd_1 * guu_11 + F_vd_1 * F_vd_2 * guu_12 +
    F_vd_2 * F_vd_0 * guu_20 + F_vd_2 * F_vd_1 * guu_21 + F_vd_2 * F_vd_2 * guu_22
  );

  const Real W2 = 1. / oo_W2;

  const Real cF = (F_rho + F_p * Rat_ID_EoS_Gam) * W2;

  // RHS (rho, p, vd0, vd1, vd2)
  const Real y0 = cF - F_p - A_rho;
  const Real y1 = K_adi * std::pow(F_rho, Gamma) - F_p;

  const Real y2 = cF * F_vd_0 - A_Sd_0;
  const Real y3 = cF * F_vd_1 - A_Sd_1;
  const Real y4 = cF * F_vd_2 - A_Sd_2;

  gsl_vector_set(f, 0, y0);
  gsl_vector_set(f, 1, y1);

  gsl_vector_set(f, 2, y2);
  gsl_vector_set(f, 3, y3);
  gsl_vector_set(f, 4, y4);

  return GSL_SUCCESS;
}
// ----------------------------------------------------------------------------


void Set_Hydro_p_rho_velocity_With_ADM_Constraints_K(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{

  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;

  const int max_iter = 1000;
  const Real tol = 1e-15;

  int status;
  size_t i, iter = 0;

  const size_t n = 5;
  Real x_init[n] = {0.1, 0.1, 0.1, 0.1, 0.1};  // seed non-zero init. vel.
  gsl_vector *x = gsl_vector_alloc(n);
  T = gsl_multiroot_fsolver_hybrids;
  s = gsl_multiroot_fsolver_alloc(T, n);


  Z4c * pz4c = pmb->pz4c;
  Hydro * phydro = pmb->phydro;

  // slice fields
  AT_N_sca sl_w_rho(   phydro->w, IDN);
  AT_N_sca sl_w_p(     phydro->w, IPR);
  AT_N_vec sl_w_util_u(phydro->w, IVX);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);
  AT_N_vec S_d_(      iu+1);
  AT_N_vec S_u_(      iu+1);
  AT_N_vec v_d_(      iu+1);
  AT_N_vec v_u_(      iu+1);

  AT_N_sca W_(             iu+1);
  AT_N_sca w_utilde_norm2_(iu+1);

  const Real Rat_ID_EoS_Gam = ID_EoS_Gamma / (ID_EoS_Gamma - 1.0);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }
    }

    for (int i=il; i<=iu; ++i)
    {
      // prepare inverse metric
      oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
    }

    for (int i=il; i<=iu; ++i)
    {
      struct sys_Hydro_rparams p = {
        ID_EoS_Gamma,
        ID_EoS_K,
        // guu
        g_uu_(0,0,i),
        g_uu_(0,1,i),
        g_uu_(0,2,i),
        g_uu_(1,0,i),
        g_uu_(1,1,i),
        g_uu_(1,2,i),
        g_uu_(2,0,i),
        g_uu_(2,1,i),
        g_uu_(2,2,i),
        // A_rho
        sl_A_rho(k,j,i),
        // A_Sd
        sl_S_d(0,k,j,i),
        sl_S_d(1,k,j,i),
        sl_S_d(2,k,j,i)
      };
      gsl_multiroot_function f = {&sys_Hydro, n, &p};

      for (int ix=0; ix<n; ++ix)
      {
        gsl_vector_set(x, ix, x_init[ix]);
      }

      gsl_multiroot_fsolver_set(s, &f, x);

      iter = 0; // reset count for next root finding proc.
      do
      {
        iter++;
        status = gsl_multiroot_fsolver_iterate(s);

        if (status)   /* check if solver is stuck */
          break;

        status = gsl_multiroot_test_residual (s->f, tol);
      }
      while (status == GSL_CONTINUE && iter < max_iter);

      sl_w_rho(k,j,i) = gsl_vector_get(s->x, 0);
      sl_w_p(k,j,i)   = gsl_vector_get(s->x, 1);
      v_d_(0,i) = gsl_vector_get(s->x, 2);
      v_d_(1,i) = gsl_vector_get(s->x, 3);
      v_d_(2,i) = gsl_vector_get(s->x, 4);

      // data for Lorentz factor
      Real const norm2_v = InnerProductSlicedVec3Metric(v_d_, g_uu_, i);
      W_(i) = 1. / std::sqrt(1.-norm2_v);
    }

    // raise idx
    LinearAlgebra::SlicedVecMet3Contraction(
      v_u_, v_d_, g_uu_,
      il, iu
    );

    // now seed fluid velocity
    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(a,k,j,i) = v_u_(a,i) * W_(i);
    }


    /*
    if (k == 4 && j == 3)
    {
      const int i = 7;

      std::cout << std::setprecision(24) << std::endl;

      std::cout << ID_EoS_K << std::endl;

      std::cout << "ADM: rho, S" << std::endl;
      std::cout << sl_A_rho(k,j,i) << std::endl;
      std::cout << sl_S_d(0,k,j,i) << std::endl;
      std::cout << sl_S_d(1,k,j,i) << std::endl;
      std::cout << sl_S_d(2,k,j,i) << std::endl;

      std::cout << "w: rho, p" << std::endl;
      std::cout << sl_w_rho(k,j,i) << std::endl;
      std::cout << sl_w_p(k,j,i) << std::endl;

      std::cout << "g_uu_" << std::endl;
      std::cout << g_uu_(0,0,k,j,i) << std::endl;
      std::cout << g_uu_(0,1,k,j,i) << std::endl;
      std::cout << g_uu_(0,2,k,j,i) << std::endl;

      std::cout << g_uu_(1,0,k,j,i) << std::endl;
      std::cout << g_uu_(1,1,k,j,i) << std::endl;
      std::cout << g_uu_(1,2,k,j,i) << std::endl;

      std::cout << g_uu_(2,0,k,j,i) << std::endl;
      std::cout << g_uu_(2,1,k,j,i) << std::endl;
      std::cout << g_uu_(2,2,k,j,i) << std::endl;


      std::cout << "v_d_" << std::endl;
      std::cout << v_d_(0,i) << std::endl;
      std::cout << v_d_(1,i) << std::endl;
      std::cout << v_d_(2,i) << std::endl;

      std::exit(0);
    }
    */

  }

  /*
  AT_N_sca K_inferred(ku+1,ju+1,iu+1);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    K_inferred(k,j,i) = sl_w_p(k,j,i) / std::pow(sl_w_rho(k,j,i),
                                                 ID_EoS_Gamma);
  }
  K_inferred.array().print_all("%.2e");

  std::cout << "max" << std::endl;
  std::cout << K_inferred.array().num_max() << std::endl;
  std::exit(0);
  */

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);


}


void Set_Hydro_With_ADM_Constraints(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku,
  const AthenaArray<Real> & x1,
  const AthenaArray<Real> & x2,
  const AthenaArray<Real> & x3)
{
  Z4c * pz4c = pmb->pz4c;
  Hydro * phydro = pmb->phydro;

  // slice fields
  AT_N_sca sl_w_rho(   phydro->w, IDN);
  AT_N_sca sl_w_p(     phydro->w, IPR);
  AT_N_vec sl_w_util_u(phydro->w, IVX);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  AT_N_sym sl_g_dd(  pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd(  pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4(  pz4c->storage.adm, Z4c::I_ADM_psi4);

  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);
  AT_N_vec S_d_(      iu+1);
  AT_N_vec S_u_(      iu+1);

  AT_N_sca W_(             iu+1);
  AT_N_sca w_utilde_norm2_(iu+1);


  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {

    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=il; i<=iu; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }

      for (int i=il; i<=iu; ++i)
      {
        w_util_u_(a,i) = sl_w_util_u(a,k,j,i);
        S_d_(     a,i) = sl_S_d(a,k,j,i);
      }
    }

    // prepare inverse metric
    for (int i=il; i<=iu; ++i)
    {
      oo_detg_(i) = 1. / Det3Metric(g_dd_, i);
      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
    }

    // Lorentz factor
    InnerProductSlicedVec3Metric(
      w_utilde_norm2_, w_util_u_, g_dd_,
      il, iu
    );

    for (int i=il; i<=iu; ++i)
    {
      W_(i) = std::sqrt(1. + w_utilde_norm2_(i));
    }

    // lower hydro vel idx
    LinearAlgebra::SlicedVecMet3Contraction(
      w_util_d_, w_util_u_, g_dd_,
      il, iu
    );

    // fix this component of util
    const int ix_a = 2;
    for (int i=il; i<=iu; ++i)
    {
      sl_w_p(k,j,i) = (-sl_A_rho(k,j,i) +
                        S_d_(ix_a,i) * W_(i) / w_util_d_(ix_a,i));

      const Real num = (
        S_d_(ix_a,i) * (1.0 - ID_EoS_Gamma) -
        sl_A_rho(k,j,i) * w_util_d_(ix_a,i) * W_(i) * ID_EoS_Gamma +
        S_d_(ix_a,i) * std::pow(W_(i), 2.0) * ID_EoS_Gamma
      );

      const Real den = (
        1.0 - ID_EoS_Gamma
      ) * w_util_d_(ix_a,i) * W_(i);

      sl_w_rho(k,j,i) = num / den;
    }


    // correction

    // raise hydro updated vel idx
    LinearAlgebra::SlicedVecMet3Contraction(
      S_u_, S_d_, g_uu_,
      il, iu
    );


    /*
    // raise hydro updated vel idx
    LinearAlgebra::SlicedVecMet3Contraction(
      S_u_, S_d_, g_uu_,
      il, iu
    );

    // correct other components using fixed w_rho, w_p ...
    for (int i=il; i<=iu; ++i)
    {
      const Real den = (
        W_(i) * (sl_w_rho(k,j,i) + sl_w_p(k,j,i) *
                                   ID_EoS_Gamma / (ID_EoS_Gamma - 1.))
      );
      for (int a=1; a<NDIM; ++a)
      {
        w_util_u_(a,i) = S_u_(a,i) / den;
      }
    }
    */

    for (int a=1; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(a,k,j,i) = w_util_u_(a,i);
    }

    /*
    // correct other components using fixed w_rho, w_p ...
    for (int i=il; i<=iu; ++i)
    {
      const Real den = (
        W_(i) * (sl_w_rho(k,j,i) + sl_w_p(k,j,i) *
                                   ID_EoS_Gamma / (ID_EoS_Gamma - 1.))
      );
      for (int a = 0; a<NDIM; ++a)
      {
        w_util_d_(a,i) = S_d_(a,i) / den;
      }
    }

    // raise hydro updated vel idx
    LinearAlgebra::SlicedVecMet3Contraction(
      w_util_u_, w_util_d_, g_uu_,
      il, iu
    );

    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_w_util_u(a,k,j,i) = w_util_u_(a,i);
    }
    */


  }


  Real w_p_min = INFINITY, w_p_max = -INFINITY;

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
      if (sl_w_p(k,j,i) < w_p_min)
        w_p_min = sl_w_p(k,j,i);

      if (sl_w_p(k,j,i) > w_p_max)
        w_p_max = sl_w_p(k,j,i);
  }


  std::cout << "(p_min, p_max): ";
  std::cout << w_p_min << ", ";
  std::cout << w_p_max << std::endl;
  std::exit(0);

}

void Adjust_ADM_Hydro_With_Constraints(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  Z4c * pz4c = pmb->pz4c;

  AT_N_sca H(  pz4c->storage.con, Z4c::I_CON_H);
  AT_N_vec M_d(pz4c->storage.con, Z4c::I_CON_Mx);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  // adjust ADM density by initial Hamiltonian error
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    sl_A_rho(k,j,i) = H(k,j,i) / (16.0 * M_PI);
  }

  for (int a=0; a<NDIM; ++a)
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    sl_S_d(a,k,j,i) = M_d(a,k,j,i) / (8.0 * M_PI);
  }

  Real A_rho_max = -std::numeric_limits<Real>::infinity();
  Real A_rho_min = std::numeric_limits<Real>::infinity();

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
      if (sl_A_rho(k,j,i) < A_rho_min)
        A_rho_min = sl_A_rho(k,j,i);

      if (sl_A_rho(k,j,i) > A_rho_max)
        A_rho_max = sl_A_rho(k,j,i);
  }


  std::cout << "(A_rho_min, A_rho_max): ";
  std::cout << A_rho_min << ", ";
  std::cout << A_rho_max << std::endl;

}

void Adjust_ADM_Constraints_With_Hydro(
  MeshBlock * pmb,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  Z4c * pz4c = pmb->pz4c;

  AT_N_sca H(  pz4c->storage.con, Z4c::I_CON_H);
  AT_N_vec M_d(pz4c->storage.con, Z4c::I_CON_Mx);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  // adjust ADM density by initial Hamiltonian error
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    H(k,j,i) += -16.0 * M_PI * sl_A_rho(k,j,i);
  }

  for (int a=0; a<NDIM; ++a)
  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    M_d(a,k,j,i) += -8.0 * M_PI * sl_S_d(a,k,j,i);
  }
}


void Constraint_Norm_VW(
  MeshBlock * pmb,
  AthenaArray<Real> & cons,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  Z4c * pz4c = pmb->pz4c;

  cons.Fill(0);

  AT_N_sca H(  pz4c->storage.con, Z4c::I_CON_H);
  AT_N_vec M_d(pz4c->storage.con, Z4c::I_CON_Mx);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    const Real vol = pz4c->mbi.dx1(i) * pz4c->mbi.dx2(j) * pz4c->mbi.dx3(k);
    cons(0) += vol * SQR(H(k,j,i));
    cons(1) += vol * SQR(M_d(0,k,j,i));
    cons(2) += vol * SQR(M_d(1,k,j,i));
    cons(3) += vol * SQR(M_d(2,k,j,i));
  }
}

void Constraint_Norm_AMAX(
  MeshBlock * pmb,
  AthenaArray<Real> & cons,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  Z4c * pz4c = pmb->pz4c;

  cons.Fill(0);

  AT_N_sca H(  pz4c->storage.con, Z4c::I_CON_H);
  AT_N_vec M_d(pz4c->storage.con, Z4c::I_CON_Mx);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    cons(0) = (cons(0)<std::abs(H(k,j,i))) ? std::abs(H(k,j,i)) : cons(0);
    cons(1) = (cons(1)<std::abs(M_d(0,k,j,i))) ?
      std::abs(M_d(0,k,j,i)) : cons(1);
    cons(2) = (cons(2)<std::abs(M_d(1,k,j,i))) ?
      std::abs(M_d(1,k,j,i)) : cons(2);
    cons(3) = (cons(3)<std::abs(M_d(2,k,j,i))) ?
      std::abs(M_d(2,k,j,i)) : cons(3);
  }
}

}


//----------------------------------------------------------------------------------------
//! \fn 
// \brief  Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // Read problem parameters
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, Maxrho,
                          "max-rho", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, linf_H,
                          "linf_H", UserHistoryOperation::max);
#if MAGNETIC_FIELDS_ENABLED
  EnrollUserHistoryOutput(1, DivB, "divB");
#endif

}


//----------------------------------------------------------------------------------------
//! \fn 
// \brief Setup User work

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // Allocate output arrays for fluxes
  AllocateUserOutputVariables(15);
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  AthenaArray<Real> &x1flux = phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = phydro->flux[X3DIR];

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1)
  {
    jl -= NGHOST;
    ju += NGHOST;
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1)
  {
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  for (int i=il; i<=iu; ++i)
  {
    user_out_var(0,k,j,i) = x1flux(0,k,j,i);
    user_out_var(1,k,j,i) = x1flux(1,k,j,i);
    user_out_var(2,k,j,i) = x1flux(2,k,j,i);
    user_out_var(3,k,j,i) = x1flux(3,k,j,i);
    user_out_var(4,k,j,i) = x1flux(4,k,j,i);
    user_out_var(5,k,j,i) = x2flux(0,k,j,i);
    user_out_var(6,k,j,i) = x2flux(1,k,j,i);
    user_out_var(7,k,j,i) = x2flux(2,k,j,i);
    user_out_var(8,k,j,i) = x2flux(3,k,j,i);
    user_out_var(9,k,j,i) = x2flux(4,k,j,i);
    user_out_var(10,k,j,i) = x3flux(0,k,j,i);
    user_out_var(11,k,j,i) = x3flux(1,k,j,i);
    user_out_var(12,k,j,i) = x3flux(2,k,j,i);
    user_out_var(13,k,j,i) = x3flux(3,k,j,i);
    user_out_var(14,k,j,i) = x3flux(4,k,j,i);
  }

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
}


//----------------------------------------------------------------------------------------
//! \fn

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  // Set parameters -----------------------------------------------------------
  matter_debug::ID_EoS_K     = pin->GetReal("hydro", "k_adi");
  matter_debug::ID_EoS_Gamma = pin->GetReal("hydro", "gamma");

  matter_debug::ID_pert_K_A_diag = pin->GetReal("problem", "ID_pert_K_A_diag");
  matter_debug::ID_pert_K_A = pin->GetReal("problem", "ID_pert_K_A");
  matter_debug::ID_pert_K_phi = pin->GetReal("problem", "ID_pert_K_phi");
  matter_debug::ID_pert_K_f2 = pin->GetReal("problem", "ID_pert_K_f2");

  matter_debug::ID_pert_g_A_diag = pin->GetReal("problem", "ID_pert_g_A_diag");
  matter_debug::ID_pert_g_A = pin->GetReal("problem", "ID_pert_g_A");
  matter_debug::ID_pert_g_phi = pin->GetReal("problem", "ID_pert_g_phi");
  matter_debug::ID_pert_g_f2 = pin->GetReal("problem", "ID_pert_g_f2");

  matter_debug::ID_pert_util_A = pin->GetReal("problem", "ID_pert_util_A");
  matter_debug::ID_pert_util_phi = pin->GetReal("problem", "ID_pert_util_phi");
  matter_debug::ID_pert_util_f2 = pin->GetReal("problem", "ID_pert_util_f2");

  matter_debug::ID_pert_util_0 = pin->GetReal("problem", "ID_pert_util_0");
  matter_debug::ID_pert_util_1 = pin->GetReal("problem", "ID_pert_util_1");
  matter_debug::ID_pert_util_2 = pin->GetReal("problem", "ID_pert_util_2");

  matter_debug::ID_pert_rho_min = pin->GetReal("problem", "ID_pert_rho_min");
  matter_debug::ID_pert_rho_A =   pin->GetReal("problem", "ID_pert_rho_A");
  matter_debug::ID_pert_rho_phi = pin->GetReal("problem", "ID_pert_rho_phi");
  matter_debug::ID_pert_rho_f2 =  pin->GetReal("problem", "ID_pert_rho_f2");

  matter_debug::ID_pert_p_min = pin->GetReal("problem", "ID_pert_p_min");
  matter_debug::ID_pert_p_A =   pin->GetReal("problem", "ID_pert_p_A");
  matter_debug::ID_pert_p_phi = pin->GetReal("problem", "ID_pert_p_phi");
  matter_debug::ID_pert_p_f2 =  pin->GetReal("problem", "ID_pert_p_f2");

  // pre-fill phydro & pz4c fields
  matter_debug::SetNAN_Fields(this);

  // extract ranges for distinct sampling
  MB_info* mbi = &(pz4c->mbi);

  // coords (ms): matter sampling; (gs): geometric sampling
  const AthenaArray<Real> x1_ms(pcoord->x1v);
  const AthenaArray<Real> x2_ms(pcoord->x2v);
  const AthenaArray<Real> x3_ms(pcoord->x3v);

  const AthenaArray<Real> x1_gs(mbi->x1);
  const AthenaArray<Real> x2_gs(mbi->x2);
  const AthenaArray<Real> x3_gs(mbi->x3);

  // indicial ranges
  const int il_gs = 0, iu_gs = mbi->nn1-1;
  const int jl_gs = 0, ju_gs = mbi->nn2-1;
  const int kl_gs = 0, ku_gs = mbi->nn3-1;

  const int il_ms = 0, iu_ms = ncells1-1;
  const int jl_ms = 0, ju_ms = ncells2-1;
  const int kl_ms = 0, ku_ms = ncells3-1;

  // --------------------------------------------------------------------------

  // Provide ADM: g, K, derivatives, and compute ADM constraints over MeshBlock
  //              This is done on _matter grid_
  matter_debug::Set_Geometry(this,
                             il_ms, iu_ms,
                             jl_ms, ju_ms,
                             kl_ms, ku_ms,
                             x1_ms, x2_ms, x3_ms);

  // Use constraint violation to seed ADM A_rho, A_S_i
  // This is done on matter grid (cc)
  matter_debug::Adjust_ADM_Hydro_With_Constraints(this,
                                                  il_ms, iu_ms,
                                                  jl_ms, ju_ms,
                                                  kl_ms, ku_ms);
  // Stored H, M_d is now updated by the violation and becomes 0
  matter_debug::Adjust_ADM_Constraints_With_Hydro(this,
                                                  il_ms, iu_ms,
                                                  jl_ms, ju_ms,
                                                  kl_ms, ku_ms);

  // compare with FD ----------------------------------------------------------

  // Calculate based on FD the constraints; this is a consistency check.
  // pz4c->ADMConstraints(pz4c->storage.con,
  //                      pz4c->storage.adm,
  //                      pz4c->storage.mat,
  //                      pz4c->storage.u);

  /*
  std::cout << "Init. fd (matter grid)" << std::endl;
  AthenaArray<Real> cons_fd_vw(4);
  AthenaArray<Real> cons_fd_amax(4);

  matter_debug::Constraint_Norm_VW(this, cons_fd_vw,
                                   il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  matter_debug::Constraint_Norm_AMAX(this, cons_fd_amax,
                                     il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  std::cout << std::setprecision(2);
  std::cout << "||.||_2: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_vw(0) << ", ";
  std::cout << cons_fd_vw(1) << ", ";
  std::cout << cons_fd_vw(2) << ", ";
  std::cout << cons_fd_vw(3) << std::endl;

  std::cout << "max|.|: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_amax(0) << ", ";
  std::cout << cons_fd_amax(1) << ", ";
  std::cout << cons_fd_amax(2) << ", ";
  std::cout << cons_fd_amax(3) << std::endl;

  std::exit(0);
  */
  // --------------------------------------------------------------------------

  // Set hydro quantities;
  // Velocity is taken as independent, and fixed, rho, P are inferred
  // matter_debug::Set_Hydro_velocity(this,
  //                                  il_ms, iu_ms,
  //                                  jl_ms, ju_ms,
  //                                  kl_ms, ku_ms,
  //                                  x1_ms, x2_ms, x3_ms);

  // Set hydro quantities.
  // F_rho, F_p are taken as prescribed, F_util must be inferred

  /*
  matter_debug::Set_Hydro_velocity_With_ADM_Constraints(
    this, il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms,
    x1_ms, x2_ms, x3_ms);
  */


  /*
  // use rho -> (p, vel)
  matter_debug::Set_Hydro_rho(this,
                              il_ms, iu_ms,
                              jl_ms, ju_ms,
                              kl_ms, ku_ms,
                              x1_ms, x2_ms, x3_ms);

  matter_debug::Set_Hydro_velocity_p_With_ADM_Constraints_Hydro_rho(
    this, il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms,
    x1_ms, x2_ms, x3_ms);
  */

  /*
  // use p -> (rho, vel)
  matter_debug::Set_Hydro_p(this,
                            il_ms, iu_ms,
                            jl_ms, ju_ms,
                            kl_ms, ku_ms,
                            x1_ms, x2_ms, x3_ms);

  matter_debug::Set_Hydro_velocity_rho_With_ADM_Constraints_Hydro_p(
    this, il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms,
    x1_ms, x2_ms, x3_ms);
  */

  // use K + root-finder -> (p, rho, vel)
  matter_debug::Set_Hydro_p_rho_velocity_With_ADM_Constraints_K(
    this, il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms,
    x1_ms, x2_ms, x3_ms);


  // constraints in norm
  std::cout << "Init. (matter grid)" << std::endl;
  AthenaArray<Real> cons_fd_vw(4);
  AthenaArray<Real> cons_fd_amax(4);

  matter_debug::Constraint_Norm_VW(this, cons_fd_vw,
                                   il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  matter_debug::Constraint_Norm_AMAX(this, cons_fd_amax,
                                     il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  std::cout << std::setprecision(2);
  std::cout << "||.||_2: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_vw(0) << ", ";
  std::cout << cons_fd_vw(1) << ", ";
  std::cout << cons_fd_vw(2) << ", ";
  std::cout << cons_fd_vw(3) << std::endl;

  std::cout << "max|.|: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_amax(0) << ", ";
  std::cout << cons_fd_amax(1) << ", ";
  std::cout << cons_fd_amax(2) << ", ";
  std::cout << cons_fd_amax(3) << std::endl;


  // populate data on geometric grid

  // Provide ADM: g, K, derivatives, and compute ADM constraints over MeshBlock
  //              This is done on _geometric grid_
  matter_debug::Set_Geometry(this,
                             il_gs, iu_gs,
                             jl_gs, ju_gs,
                             kl_gs, ku_gs,
                             x1_gs, x2_gs, x3_gs);

  matter_debug::Adjust_ADM_Hydro_With_Constraints(this,
                                                  il_gs, iu_gs,
                                                  jl_gs, ju_gs,
                                                  kl_gs, ku_gs);

  // Stored H, M_d is now updated by the violation and becomes 0
  matter_debug::Adjust_ADM_Constraints_With_Hydro(this,
                                                  il_gs, iu_gs,
                                                  jl_gs, ju_gs,
                                                  kl_gs, ku_gs);


  // sanity check for GetMatter; seed secondary mat register directly ---------
  /*
  AthenaArray<Real> tmp_mat(Z4c::N_MAT, mbi->nn3, mbi->nn2, mbi->nn1);

  pz4c->GetMatter(tmp_mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pfield->bcc);

  matter_debug::AT_N_sca sl_A_rho_tmp(tmp_mat, Z4c::I_MAT_rho);
  matter_debug::AT_N_sca sl_A_rho_ctr(pz4c->storage.mat, Z4c::I_MAT_rho);

  // matter_debug::AT_N_vec sl_S_d_tmp(tmp_mat, Z4c::I_MAT_Sx);
  // matter_debug::AT_N_vec sl_S_d_ctr(pz4c->storage.mat, Z4c::I_MAT_Sx);

  // sl_A_rho_tmp.array().print_all("%.1e");
  // sl_A_rho_ctr.array().print_all("%.1e");

  std::cout << sl_A_rho_tmp(8,8,8) - sl_A_rho_ctr(8,8,8)<< ", ";

  // std::cout << sl_S_d_tmp(0,8,8,8) << ", ";
  // std::cout << sl_S_d_ctr(0,8,8,8) << std::endl;

  std::exit(0);
  */
  // --------------------------------------------------------------------------


  // Initialise matter (seed storage.mat with phydro)
  pz4c->GetMatter(pz4c->storage.mat,
                  pz4c->storage.adm,
                  phydro->w,
                  pfield->bcc);

  // Recalculate constraints
  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  // // Initialise conserved variables
  // peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
  //                            0, ncells1-1,
  //                            0, ncells2-1,
  //                            0, ncells3-1);


  // sanity check [basic debug impl.]

  matter_debug::AT_N_sca sl_w_rho( phydro->w,  IDN);
  matter_debug::AT_N_sca sl_w1_rho(phydro->w1, IDN);


  matter_debug::AT_N_sca sl_w_p(   phydro->w,  IPR);
  matter_debug::AT_N_sca sl_w1_p(  phydro->w1, IPR);

  matter_debug::AT_N_sca sl_q_D( phydro->u,  IDN);
  matter_debug::AT_N_sca sl_q1_D(phydro->u1, IDN);

  const int ng_cmp = NGHOST;

  phydro->Hydro_IdealEoS_Prim2Cons(
    matter_debug::ID_EoS_Gamma,
    phydro->w,
    phydro->u,
    0, ncells1-1,
    0, ncells2-1,
    0, ncells3-1);

  phydro->Hydro_IdealEoS_Cons2Prim(
    matter_debug::ID_EoS_Gamma,
    phydro->u,
    phydro->w1,
    0, ncells1-1,
    0, ncells2-1,
    0, ncells3-1);

  /*
  std::cout << std::setprecision(16);
  std::cout << "prim2cons, cons2prim; debug" << std::endl;
  std::cout << "sl_w_rho" << std::endl;
  std::cout << sl_w_rho.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w_rho.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w_rho.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w1_rho" << std::endl;
  std::cout << sl_w1_rho.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w1_rho.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w1_rho.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w_p" << std::endl;
  std::cout << sl_w_p.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w_p.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w_p.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w1_p" << std::endl;
  std::cout << sl_w1_p.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w1_p.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w1_p.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_q_D" << std::endl;
  std::cout << sl_q_D.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_q_D.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_q_D.array().num_avg(ng_cmp) << std::endl;

  // out-source to reprimand
  peos->PrimitiveToConserved(phydro->w,
                             pfield->bcc,  // dummy here
                             phydro->u,
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1);

  const int coarse_flag = 0;
  peos->ConservedToPrimitive(phydro->u,    // cons.
                             phydro->w,    // prim_old (unused)
                             pfield->b,    // dummy here
                             phydro->w1,   // result goes here
                             pfield->bcc,  // dummy here
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1, coarse_flag);


  std::cout << "prim2cons, cons2prim; reprimand" << std::endl;
  std::cout << "sl_w_rho" << std::endl;
  std::cout << sl_w_rho.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w_rho.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w_rho.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w1_rho" << std::endl;
  std::cout << sl_w1_rho.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w1_rho.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w1_rho.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w_p" << std::endl;
  std::cout << sl_w_p.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w_p.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w_p.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_w1_p" << std::endl;
  std::cout << sl_w1_p.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_w1_p.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_w1_p.array().num_avg(ng_cmp) << std::endl;

  std::cout << "sl_q_D" << std::endl;
  std::cout << sl_q_D.array().num_min(ng_cmp) << std::endl;
  std::cout << sl_q_D.array().num_max(ng_cmp) << std::endl;
  std::cout << sl_q_D.array().num_avg(ng_cmp) << std::endl;
  */


  // Get matter from reprimand loop & recompute constraints

  // // Initialise matter (seed storage.mat with phydro)
  // pz4c->GetMatter(pz4c->storage.mat,
  //                 pz4c->storage.adm,
  //                 phydro->w1,           // use reprimand data
  //                 pfield->bcc);

  // // Recalculate constraints
  // pz4c->ADMConstraints(pz4c->storage.con,
  //                      pz4c->storage.adm,
  //                      pz4c->storage.mat,
  //                      pz4c->storage.u);

  // std::exit(0);


  // std::cout << phydro->w1(8,8,8) - phydro->w(8,8,8)<< ", ";
  // std::cout << phydro->w1(20,20,20) - phydro->w(20,20,20)<< ", ";
  // std::exit(0);

  /*
  const int coarse_flag = 0;
  peos->ConservedToPrimitive(phydro->u,    // cons.
                             phydro->w,    // prim_old (unused)
                             pfield->b,    // dummy here
                             phydro->w1,   // result goes here
                             pfield->bcc,  // dummy here
                             pcoord,
                             0, ncells1-1,
                             0, ncells2-1,
                             0, ncells3-1, coarse_flag);

  std::cout << phydro->w1(8,8,8) - phydro->w(8,8,8)<< ", ";
  std::cout << phydro->w1(20,20,20) - phydro->w(20,20,20)<< ", ";
  std::exit(0);
  */


  /*
  phydro->Hydro_IdealEoS_Prim2Cons(
    matter_debug::ID_EoS_Gamma,
    phydro->w, phydro->u,
    0, ncells1-1,
    0, ncells2-1,
    0, ncells3-1);
  */
  // std::cout << phydro->w1(5,8,8) - phydro->w(5,8,8)<< ", ";
  // phydro->u.print_all();
  // std::exit(0);

  std::cout << "post Init." << std::endl;

  matter_debug::Constraint_Norm_VW(this, cons_fd_vw,
                                   il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  matter_debug::Constraint_Norm_AMAX(this, cons_fd_amax,
                                     il_ms, iu_ms, jl_ms, ju_ms, kl_ms, ku_ms);

  std::cout << std::setprecision(2);
  std::cout << "||.||_2: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_vw(0) << ", ";
  std::cout << cons_fd_vw(1) << ", ";
  std::cout << cons_fd_vw(2) << ", ";
  std::cout << cons_fd_vw(3) << std::endl;

  std::cout << "max|.|: (H, M_x, M_y, M_z) = ";
  std::cout << cons_fd_amax(0) << ", ";
  std::cout << cons_fd_amax(1) << ", ";
  std::cout << cons_fd_amax(2) << ", ";
  std::cout << cons_fd_amax(3) << std::endl;

  std::cout << "pgen done!" << std::endl;

  // std::exit(0);

  return;
}


namespace {

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: simple time-dependent test

int RefinementCondition(MeshBlock *pmb){

  const Real ref_dx = 0.1;
  const Real t = pmb->pmy_mesh->time + pmb->pmy_mesh->dt;

  // consider wrapped t as the refinement region centre
  // const Real ref_x1_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  // const Real ref_x1_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;

  const Real ref_x1_0 = 2 * (std::fmod(t + 0.5, 1) - 0.5);
  const Real ref_x1_l = ref_x1_0 - ref_dx;
  const Real ref_x1_r = ref_x1_0 + ref_dx;


  // check whether range formed by extent of this MeshBlock overlaps with
  // ref_x1_0 + [-ref_dx, ref_dx]
  const Real Mb_x1_l = pmb->pcoord->x1f(0);
  const Real Mb_x1_r = pmb->pcoord->x1f(pmb->ncells1-1);

  bool ol_x1 = std::max(ref_x1_l, Mb_x1_l) <= std::min(ref_x1_r, Mb_x1_r);

  // const Real ref_x2_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  const Real ref_x2_0 = 0;
  const Real ref_x2_l = ref_x2_0 - ref_dx;
  const Real ref_x2_r = ref_x2_0 + ref_dx;

  const Real Mb_x2_l = pmb->pcoord->x2f(0);
  const Real Mb_x2_r = pmb->pcoord->x2f(pmb->ncells2-1);

  bool ol_x2 = std::max(ref_x2_l, Mb_x2_l) <= std::min(ref_x2_r, Mb_x2_r);


  // const Real ref_x3_0 = std::asin(std::sin(PI / 2. * t / SQRT2)) * 2. / PI;
  const Real ref_x3_0 = 0;
  const Real ref_x3_l = ref_x3_0 - ref_dx;
  const Real ref_x3_r = ref_x3_0 + ref_dx;

  const Real Mb_x3_l = pmb->pcoord->x3f(0);
  const Real Mb_x3_r = pmb->pcoord->x3f(pmb->ncells2-1);

  bool ol_x3 = std::max(ref_x3_l, Mb_x3_l) <= std::min(ref_x3_r, Mb_x3_r);

  if (ol_x1 && ol_x2 && ol_x3)
  {
    // std::cout << "ref: " << pmb->gid << std::endl;
    return 1;
  }
  else
  {
    // std::cout << "deref: " << pmb->gid << std::endl;
    return -1;
  }

}

Real Maxrho(MeshBlock *pmb, int iout)
{
  Real max_rho = 0.0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    max_rho = std::max(std::abs(w(IDN,k,j,i)), max_rho);
  }

  return max_rho;
}

Real linf_H(MeshBlock *pmb, int iout)
{
  Z4c * pz4c = pmb->pz4c;

  Real linf_H = 0.0;
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  matter_debug::AT_N_sca H(pz4c->storage.con, Z4c::I_CON_H);

  for (int k=ks; k<=ke; k++)
  for (int j=js; j<=je; j++)
  for (int i=is; i<=ie; i++)
  {
    linf_H = std::max(std::abs(H(k,j,i)), linf_H);
  }

  return linf_H;
}


} // namespace
