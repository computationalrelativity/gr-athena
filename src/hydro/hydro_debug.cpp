//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition

// C++ standard headers
#include <algorithm> // max
#include <cmath> // exp, pow, sqrt
#include <iomanip>
#include <iostream>
#include <fstream>

// Athena++ headers
#include "hydro.hpp"

#include "../athena_tensor.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/finite_differencing.hpp"
#include "../utils/linear_algebra.hpp"
#include "../z4c/z4c.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>

//----------------------------------------------------------------------------------------
// Direct evaluation of Hydro equation RHS [conservative form]
void Hydro::HydroRHS(AthenaArray<Real> & u_cons, AthenaArray<Real> & u_rhs)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;
  Z4c * pz4c = pmb->pz4c;

  // Grab fd from coordinates...
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  FiniteDifference::Uniform *fd = pco_gr->fd_cc;

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;
  const int nn1 = pmb->ie + 1;  // CC hydro

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 2> AT_N_mat;

  // scalar field derivatives
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_D1sca;

  // vector field derivatives
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 2> AT_N_D1vec;

  // symmetric tensor derivatives
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 3> AT_N_D1sym;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 4> AT_N_D2sym;

  // slice storage
  AT_N_sym sl_g_dd( pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym sl_K_dd( pz4c->storage.adm, Z4c::I_ADM_Kxx);
  AT_N_sca sl_psi4( pz4c->storage.adm, Z4c::I_ADM_psi4);

  AT_N_sca sl_alpha( pz4c->storage.u, Z4c::I_Z4c_alpha);
  AT_N_vec sl_beta_u(pz4c->storage.u, Z4c::I_Z4c_betax);

  AT_N_sca sl_A_rho(pz4c->storage.mat, Z4c::I_MAT_rho);
  AT_N_vec sl_S_d(  pz4c->storage.mat, Z4c::I_MAT_Sx);

  // Fluid primitives
  AT_N_sca sl_w_rho(   w, IDN);
  AT_N_sca sl_w_p(     w, IPR);
  AT_N_vec sl_w_util_u(w, IVX);

  // Fluid conservatives
  AT_N_sca sl_q_D(  u_cons, IDN);
  AT_N_vec sl_q_S_d(u_cons, IM1);
  AT_N_sca sl_q_tau(u_cons, IEN);

  AT_N_sca rhs_q_D(  u_rhs, IDN);
  AT_N_vec rhs_q_S_d(u_rhs, IM1);
  AT_N_sca rhs_q_tau(u_rhs, IEN);

  // various scratches
  AT_N_sca oo_detg_(  pmb->ncells1);
  AT_N_sca detg_(     pmb->ncells1);
  AT_N_sca sqrt_detg_(pmb->ncells1);

  AT_N_sym g_dd_(          pmb->ncells1);
  // AT_N_sym g_uu_(          pmb->ncells1);
  AT_N_vec w_util_u_(      pmb->ncells1);
  // AT_N_vec w_util_d_(      pmb->ncells1);

  AT_N_vec w_vtil_u_(      pmb->ncells1);
  AT_N_vec w_v_u_(         pmb->ncells1);

  AT_N_sca W_(             pmb->ncells1);
  AT_N_sca w_utilde_norm2_(pmb->ncells1);

  // Dense scratch for div.
  AT_N_vec F_D(  pmb->ncells3, pmb->ncells2, pmb->ncells1);
  AT_N_mat F_S_d(pmb->ncells3, pmb->ncells2, pmb->ncells1);
  AT_N_vec F_tau(pmb->ncells3, pmb->ncells2, pmb->ncells1);

  // Prepare F fields
  for (int k=0; k<pmb->ncells3; ++k)
  for (int j=0; j<pmb->ncells2; ++j)
  {

    // prepare slice (k,j fixed)
    for (int a=0; a<NDIM; ++a)
    {
      for (int b=a; b<NDIM; ++b)
      for (int i=0; i<pmb->ncells1; ++i)
      {
        g_dd_(a,b,i) = sl_g_dd(a,b,k,j,i);
      }

      for (int i=0; i<pmb->ncells1; ++i)
      {
        w_util_u_(a,i) = sl_w_util_u(a,k,j,i);
      }
    }

    // prepare det. terms
    for (int i=0; i<pmb->ncells1; ++i)
    {
      detg_(i) = Det3Metric(g_dd_, i);
      oo_detg_(i) = 1.0 / oo_detg_(i);
      sqrt_detg_(i) = std::sqrt(detg_(i));
    }
    // for (int i=0; i<pmb->ncells1; ++i)
    // {
    //   Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
    // }

    // // lower idx
    // LinearAlgebra::SlicedVecMet3Contraction(
    //   w_util_d_, w_util_u_, g_dd_,
    //   0, pmb->ncells1-1
    // );

    // prepare Lorentz
    for (int i=0; i<pmb->ncells1; ++i)
    {
      // Real const norm2_utilde = InnerProductSlicedVec3Metric(w_util_d_,
      //                                                        g_uu_,
      //                                                        i);

      Real const norm2_utilde = InnerProductSlicedVec3Metric(w_util_u_,
                                                             g_dd_,
                                                             i);

      W_(i) = std::sqrt(1.0 + norm2_utilde);
    }

    // various velocity representations
    for (int a=0; a<NDIM; ++a)
    {
      for (int i=0; i<pmb->ncells1; ++i)
      {
        w_v_u_(   a,i) = sl_w_util_u(a,k,j,i) / W_(i);
        w_vtil_u_(a,i) = w_v_u_(a,i) - sl_beta_u(a,k,j,i) / sl_alpha(k,j,i);
      }
    }

    // F terms
    for (int a=0; a<NDIM; ++a)
    for (int i=0; i<pmb->ncells1; ++i)
    {
      F_D(  a,k,j,i) = sl_q_D(k,j,i) * sl_alpha(k,j,i) * w_vtil_u_(a,i);

      F_tau(a,k,j,i) = (
        sl_q_tau(k,j,i) * w_vtil_u_(a,i) +
        sqrt_detg_(i) * sl_w_p(k,j,i) * w_v_u_(a,i)
      ) * sl_alpha(k,j,i);
    }

    for (int c=0; c<NDIM; ++c)
    for (int a=0; a<NDIM; ++a)
    for (int i=0; i<pmb->ncells1; ++i)
    {
      F_S_d(c,a,k,j,i) = (
        sl_q_S_d(c,k,j,i) * w_vtil_u_(a,i) +
        (c == a) * sl_w_p(k,j,i) * sqrt_detg_(i)
      ) * sl_alpha(k,j,i);
    }

  }

  // prepare derivatives and update RHS (N.B. subtraction)
  u_rhs.ZeroClear();
  for (int k=pmb->ks; k<=pmb->ke; ++k)
  for (int j=pmb->js; j<=pmb->je; ++j)
  {
    for (int a=0; a<NDIM; ++a)
    for (int i=pmb->is; i<=pmb->ie; ++i)
    {
      rhs_q_D(  k,j,i) -= fd->Dx(a, F_D(  a,k,j,i));
      rhs_q_tau(k,j,i) -= fd->Dx(a, F_tau(a,k,j,i));
    }

    for (int c=0; c<NDIM; ++c)
    for (int a=0; a<NDIM; ++a)
    for (int i=pmb->is; i<=pmb->ie; ++i)
    {
      rhs_q_S_d(c,k,j,i) -= fd->Dx(a, F_S_d(c,a,k,j,i));
    }
  }

  // geometric sources
  AthenaArray<Real> * _flux; // unused
  AthenaArray<Real> _bcc;

  pco_gr->AddCoordTermsDivergence(1, _flux, w, _bcc, u_rhs);

  // dissipation
  for(int n = 0; n < NHYDRO; ++n)
  for(int a = 0; a < NDIM; ++a) {
    ILOOP3(k,j,i) {
      u_rhs(n,k,j,i) += fd->Diss(a, u_cons(n,k,j,i), pz4c->opt.diss);
    }
  }

}


void Hydro::AddHydroRHS(AthenaArray<Real> & rhs,
                        Real const wght,
                        AthenaArray<Real> &u_out)
{
  MeshBlock * pmb = pmy_block;

  for(int n = 0; n < NHYDRO; ++n)
  for (int k=pmb->ks; k<=pmb->ke; ++k)
  for (int j=pmb->js; j<=pmb->je; ++j)
  for (int i=pmb->is; i<=pmb->ie; ++i)
  {
    u_out(n,k,j,i) += wght*(pmy_block->pmy_mesh->dt)*rhs(n,k,j,i);
  }

}


void Hydro::Hydro_IdealEoS_Prim2Cons(
  const Real Gamma,
  AthenaArray<Real> & prim,
  AthenaArray<Real> & cons,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // slice fields
  AT_N_sym sl_g_dd(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx);

  AT_N_sca sl_w_rho(   prim, IDN);
  AT_N_sca sl_w_p(     prim, IPR);
  AT_N_vec sl_w_util_u(prim, IVX);

  AT_N_sca sl_q_D(  cons, IDN);
  AT_N_sca sl_q_tau(cons, IEN);
  AT_N_vec sl_q_S_d(cons, IM1);


  // various scratches
  AT_N_sym g_dd_(     iu+1);
  AT_N_sym g_uu_(     iu+1);
  AT_N_sca detg_(     iu+1);
  AT_N_sca oo_detg_(  iu+1);
  AT_N_sca W_(        iu+1);
  AT_N_sca sqrt_detg_(iu+1);
  AT_N_vec w_util_u_( iu+1);
  AT_N_vec w_util_d_( iu+1);

  AT_N_sca w_hrho_(  iu+1);

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
      }
    }

    for (int i=il; i<=iu; ++i)
    {
      w_hrho_(i) = sl_w_rho(k,j,i) + Eos_Gamma_ratio * sl_w_p(k,j,i);
    }

    // prepare inverse metric
    for (int i=il; i<=iu; ++i)
    {
      detg_(i) = Det3Metric(g_dd_, i);
      oo_detg_(i) = 1.0 / detg_(i);
      // detg_(i) = (detg_(i) > 0) ? detg_(i) : 1;
      sqrt_detg_(i) = std::sqrt(detg_(i));

      Inv3Metric(oo_detg_, g_dd_, g_uu_, i);
    }

    // lower idx
    LinearAlgebra::SlicedVecMet3Contraction(
      w_util_d_, w_util_u_, g_dd_,
      il, iu
    );

    // Lorentz factor
    for (int i=il; i<=iu; ++i)
    {
      Real const norm2_utilde = InnerProductSlicedVec3Metric(
        w_util_u_, g_dd_, i
      );

      W_(i) = std::sqrt(1. + norm2_utilde);
    }


    for (int a=0; a<NDIM; ++a)
    for (int i=il; i<=iu; ++i)
    {
      sl_q_S_d(a,k,j,i) = sqrt_detg_(i) * w_hrho_(i) * W_(i) * w_util_d_(a,i);
    }

    for (int i=il; i<=iu; ++i)
    {
      sl_q_D(  k,j,i) = sqrt_detg_(i) * sl_w_rho(k,j,i) * W_(i);
      sl_q_tau(k,j,i) = sqrt_detg_(i) * (
        w_hrho_(i) * SQR(W_(i)) - sl_w_rho(k,j,i) * W_(i) - sl_w_p(k,j,i)
      );


      if ((sl_q_D(k,j,i) < 0.) || (sl_q_tau(k,j,i) < 0.))
      {
        sl_q_D(k,j,i) = 1e-20;
        sl_q_S_d(0,k,j,i) = 0;
        sl_q_S_d(1,k,j,i) = 0;
        sl_q_S_d(2,k,j,i) = 0;
        sl_q_tau(k,j,i) = 0;
        // std::cout << "D Flooring... " << std::endl;
        // std::exit(0);
      }

    }

  }


}

// root-finding fcn -----------------------------------------------------------
struct sys_Hydro_cons2prim_rparams
{
  Real Eos_Gamma_ratio;

  Real gdd_00;
  Real gdd_01;
  Real gdd_02;

  Real gdd_10;
  Real gdd_11;
  Real gdd_12;

  Real gdd_20;
  Real gdd_21;
  Real gdd_22;

  Real sqrt_detg;

  Real q_D;
  Real q_Sd_0;
  Real q_Sd_1;
  Real q_Sd_2;
  Real q_tau;
};

int sys_Hydro_cons2prim(
  const gsl_vector * x,
  void *params,
  gsl_vector * f)
{
  // consts. for eqs
  const Real Eos_Gamma_ratio = ((struct sys_Hydro_cons2prim_rparams *) params)->Eos_Gamma_ratio;

  const Real gdd_00 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_00;
  const Real gdd_01 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_01;
  const Real gdd_02 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_02;

  const Real gdd_10 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_10;
  const Real gdd_11 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_11;
  const Real gdd_12 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_12;

  const Real gdd_20 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_20;
  const Real gdd_21 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_21;
  const Real gdd_22 = ((struct sys_Hydro_cons2prim_rparams *) params)->gdd_22;

  const Real sqrt_detg = ((struct sys_Hydro_cons2prim_rparams *) params)->sqrt_detg;

  const Real q_D = ((struct sys_Hydro_cons2prim_rparams *) params)->q_D;
  const Real q_Sd_0 = ((struct sys_Hydro_cons2prim_rparams *) params)->q_Sd_0;
  const Real q_Sd_1 = ((struct sys_Hydro_cons2prim_rparams *) params)->q_Sd_1;
  const Real q_Sd_2 = ((struct sys_Hydro_cons2prim_rparams *) params)->q_Sd_2;
  const Real q_tau = ((struct sys_Hydro_cons2prim_rparams *) params)->q_tau;


  const Real F_W      = gsl_vector_get(x, 0);
  const Real F_rho    = gsl_vector_get(x, 1);
  const Real F_uu_0   = gsl_vector_get(x, 2);
  const Real F_uu_1   = gsl_vector_get(x, 3);
  const Real F_uu_2   = gsl_vector_get(x, 4);
  const Real F_p      = gsl_vector_get(x, 5);


  //
  const Real W2 = 1. + (
    F_uu_0 * F_uu_0 * gdd_00 + F_uu_0 * F_uu_1 * gdd_01 + F_uu_0 * F_uu_2 * gdd_02 +
    F_uu_1 * F_uu_0 * gdd_10 + F_uu_1 * F_uu_1 * gdd_11 + F_uu_1 * F_uu_2 * gdd_12 +
    F_uu_2 * F_uu_0 * gdd_20 + F_uu_2 * F_uu_1 * gdd_21 + F_uu_2 * F_uu_2 * gdd_22
  );

  const Real W      = std::sqrt(W2);
  const Real F_hrho = F_rho + Eos_Gamma_ratio * F_p;

  const Real F_ud_0 = F_uu_0 * gdd_00 + F_uu_1 * gdd_01 + F_uu_2 * gdd_02;
  const Real F_ud_1 = F_uu_0 * gdd_10 + F_uu_1 * gdd_11 + F_uu_2 * gdd_12;
  const Real F_ud_2 = F_uu_0 * gdd_20 + F_uu_1 * gdd_21 + F_uu_2 * gdd_22;


  // RHS (W, F_rho, util_u, F_p)
  const Real y0 = W - F_W;

  const Real y1 = sqrt_detg * F_rho * W - q_D;

  const Real y2 = sqrt_detg * F_hrho * W * F_ud_0 - q_Sd_0;
  const Real y3 = sqrt_detg * F_hrho * W * F_ud_1 - q_Sd_1;
  const Real y4 = sqrt_detg * F_hrho * W * F_ud_2 - q_Sd_2;

  const Real y5 = sqrt_detg * (F_hrho * SQR(W) - F_rho * W - F_p) - q_tau;

  gsl_vector_set(f, 0, y0);
  gsl_vector_set(f, 1, y1);
  gsl_vector_set(f, 2, y2);
  gsl_vector_set(f, 3, y3);
  gsl_vector_set(f, 4, y4);
  gsl_vector_set(f, 5, y5);

  return GSL_SUCCESS;
}
// ----------------------------------------------------------------------------

void Hydro::Hydro_IdealEoS_Cons2Prim(
  const Real Gamma,
  AthenaArray<Real> & cons,
  AthenaArray<Real> & prim,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // slice fields
  AT_N_sym sl_g_dd(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx);

  AT_N_sca sl_w_rho(   prim, IDN);
  AT_N_sca sl_w_p(     prim, IPR);
  AT_N_vec sl_w_util_u(prim, IVX);

  AT_N_sca sl_q_D(  cons, IDN);
  AT_N_sca sl_q_tau(cons, IEN);
  AT_N_vec sl_q_S_d(cons, IM1);

  // solver ------
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;

  const int max_iter = 10000;
  const Real tol = 1e-15;

  int status;
  size_t i, iter = 0;

  const size_t n = 6;
  // W, F_rho, util_u, F_p
  Real x_init[n] = {
    1.0,
    1e-2,
    1e-2, 1e-2, 1e-2,
    1e-2};  // initial value for search
  gsl_vector *x = gsl_vector_alloc(n);
  T = gsl_multiroot_fsolver_hybrids;
  s = gsl_multiroot_fsolver_alloc(T, n);
  // ----

  int sz = 0;
  int avg_iter = 0;

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    for (int i=il; i<=iu; ++i)
    {
      const Real sqrt_detg = std::sqrt(Det3Metric(sl_g_dd, k,j,i));

      struct sys_Hydro_cons2prim_rparams p = {
        Eos_Gamma_ratio,
        // gdd
        sl_g_dd(0,0,k,j,i),
        sl_g_dd(0,1,k,j,i),
        sl_g_dd(0,2,k,j,i),
        sl_g_dd(1,0,k,j,i),
        sl_g_dd(1,1,k,j,i),
        sl_g_dd(1,2,k,j,i),
        sl_g_dd(2,0,k,j,i),
        sl_g_dd(2,1,k,j,i),
        sl_g_dd(2,2,k,j,i),
        // modify if unweighted cons.
        sqrt_detg,
        // cons
        sl_q_D(k,j,i),
        sl_q_S_d(0,k,j,i),
        sl_q_S_d(1,k,j,i),
        sl_q_S_d(2,k,j,i),
        sl_q_tau(k,j,i)
      };

      gsl_multiroot_function f = {&sys_Hydro_cons2prim, n, &p};

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

      const Real W         = gsl_vector_get(s->x, 0);
      sl_w_rho(k,j,i)      = gsl_vector_get(s->x, 1);
      sl_w_util_u(0,k,j,i) = gsl_vector_get(s->x, 2);
      sl_w_util_u(1,k,j,i) = gsl_vector_get(s->x, 3);
      sl_w_util_u(2,k,j,i) = gsl_vector_get(s->x, 4);
      sl_w_p(k,j,i)        = gsl_vector_get(s->x, 5);

      avg_iter += iter;
      sz++;

      /*
      if ((W < 1-tol) || (sl_w_rho(k,j,i) < 0.) || (sl_w_p(k,j,i) < 0.))
      {
        std::cout << std::setprecision(16);
        std::cout << "W: " << W << std::endl;
        std::cout << "sl_w_rh: " << sl_w_rho(k,j,i)       << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(0,k,j,i)  << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(1,k,j,i)  << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(2,k,j,i)  << std::endl;
        std::cout << "sl_w_p(: " << sl_w_p(k,j,i)         << std::endl;
        std::cout << "iter: " << iter << std::endl;
        std::exit(0);
      }
      */

      if ((W < 1-tol) || (sl_w_rho(k,j,i) < 0.) || (sl_w_p(k,j,i) < 0.))
      {
        sl_w_rho(k,j,i) = 1e-16;
        sl_w_util_u(0,k,j,i) = 0;
        sl_w_util_u(1,k,j,i) = 0;
        sl_w_util_u(2,k,j,i) = 0;
        sl_w_p(k,j,i) = 1e-30;
        // std::cout << "sl_w_rh: " << sl_w_rho(k,j,i)       << std::endl;
        // std::cout << "sl_w_ut: " << sl_w_util_u(0,k,j,i)  << std::endl;
        // std::cout << "sl_w_ut: " << sl_w_util_u(1,k,j,i)  << std::endl;
        // std::cout << "sl_w_ut: " << sl_w_util_u(2,k,j,i)  << std::endl;
        // std::cout << "sl_w_p(: " << sl_w_p(k,j,i)         << std::endl;
        std::cout << "Flooring... " << iter << std::endl;
        // std::exit(0);
      }

    }

  }

  std::cout << "avg_iter: " << static_cast<Real>(avg_iter) / sz << std::endl;

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);
}



void Hydro::Hydro_IdealEoS_Cons2Prim(
  const Real Gamma,
  AthenaArray<Real> & cons,
  AthenaArray<Real> & prim,
  AthenaArray<Real> & prim_old,
  const int il, const int iu,
  const int jl, const int ju,
  const int kl, const int ku)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;

  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);

  // for readability
  const int D = NDIM + 1;
  const int N = NDIM;

  typedef AthenaArray< Real>                         AA;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
  typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
  typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

  // slice fields
  AT_N_sym sl_g_dd(pmb->pz4c->storage.adm, Z4c::I_ADM_gxx);

  AT_N_sca sl_w_rho(   prim, IDN);
  AT_N_sca sl_w_p(     prim, IPR);
  AT_N_vec sl_w_util_u(prim, IVX);

  AT_N_sca sl_q_D(  cons, IDN);
  AT_N_sca sl_q_tau(cons, IEN);
  AT_N_vec sl_q_S_d(cons, IM1);

  // solver ------
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;

  const int max_iter = 10000;
  const Real tol = 1e-15;

  int status;
  size_t i, iter = 0;

  const size_t n = 6;
  // W, F_rho, util_u, F_p
  Real x_init[n] = {
    1.0,
    0.0,
    0.0, 0.0, 0.0,
    0.0};  // initial value for search
  gsl_vector *x = gsl_vector_alloc(n);
  T = gsl_multiroot_fsolver_hybrids;
  s = gsl_multiroot_fsolver_alloc(T, n);
  // ----

  int sz = 0;
  int avg_iter = 0;

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    for (int i=il; i<=iu; ++i)
    {
      const Real sqrt_detg = std::sqrt(Det3Metric(sl_g_dd, k,j,i));

      x_init[1] = prim_old(IDN,k,j,i);
      x_init[2] = prim_old(IVX+0,k,j,i);
      x_init[3] = prim_old(IVX+1,k,j,i);
      x_init[4] = prim_old(IVX+2,k,j,i);
      x_init[5] = prim_old(IPR,k,j,i);

      struct sys_Hydro_cons2prim_rparams p = {
        Eos_Gamma_ratio,
        // gdd
        sl_g_dd(0,0,k,j,i),
        sl_g_dd(0,1,k,j,i),
        sl_g_dd(0,2,k,j,i),
        sl_g_dd(1,0,k,j,i),
        sl_g_dd(1,1,k,j,i),
        sl_g_dd(1,2,k,j,i),
        sl_g_dd(2,0,k,j,i),
        sl_g_dd(2,1,k,j,i),
        sl_g_dd(2,2,k,j,i),
        // modify if unweighted cons.
        sqrt_detg,
        // cons
        sl_q_D(k,j,i),
        sl_q_S_d(0,k,j,i),
        sl_q_S_d(1,k,j,i),
        sl_q_S_d(2,k,j,i),
        sl_q_tau(k,j,i)
      };

      gsl_multiroot_function f = {&sys_Hydro_cons2prim, n, &p};

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

      const Real W         = gsl_vector_get(s->x, 0);
      sl_w_rho(k,j,i)      = gsl_vector_get(s->x, 1);
      sl_w_util_u(0,k,j,i) = gsl_vector_get(s->x, 2);
      sl_w_util_u(1,k,j,i) = gsl_vector_get(s->x, 3);
      sl_w_util_u(2,k,j,i) = gsl_vector_get(s->x, 4);
      sl_w_p(k,j,i)        = gsl_vector_get(s->x, 5);

      avg_iter += iter;
      sz++;

      // if ((W < 1) || (sl_w_rho(k,j,i) < 0.))
      if ((W < 1-tol) || (sl_w_rho(k,j,i) < 0.) || (sl_w_p(k,j,i) < 0.))
      {
        std::cout << std::setprecision(16);
        std::cout << "W: " << W << std::endl;
        std::cout << "sl_w_rh: " << sl_w_rho(k,j,i)       << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(0,k,j,i)  << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(1,k,j,i)  << std::endl;
        std::cout << "sl_w_ut: " << sl_w_util_u(2,k,j,i)  << std::endl;
        std::cout << "sl_w_p(: " << sl_w_p(k,j,i)         << std::endl;
        std::cout << "iter: " << iter << std::endl;
        std::exit(0);
      }
    }

  }

  std::cout << "avg_iter: " << static_cast<Real>(avg_iter) / sz << std::endl;

  gsl_multiroot_fsolver_free(s);
  gsl_vector_free(x);
}