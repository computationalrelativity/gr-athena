//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file hd_hllc.cpp
//  \brief Implements HLLC Riemann solver for relativistic hydrodynamics
//  using a tetrad (locally Minkowski) frame transformation.
//
// Refs:
// Kiuchi '22 @ http://arxiv.org/abs/2205.04487
// Mignone '05 @ http://arxiv.org/abs/astro-ph/0506414

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()
#include <cstdio>     // fprintf

// Athena++ headers
#include "../../athena_aliases.hpp"
#include "../../coordinates/coordinates.hpp"  // Coordinates
#include "../../eos/eos.hpp"                  // EquationOfState
#include "../../mesh/mesh.hpp"                // MeshBlock
#include "../../utils/floating_point.hpp"
#include "../../utils/interp_intergrid.hpp"
#include "../../utils/linear_algebra.hpp"
#include "../../z4c/ahf.hpp"
#include "../../z4c/z4c.hpp"
#include "../hydro.hpp"
#include "eigenvalues.hpp"  // Eigenvalues::HydroEigenvalues

using namespace gra::aliases;

//----------------------------------------------------------------------------------------
//! \fn sr_eigen_plus, sr_eigen_minus
//  \brief Pure SR hydro eigenvalues in the tetrad frame

inline Real sr_eigen_plus(Real vd, Real v2, Real cs2)
{
  const Real disc =
    cs2 * (1.0 - v2) * (1.0 - v2 * cs2 - (1.0 - cs2) * SQR(vd));
  const Real sqrt_disc = std::sqrt(std::max(disc, 0.0));
  const Real oo_den = 1.0 / std::max(1.0 - v2 * cs2, 1e-15);
  return (vd * (1.0 - cs2) + sqrt_disc) * oo_den;
}

inline Real sr_eigen_minus(Real vd, Real v2, Real cs2)
{
  const Real disc =
    cs2 * (1.0 - v2) * (1.0 - v2 * cs2 - (1.0 - cs2) * SQR(vd));
  const Real sqrt_disc = std::sqrt(std::max(disc, 0.0));
  const Real oo_den = 1.0 / std::max(1.0 - v2 * cs2, 1e-15);
  return (vd * (1.0 - cs2) - sqrt_disc) * oo_den;
}

//----------------------------------------------------------------------------------------
//! \fn FluxBackTransform
//  \brief Transform tetrad HLLC flux back to densitized Eulerian flux
//  using the lower-triangular covariant tetrad block.

inline void FluxBackTransform(const int d,
                              const int a,
                              const int b,
                              Real sqrtg,
                              Real alpha,
                              Real et_x,
                              Real ex_x,
                              Real e_cov_dd,
                              Real e_cov_ad,
                              Real e_cov_aa,
                              Real e_cov_bd,
                              Real e_cov_ba,
                              Real e_cov_bb,
                              const Real* q,
                              const Real* f,
                              Real& F_D,
                              Real& F_Sd,
                              Real& F_Sa,
                              Real& F_Sb,
                              Real& F_E)
{
  F_D = sqrtg * alpha * (et_x * q[0] + ex_x * f[0]);

  // Momentum d:
  Real sum_J_d =
    e_cov_dd * q[1] + e_cov_ad * q[2] + e_cov_bd * q[3];
  Real sum_fJ_d =
    e_cov_dd * f[1] + e_cov_ad * f[2] + e_cov_bd * f[3];
  F_Sd = sqrtg * alpha * (et_x * sum_J_d + ex_x * sum_fJ_d);

  // Momentum a:
  Real sum_J_a = e_cov_aa * q[2] + e_cov_ba * q[3];
  Real sum_fJ_a = e_cov_aa * f[2] + e_cov_ba * f[3];
  F_Sa = sqrtg * alpha * (et_x * sum_J_a + ex_x * sum_fJ_a);

  // Momentum b:
  Real sum_J_b = e_cov_bb * q[3];
  Real sum_fJ_b = e_cov_bb * f[3];
  F_Sb = sqrtg * alpha * (et_x * sum_J_b + ex_x * sum_fJ_b);

  // Energy: F(sqrt(gamma) * rho_H), NOT F(tau). Caller must subtract F_D to get F(tau).
  F_E = sqrtg * alpha * (et_x * q[4] + ex_x * f[4]);
}

//----------------------------------------------------------------------------------------
// RiemannSolverHLLC
//
// Inputs / Outputs: same signature as Hydro::RiemannSolver.
// Called from Hydro::RiemannSolver when solver_method_ == SolverMethod::hllc.

void Hydro::RiemannSolverHLLC(
  const int ivx,
  const int k,
  const int j,
  const int il,
  const int iu,
  AA& prim_l_,
  AA& prim_r_,
  AA& pscalars_l_,
  AA& pscalars_r_,
  AA& aux_l_,
  AA& aux_r_,
  AT_N_sca& alpha_,
  AT_N_vec& beta_u_,
  AT_N_sym& gamma_dd_,
  AT_N_sca& sqrt_detgamma_,
  AA& flux,
  AA& s_flux,
  const AA& dxw_,
  const Real lambda_rescaling)
{
  using namespace LinearAlgebra;
  using namespace FloatingPoint;

  MeshBlock* pmb         = pmy_block;
  Hydro* ph              = pmb->phydro;
  EquationOfState* peos  = pmb->peos;
  Reconstruction* precon = pmb->precon;

  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  const int d   = ivx - 1;
  const int a   = (d + 1) % 3;
  const int b   = (d + 2) % 3;

  const Real mb = pmb->peos->GetEOS().GetBaryonMass();

  // --- 1d slices ----------------------------------------------------------
  AT_N_sca w_rho_l_(prim_l_, IDN);
  AT_N_sca w_rho_r_(prim_r_, IDN);
  AT_N_sca w_p_l_(prim_l_, IPR);
  AT_N_sca w_p_r_(prim_r_, IPR);

  AT_N_vec w_util_u_l_(prim_l_, IVX);
  AT_N_vec w_util_u_r_(prim_r_, IVX);

  // --- excision (same as hd_llf.cpp) -------------------------------------
  Real T_min = peos->GetEOS().GetTemperatureFloor();
  Real h_min = peos->GetEOS().GetMinimumEnthalpy();


  auto excise = [&](const int i)
  {
    PrimHelper::SetPrimAtmo(peos->GetEOS(), prim_l_, pscalars_l_, i);
    PrimHelper::SetPrimAtmo(peos->GetEOS(), prim_r_, pscalars_r_, i);
    aux_l_(IX_T, i)  = T_min;
    aux_r_(IX_T, i)  = T_min;
    aux_l_(IX_ETH, i) = h_min;
    aux_r_(IX_ETH, i) = h_min;
    aux_l_(IX_LOR, i) = 1.0;
    aux_r_(IX_LOR, i) = 1.0;
  };

  auto excise_with_factor = [&](Real excision_factor, const int i)
  {
    for (int n = 0; n < NHYDRO; ++n)
    {
      prim_l_(n, i) *= ph->excision_mask(k, j, i);
      prim_r_(n, i) *= ph->excision_mask(k, j, i);
    }
    aux_l_(IX_T, i) *= excision_factor;
    aux_r_(IX_T, i) *= excision_factor;
    aux_l_(IX_ETH, i) *= excision_factor;
    aux_r_(IX_ETH, i) *= excision_factor;
    aux_l_(IX_LOR, i) *= excision_factor;
    aux_r_(IX_LOR, i) *= excision_factor;
  };

  AA *x1, *x2, *x3;
  switch (ivx)
  {
    case IVX: x1 = &pco_gr->x1f; x2 = &pco_gr->x2v; x3 = &pco_gr->x3v; break;
    case IVY: x1 = &pco_gr->x1v; x2 = &pco_gr->x2f; x3 = &pco_gr->x3v; break;
    case IVZ: x1 = &pco_gr->x1v; x2 = &pco_gr->x2v; x3 = &pco_gr->x3f; break;
  }

  if (ph->opt_excision.excise_flux)
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      Real excision_factor  = 1;
      const bool can_excise = peos->CanExcisePoint(
        excision_factor, true, alpha_, *x1, *x2, *x3, i, j, k);
      if (can_excise)
      {
        if (ph->opt_excision.use_taper)
          excise_with_factor(excision_factor, i);
        else
          excise(i);
      }
    }
  }

  // --- lower velocity indices (same as hd_llf.cpp) -----------------------
  SlicedVecMet3Contraction(w_util_d_l_, w_util_u_l_, gamma_dd_, il, iu);
  SlicedVecMet3Contraction(w_util_d_r_, w_util_u_r_, gamma_dd_, il, iu);

  // --- Lorentz factors ---------------------------------------------------
  if (precon->xorder_use_aux_W)
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      W_l_(i) = aux_l_(IX_LOR, i);
      W_r_(i) = aux_r_(IX_LOR, i);
    }
  }
  else
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      const Real n2_l = InnerProductSlicedVec3Metric(w_util_u_l_, gamma_dd_, i);
      const Real n2_r = InnerProductSlicedVec3Metric(w_util_u_r_, gamma_dd_, i);
      W_l_(i) = std::sqrt(1. + std::abs(n2_l));
      W_r_(i) = std::sqrt(1. + std::abs(n2_r));
    }
  }

  // --- Eulerian velocities -----------------------------------------------
  for (int ax = 0; ax < NDIM; ++ax)
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      w_v_u_l_(ax, i) = w_util_u_l_(ax, i) / W_l_(i);
      w_v_u_r_(ax, i) = w_util_u_r_(ax, i) / W_r_(i);
    }
  }

  InnerProductSlicedVec3Metric(w_norm2_v_l_, w_v_u_l_, gamma_dd_, il, iu);
  InnerProductSlicedVec3Metric(w_norm2_v_r_, w_v_u_r_, gamma_dd_, il, iu);

  // --- sound speed & rho*h (same as hd_llf.cpp) ---------------------------
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    Real hl = aux_l_(IX_ETH, i);
    Real hr = aux_r_(IX_ETH, i);
    w_hrho_l_(i) = w_rho_l_(i) * hl;
    w_hrho_r_(i) = w_rho_r_(i) * hr;

    Real cs2l, cs2r;
    if (precon->xorder_use_aux_cs2 ||
        precon->xorder_use_aux_eos_conditioned)
    {
      cs2l = aux_l_(IX_CS2, i);
      cs2r = aux_r_(IX_CS2, i);
    }
    else
    {
      Real nl = w_rho_l_(i) / mb;
      Real nr = w_rho_r_(i) / mb;
      Real Tl = aux_l_(IX_T, i);
      Real Tr = aux_r_(IX_T, i);
      Real Yl[MAX_SPECIES] = { 0.0 };
      Real Yr[MAX_SPECIES] = { 0.0 };
      for (int n = 0; n < NSCALARS; n++)
      {
        Yl[n] = pscalars_l_(n, i);
        Yr[n] = pscalars_r_(n, i);
      }
      Real csl = peos->GetEOS().GetSoundSpeed(nl, Tl, Yl);
      Real csr = peos->GetEOS().GetSoundSpeed(nr, Tr, Yr);
      cs2l     = csl * csl;
      cs2r     = csr * csr;
      if (peos->restrict_cs2)
      {
        cs2l = std::min(cs2l, peos->max_cs2);
        cs2r = std::min(cs2r, peos->max_cs2);
      }
    }

    cs2_tet_l_(i) = cs2l;
    cs2_tet_r_(i) = cs2r;

    // Global eigenvalues (for LLF fallback)
    Eigenvalues::HydroEigenvalues(cs2l,
                                  w_v_u_l_(ivx - 1, i),
                                  w_norm2_v_l_(i),
                                  alpha_(i),
                                  beta_u_(ivx - 1, i),
                                  gamma_uu_(ivx - 1, ivx - 1, i),
                                  &lambda_p_l(i),
                                  &lambda_m_l(i));
    Eigenvalues::HydroEigenvalues(cs2r,
                                  w_v_u_r_(ivx - 1, i),
                                  w_norm2_v_r_(i),
                                  alpha_(i),
                                  beta_u_(ivx - 1, i),
                                  gamma_uu_(ivx - 1, ivx - 1, i),
                                  &lambda_p_r(i),
                                  &lambda_m_r(i));
  }

  // --- global wave speed for LLF fallback ---------------------------------
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
    const Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
    lambda(i)           = lambda_rescaling * std::max(lambda_r, -lambda_l);
  }

  // --- global conserved / fluxes (for LLF fallback) -----------------------
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    cons_l_(IDN, i) = w_rho_l_(i) * W_l_(i) * sqrt_detgamma_(i);
    cons_l_(IEN, i) = sqrt_detgamma_(i) *
      (w_hrho_l_(i) * SQR(W_l_(i)) - w_rho_l_(i) * W_l_(i) - w_p_l_(i));
    cons_r_(IDN, i) = w_rho_r_(i) * W_r_(i) * sqrt_detgamma_(i);
    cons_r_(IEN, i) = sqrt_detgamma_(i) *
      (w_hrho_r_(i) * SQR(W_r_(i)) - w_rho_r_(i) * W_r_(i) - w_p_r_(i));
  }

  for (int ax = 0; ax < NDIM; ++ax)
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      cons_l_(IVX + ax, i) = sqrt_detgamma_(i) *
        (w_hrho_l_(i) * W_l_(i) * w_util_d_l_(ax, i));
      cons_r_(IVX + ax, i) = sqrt_detgamma_(i) *
        (w_hrho_r_(i) * W_r_(i) * w_util_d_r_(ax, i));
    }
  }

  Real alpha_vmb_l_u_, alpha_vmb_r_u_;
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    alpha_vmb_l_u_ = alpha_(i) * w_v_u_l_(ivx - 1, i) - beta_u_(ivx - 1, i);
    flux_l_(IDN, i) = cons_l_(IDN, i) * alpha_vmb_l_u_;
    flux_l_(IEN, i) = cons_l_(IEN, i) * alpha_vmb_l_u_ +
      alpha_(i) * sqrt_detgamma_(i) * w_p_l_(i) * w_v_u_l_(ivx - 1, i);

    alpha_vmb_r_u_ = alpha_(i) * w_v_u_r_(ivx - 1, i) - beta_u_(ivx - 1, i);
    flux_r_(IDN, i) = cons_r_(IDN, i) * alpha_vmb_r_u_;
    flux_r_(IEN, i) = cons_r_(IEN, i) * alpha_vmb_r_u_ +
      alpha_(i) * sqrt_detgamma_(i) * w_p_r_(i) * w_v_u_r_(ivx - 1, i);
  }

  for (int ax = 0; ax < NDIM; ++ax)
  {
#pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      alpha_vmb_l_u_ =
        alpha_(i) * w_v_u_l_(ivx - 1, i) - beta_u_(ivx - 1, i);
      flux_l_(IVX + ax, i) = cons_l_(IVX + ax, i) * alpha_vmb_l_u_;
      alpha_vmb_r_u_ =
        alpha_(i) * w_v_u_r_(ivx - 1, i) - beta_u_(ivx - 1, i);
      flux_r_(IVX + ax, i) = cons_r_(IVX + ax, i) * alpha_vmb_r_u_;
    }
  }

#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    flux_l_(ivx, i) += w_p_l_(i) * alpha_(i) * sqrt_detgamma_(i);
    flux_r_(ivx, i) += w_p_r_(i) * alpha_(i) * sqrt_detgamma_(i);
  }

  // ====================================================================
  //  PER-CELL HLLC CORE
  // ====================================================================

#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    hllc_wave_side_(i) = 0.0;
  }

  for (int i = il; i <= iu; ++i)
  {
    ++hllc_ncells_;

    const Real alpha  = alpha_(i);
    const Real beta_d = beta_u_(d, i);
    const Real sqrt_gdd =
      std::sqrt(std::max(gamma_uu_(d, d, i), static_cast<Real>(0)));
    const Real inv_sqrt_gdd = (sqrt_gdd > 0.0) ? 1.0 / sqrt_gdd : 1.0;

    // 2x2 sub-determinant for (a,b) block
    const Real det2 = gamma_dd_(a, a, i) * gamma_dd_(b, b, i)
                    - SQR(gamma_dd_(a, b, i));
    const Real inv_sqrt_gbb  = 1.0 / std::sqrt(std::max(gamma_dd_(b, b, i),
                                                 static_cast<Real>(1e-30)));
    const Real D_hat =
      (det2 > 0.0) ? inv_sqrt_gbb / std::sqrt(det2) : 0.0;

    // Covariant shift for tangential
    Real beta_a_cov = 0, beta_b_cov = 0;
    for (int jj = 0; jj < 3; ++jj)
    {
      beta_a_cov += gamma_dd_(a, jj, i) * beta_u_(jj, i);
      beta_b_cov += gamma_dd_(b, jj, i) * beta_u_(jj, i);
    }

    // --- Covariant tetrad entries (face-dependent, same for L and R) ---
    const Real e_cov_dd = inv_sqrt_gdd;
    const Real e_a_d = D_hat * (gamma_dd_(d, a, i) * gamma_dd_(b, b, i)
                              - gamma_dd_(d, b, i) * gamma_dd_(a, b, i));
    const Real e_a_a = D_hat * det2;
    const Real e_cov_ad = e_a_d;
    const Real e_cov_aa = e_a_a;
    const Real e_cov_bd = gamma_dd_(d, b, i) * inv_sqrt_gbb;
    const Real e_cov_ba = gamma_dd_(a, b, i) * inv_sqrt_gbb;
    const Real e_cov_bb = std::sqrt(std::max(gamma_dd_(b, b, i),
                                     static_cast<Real>(0)));

    // ================================================================
    //  LEFT STATE
    // ================================================================

    // Tetrad velocity: v^(i^) = (e_(i^)_t + e_(i^)_j v^j)/alpha
    // with v^j = alpha*v_u[j] - beta_u[j] (transport velocity). For the NORMAL
    // direction the e_(d^)_t = beta^d/sqrt(gamma^dd) time component and the
    // beta^d in v^d cancel against alpha, leaving v^(d^) = v_u[d]/sqrt(gamma^dd).
    v_tet_l_(d, i) = w_v_u_l_(d, i) / sqrt_gdd;
    {
      const Real vK_d = alpha * w_v_u_l_(d, i) - beta_u_(d, i);
      const Real vK_a = alpha * w_v_u_l_(a, i) - beta_u_(a, i);
      const Real vK_b = alpha * w_v_u_l_(b, i) - beta_u_(b, i);
      Real e_a_t = D_hat * (beta_a_cov * gamma_dd_(b, b, i)
                          - beta_b_cov * gamma_dd_(a, b, i));
      v_tet_l_(a, i) =
        (e_a_t + e_a_d * vK_d + e_a_a * vK_a) / alpha;
      Real e_b_t = beta_b_cov * inv_sqrt_gbb;
      Real e_b_d = gamma_dd_(d, b, i) * inv_sqrt_gbb;
      Real e_b_a = gamma_dd_(a, b, i) * inv_sqrt_gbb;
      Real e_b_b = std::sqrt(std::max(gamma_dd_(b, b, i),
                              static_cast<Real>(0)));
      v_tet_l_(b, i) =
        (e_b_t + e_b_d * vK_d + e_b_a * vK_a + e_b_b * vK_b) / alpha;
    }

    const Real rho_l   = w_rho_l_(i);
    const Real P_l     = w_p_l_(i);
    const Real rho_h_l = w_hrho_l_(i);
    const Real W_l     = W_l_(i);
    const Real W2_l    = SQR(W_l);

    // Tetrad cons (undensitized):
    q_tet_l_(0, i) = rho_l * W_l;
    q_tet_l_(1, i) = rho_h_l * W2_l * v_tet_l_(d, i);
    q_tet_l_(2, i) = rho_h_l * W2_l * v_tet_l_(a, i);
    q_tet_l_(3, i) = rho_h_l * W2_l * v_tet_l_(b, i);
    q_tet_l_(4, i) = rho_h_l * W2_l - P_l;

    // Tetrad fluxes:
    const Real vn_l = v_tet_l_(d, i);
    f_tet_l_(0, i) = q_tet_l_(0, i) * vn_l;
    f_tet_l_(1, i) = q_tet_l_(1, i) * vn_l + P_l;
    f_tet_l_(2, i) = q_tet_l_(2, i) * vn_l;
    f_tet_l_(3, i) = q_tet_l_(3, i) * vn_l;
    f_tet_l_(4, i) = q_tet_l_(4, i) * vn_l + P_l * vn_l;

    // Tetrad SR eigenvalues:
    const Real v2_l = w_norm2_v_l_(i);
    const Real cs2l = cs2_tet_l_(i);
    lam_p_tet_l(i) = sr_eigen_plus(vn_l, v2_l, cs2l);
    lam_m_tet_l(i) = sr_eigen_minus(vn_l, v2_l, cs2l);

    // ================================================================
    //  RIGHT STATE
    // ================================================================

    v_tet_r_(d, i) = w_v_u_r_(d, i) / sqrt_gdd;
    {
      const Real vK_d = alpha * w_v_u_r_(d, i) - beta_u_(d, i);
      const Real vK_a = alpha * w_v_u_r_(a, i) - beta_u_(a, i);
      const Real vK_b = alpha * w_v_u_r_(b, i) - beta_u_(b, i);
      Real e_a_t = D_hat * (beta_a_cov * gamma_dd_(b, b, i)
                          - beta_b_cov * gamma_dd_(a, b, i));
      v_tet_r_(a, i) =
        (e_a_t + e_a_d * vK_d + e_a_a * vK_a) / alpha;
      Real e_b_t = beta_b_cov * inv_sqrt_gbb;
      Real e_b_d = gamma_dd_(d, b, i) * inv_sqrt_gbb;
      Real e_b_a = gamma_dd_(a, b, i) * inv_sqrt_gbb;
      Real e_b_b = std::sqrt(std::max(gamma_dd_(b, b, i),
                              static_cast<Real>(0)));
      v_tet_r_(b, i) =
        (e_b_t + e_b_d * vK_d + e_b_a * vK_a + e_b_b * vK_b) / alpha;
    }

    const Real rho_r   = w_rho_r_(i);
    const Real P_r     = w_p_r_(i);
    const Real rho_h_r = w_hrho_r_(i);
    const Real W_r     = W_r_(i);
    const Real W2_r    = SQR(W_r);

    q_tet_r_(0, i) = rho_r * W_r;
    q_tet_r_(1, i) = rho_h_r * W2_r * v_tet_r_(d, i);
    q_tet_r_(2, i) = rho_h_r * W2_r * v_tet_r_(a, i);
    q_tet_r_(3, i) = rho_h_r * W2_r * v_tet_r_(b, i);
    q_tet_r_(4, i) = rho_h_r * W2_r - P_r;

    const Real vn_r = v_tet_r_(d, i);
    f_tet_r_(0, i) = q_tet_r_(0, i) * vn_r;
    f_tet_r_(1, i) = q_tet_r_(1, i) * vn_r + P_r;
    f_tet_r_(2, i) = q_tet_r_(2, i) * vn_r;
    f_tet_r_(3, i) = q_tet_r_(3, i) * vn_r;
    f_tet_r_(4, i) = q_tet_r_(4, i) * vn_r + P_r * vn_r;

    const Real v2_r = w_norm2_v_r_(i);
    const Real cs2r = cs2_tet_r_(i);
    lam_p_tet_r(i) = sr_eigen_plus(vn_r, v2_r, cs2r);
    lam_m_tet_r(i) = sr_eigen_minus(vn_r, v2_r, cs2r);

    // ================================================================
    //  LAMBDA ESTIMATE & INTERFACE VELOCITY
    // ================================================================
    const Real lam_L = std::min(lam_m_tet_l(i), lam_m_tet_r(i));
    const Real lam_R = std::max(lam_p_tet_l(i), lam_p_tet_r(i));

    const Real v_iface = beta_d / (alpha * sqrt_gdd);  // tetrad frame

    // Select physical or star state; back-transform once afterward.
    // Physical supersonic branches copy the upwind state directly;
    // LLF fallback paths inside the else use continue to skip the
    // common back-transform (LLF is already in the global frame).
    Real q_sel[NHYDRO], f_sel[NHYDRO];

    if (lam_L > v_iface) {
      // Entire fan right of interface: select left physical state
      hllc_wave_side_(i) = -1.0;
      for (int n = 0; n < NHYDRO; ++n) {
        q_sel[n] = q_tet_l_(n, i);
        f_sel[n] = f_tet_l_(n, i);
      }
    } else if (lam_R <= v_iface) {
      // Entire fan left of interface: select right physical state
      hllc_wave_side_(i) = 1.0;
      for (int n = 0; n < NHYDRO; ++n) {
        q_sel[n] = q_tet_r_(n, i);
        f_sel[n] = f_tet_r_(n, i);
      }
    } else {
    // Interface is inside the acoustic fan -- construct HLL states
    const Real dlam  = lam_R - lam_L;
    const Real dlam_tol =
      hlle_eps_abs +
      hlle_eps_rel * std::max({ std::abs(lam_L), std::abs(lam_R), 1.0 });

    if (dlam <= dlam_tol)
    {
      ++hllc_nfallback_;
      for (int n = 0; n < NHYDRO; ++n)
        flux(n, k, j, i) = 0.5 *
          ((flux_l_(n, i) + flux_r_(n, i)) -
           lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
      continue;
    }

    // ================================================================
    //  HLL STATE & FLUX (5-vector)
    // ================================================================
    const Real oo_dlam = 1.0 / dlam;
    for (int n = 0; n < NHYDRO; ++n)
    {
      q_hll(n, i) = (lam_R * q_tet_r_(n, i) - lam_L * q_tet_l_(n, i)
                    + f_tet_l_(n, i) - f_tet_r_(n, i)) * oo_dlam;
      f_hll(n, i) = (lam_R * f_tet_l_(n, i) - lam_L * f_tet_r_(n, i)
                    + lam_R * lam_L *
                      (q_tet_r_(n, i) - q_tet_l_(n, i))) * oo_dlam;
    }

    // ================================================================
    //  CONTACT WAVE SPEED lambda_c (scale-invariant quadratic)
    // ================================================================
    const Real a_q = f_hll(4, i);
    const Real b_q = -(q_hll(4, i) + f_hll(1, i));
    const Real c_q = q_hll(1, i);

    const Real coeff_scale = std::max({ std::abs(a_q), std::abs(b_q), std::abs(c_q) });
    if (!std::isfinite(coeff_scale) || coeff_scale <= 0.0) {
      ++hllc_nfallback_;
      for (int n = 0; n < NHYDRO; ++n)
        flux(n, k, j, i) = 0.5 *
          ((flux_l_(n, i) + flux_r_(n, i)) -
           lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
      continue;
    }

    const Real a = a_q / coeff_scale;
    const Real b = b_q / coeff_scale;
    const Real c = c_q / coeff_scale;
    const Real coeff_tol = 1e-12;

    if (std::abs(a) > coeff_tol) {
      // Quadratic branch
      Real disc = b * b - 4.0 * a * c;
      const Real disc_scale = b * b + 4.0 * std::abs(a * c);
      const Real disc_tol   = 1e-12 * disc_scale;

      if (!std::isfinite(disc) || disc < -disc_tol) {
        // Materially negative discriminant -- fall back to LLF
        ++hllc_nfallback_;
        for (int n = 0; n < NHYDRO; ++n)
          flux(n, k, j, i) = 0.5 *
            ((flux_l_(n, i) + flux_r_(n, i)) -
             lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
        continue;
      }
      disc = std::max(disc, 0.0);

      // Cancellation-resistant minus root
      if (b >= 0.0)
        lambda_c(i) = (-b - std::sqrt(disc)) / (2.0 * a);
      else
        lambda_c(i) = -2.0 * c / (b - std::sqrt(disc));
    } else if (std::abs(b) > coeff_tol) {
      // Linear branch: a ~ 0 but b is resolved
      lambda_c(i) = -c / b;
    } else {
      // Both a and b are unresolved -- fall back to LLF
      ++hllc_nfallback_;
      for (int n = 0; n < NHYDRO; ++n)
        flux(n, k, j, i) = 0.5 *
          ((flux_l_(n, i) + flux_r_(n, i)) -
           lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
      continue;
    }

    // Mignone & Bodo (2005) Appendix A, Proposition 4 proves
    // lambda_c in [lam_L, lam_R] for all physically admissible states.
    if (lambda_c(i) < lam_L || lambda_c(i) > lam_R) {
      ++hllc_nlambda_c_oor_;
      for (int n = 0; n < NHYDRO; ++n)
        flux(n, k, j, i) = 0.5 *
          ((flux_l_(n, i) + flux_r_(n, i)) -
           lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
      continue;
    }

    // Auxiliary contact pressure (not thermodynamic P, may be negative)
    P_c(i) = -lambda_c(i) * f_hll(4, i) + f_hll(1, i);

    // ================================================================
    //  INTERMEDIATE STATE (one selected star region)
    // ================================================================

    if (lambda_c(i) > v_iface) {
      // Interface in left star region: construct only cL
      hllc_wave_side_(i) = -1.0;

      const Real oo_dcL = 1.0 / (lam_L - lambda_c(i));
      const Real dv_L  = lam_L - vn_l;

      q_c_l(0, i) = q_tet_l_(0, i) * dv_L * oo_dcL;
      q_c_l(1, i) = (q_tet_l_(1, i) * dv_L + (P_c(i) - P_l)) * oo_dcL;
      q_c_l(2, i) = rho_h_l * W2_l * v_tet_l_(a, i) * dv_L * oo_dcL;
      q_c_l(3, i) = rho_h_l * W2_l * v_tet_l_(b, i) * dv_L * oo_dcL;
      q_c_l(4, i) = (q_tet_l_(4, i) * dv_L + P_c(i) * lambda_c(i)
                      - P_l * vn_l) * oo_dcL;
      for (int n = 0; n < NHYDRO; ++n)
        f_c_l(n, i) = f_tet_l_(n, i)
                      + lam_L * (q_c_l(n, i) - q_tet_l_(n, i));
      for (int n = 0; n < NHYDRO; ++n) {
        q_sel[n] = q_c_l(n, i);
        f_sel[n] = f_c_l(n, i);
      }
    } else {
      // Interface in right star region: construct only cR
      hllc_wave_side_(i) = 1.0;

      const Real oo_dcR = 1.0 / (lam_R - lambda_c(i));
      const Real dv_R  = lam_R - vn_r;

      q_c_r(0, i) = q_tet_r_(0, i) * dv_R * oo_dcR;
      q_c_r(1, i) = (q_tet_r_(1, i) * dv_R + (P_c(i) - P_r)) * oo_dcR;
      q_c_r(2, i) = rho_h_r * W2_r * v_tet_r_(a, i) * dv_R * oo_dcR;
      q_c_r(3, i) = rho_h_r * W2_r * v_tet_r_(b, i) * dv_R * oo_dcR;
      q_c_r(4, i) = (q_tet_r_(4, i) * dv_R + P_c(i) * lambda_c(i)
                      - P_r * vn_r) * oo_dcR;
      for (int n = 0; n < NHYDRO; ++n)
        f_c_r(n, i) = f_tet_r_(n, i)
                      + lam_R * (q_c_r(n, i) - q_tet_r_(n, i));
      for (int n = 0; n < NHYDRO; ++n) {
        q_sel[n] = q_c_r(n, i);
        f_sel[n] = f_c_r(n, i);
      }
    }

    }  // end else (acoustic fan)

    // ================================================================
    //  FLUX BACK-TRANSFORM --> DENSITIZED (common to all branches)
    // ================================================================
    const Real sqrtg = sqrt_detgamma_(i);
    const Real et_x  = -beta_d / alpha;               // e^(t^)^d
    const Real ex_x  = sqrt_gdd;                      // e^(d^)^d

    Real F_D, F_Sd, F_Sa, F_Sb, F_E;
    FluxBackTransform(d, a, b,
                      sqrtg, alpha, et_x, ex_x,
                      e_cov_dd, e_cov_ad, e_cov_aa,
                      e_cov_bd, e_cov_ba, e_cov_bb,
                      q_sel, f_sel,
                      F_D, F_Sd, F_Sa, F_Sb, F_E);

    const Real F_tau = F_E - F_D;
    if (!std::isfinite(F_D) || !std::isfinite(F_Sd) ||
        !std::isfinite(F_Sa) || !std::isfinite(F_Sb) ||
        !std::isfinite(F_tau))
    {
      hllc_wave_side_(i) = 0.0;
      ++hllc_nfallback_;
      for (int n = 0; n < NHYDRO; ++n)
        flux(n, k, j, i) = 0.5 *
          ((flux_l_(n, i) + flux_r_(n, i)) -
           lambda(i) * (cons_r_(n, i) - cons_l_(n, i)));
      continue;
    }

    flux(IDN, k, j, i)      = F_D;
    flux(IVX + d, k, j, i)  = F_Sd;
    flux(IVX + a, k, j, i)  = F_Sa;
    flux(IVX + b, k, j, i)  = F_Sb;
    flux(IEN, k, j, i)      = F_tau;
    ++hllc_nhit_;
  }  // for i = il..iu

  // ====================================================================
  //  PASSIVE SCALAR ADVECTION
  // ====================================================================
  if (!pmy_block->precon->xorder_upwind_scalars)
  {
    for (int n = 0; n < NSCALARS; ++n)
    {
#pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        s_flux(n, k, j, i) =
          0.5 * ((flux_l_(IDN, i) * pscalars_l_(n, i) +
                  flux_r_(IDN, i) * pscalars_r_(n, i)) -
                 lambda(i) *
                   (cons_r_(IDN, i) * pscalars_r_(n, i) -
                    cons_l_(IDN, i) * pscalars_l_(n, i)));
      }
    }
  }
  else
  {
    for (int n = 0; n < NSCALARS; ++n)
    {
      for (int i = il; i <= iu; ++i)
      {
        const Real mass_flx = flux(IDN, k, j, i);
        const Real side = hllc_wave_side_(i);
        if (side < 0.0) {
          s_flux(n, k, j, i) = mass_flx * pscalars_l_(n, i);
        } else if (side > 0.0) {
          s_flux(n, k, j, i) = mass_flx * pscalars_r_(n, i);
        } else {
          s_flux(n, k, j, i) = (mass_flx >= 0.0) ? mass_flx * pscalars_l_(n, i)
                                                  : mass_flx * pscalars_r_(n, i);
        }
      }
    }
  }
  if (hllc_print_interval_ > 0) {
    Mesh* pm = pmy_block->pmy_mesh;
    if (pm->ncycle % hllc_print_interval_ == 0 &&
        pm->ncycle != hllc_last_print_cycle_) {
      if (Globals::my_rank == 0) {
        Real hit_rate = (hllc_ncells_ > 0)
          ? static_cast<Real>(100) * hllc_nhit_
              / static_cast<Real>(hllc_ncells_)
          : static_cast<Real>(0);
        fprintf(stderr,
          "[HLLC] cycle=%d ncells=%lu nhit=%lu nfallback=%lu "
          "nlc_oor=%lu hit_rate=%.1f%%\n",
          pm->ncycle,
          static_cast<unsigned long>(hllc_ncells_),
          static_cast<unsigned long>(hllc_nhit_),
          static_cast<unsigned long>(hllc_nfallback_),
          static_cast<unsigned long>(hllc_nlambda_c_oor_),
          hit_rate);
      }
      hllc_ncells_        = 0;
      hllc_nhit_          = 0;
      hllc_nfallback_     = 0;
      hllc_nlambda_c_oor_ = 0;
      hllc_last_print_cycle_ = pm->ncycle;
    }
  }
}
