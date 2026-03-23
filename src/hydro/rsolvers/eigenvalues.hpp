#ifndef HYDRO_RSOLVERS_EIGENVALUES_HPP_
#define HYDRO_RSOLVERS_EIGENVALUES_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file eigenvalues.hpp
//  \brief Characteristic wavespeed (eigenvalue) computations for GR hydro and
//  MHD
//         Riemann solvers.
//
//  These functions compute the eigenvalues (characteristic wavespeeds) of the
//  GR hydrodynamics and MHD systems.  Despite their former names
//  (SoundSpeedsGR, FastMagnetosonicSpeedsGR), they output the plus/minus
//  eigenvalues lambda_+/-, not sound speeds.
//
//  Hydro follows the Font (2008) / HARM vchar() procedure.
//  MHD solves the quadratic for fast magnetosonic eigenvalues.
//
//  Both functions take pre-computed thermodynamic inputs (cs^2, rho*h) so that
//  EOS table lookups are performed once at the caller rather than redundantly
//  inside the eigenvalue computation.  The restrict_cs2 clamp (if needed) must
//  also be applied by the caller before passing cs_sq.

// C++ headers
#include <algorithm>  // std::max
#include <cmath>      // std::sqrt, std::isfinite

// Athena++ headers
#include "../../athena.hpp"  // Real
#include "../../defs.hpp"    // SQR

namespace Eigenvalues
{

//----------------------------------------------------------------------------------------
// HydroEigenvalues
//
// Inputs:
//   cs_sq:   sound speed squared (pre-computed, already clamped if
//   restrict_cs2) vi:      Eulerian 3-velocity component v^i in the normal
//   direction v2:      norm-squared of Eulerian 3-velocity  gamma_{ab} v^a v^b
//   alpha:   lapse
//   betai:   shift component beta^i in the normal direction
//   gammaii: inverse 3-metric component gamma^{ii} in the normal direction
// Outputs:
//   plambda_plus:  most positive eigenvalue
//   plambda_minus: most negative eigenvalue

inline void HydroEigenvalues(Real cs_sq,
                             Real vi,
                             Real v2,
                             Real alpha,
                             Real betai,
                             Real gammaii,
                             Real* plambda_plus,
                             Real* plambda_minus)
{
  const Real cs = std::sqrt(cs_sq);

  const Real sqrt_term = std::sqrt(
    (1 - v2) * (gammaii * (1.0 - v2 * cs_sq) - vi * vi * (1.0 - cs_sq)));

  Real root_1 =
    alpha * (vi * (1.0 - cs_sq) + cs * sqrt_term) / (1.0 - v2 * cs_sq) - betai;
  Real root_2 =
    alpha * (vi * (1.0 - cs_sq) - cs * sqrt_term) / (1.0 - v2 * cs_sq) - betai;

  if (!std::isfinite(root_1 + root_2))
  {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2)
  {
    *plambda_plus  = root_1;
    *plambda_minus = root_2;
  }
  else
  {
    *plambda_plus  = root_2;
    *plambda_minus = root_1;
  }
}

//----------------------------------------------------------------------------------------
// MHDEigenvalues
//
// Additional inputs beyond HydroEigenvalues:
//   cs_sq:   sound speed squared (pre-computed, already clamped if
//   restrict_cs2) rho_h:   rho * h  (rest-mass density times specific
//   enthalpy, pre-computed) bsq:     covariant magnetic field squared  b_mu
//   b^mu

inline void MHDEigenvalues(Real cs_sq,
                           Real rho_h,
                           Real bsq,
                           Real vi,
                           Real v2,
                           Real alpha,
                           Real betai,
                           Real gammaii,
                           Real* plambda_plus,
                           Real* plambda_minus)
{
  // Reconstruct 4-velocity and contravariant metric components
  Real Wlor = std::sqrt(1.0 - v2);
  Wlor      = 1.0 / Wlor;
  Real u0   = Wlor / alpha;
  Real g00  = -1.0 / (alpha * alpha);
  Real g01  = betai / (alpha * alpha);
  Real u1   = (vi - betai / alpha) * Wlor;
  Real g11  = gammaii - betai * betai / (alpha * alpha);

  Real va_sq  = bsq / (bsq + rho_h);
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Solve quadratic for fast magnetosonic eigenvalues
  Real a      = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b      = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
  Real c      = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real d      = std::max(SQR(b) - 4.0 * a * c, 0.0);
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0 * a);
  Real root_2 = (-b - d_sqrt) / (2.0 * a);

  if (!std::isfinite(root_1 + root_2))
  {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2)
  {
    *plambda_plus  = root_1;
    *plambda_minus = root_2;
  }
  else
  {
    *plambda_plus  = root_2;
    *plambda_minus = root_1;
  }
}

}  // namespace Eigenvalues

#endif  // HYDRO_RSOLVERS_EIGENVALUES_HPP_
