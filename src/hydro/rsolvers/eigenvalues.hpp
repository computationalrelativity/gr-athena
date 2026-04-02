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
  const Real cs_sqrt_term =
    std::sqrt(cs_sq * (1 - v2) *
              (gammaii * (1.0 - v2 * cs_sq) - vi * vi * (1.0 - cs_sq)));
  const Real oo_denom = 1.0 / (1.0 - v2 * cs_sq);
  const Real vi_term  = vi * (1.0 - cs_sq);
  Real root_1         = alpha * (vi_term + cs_sqrt_term) * oo_denom - betai;
  Real root_2         = alpha * (vi_term - cs_sqrt_term) * oo_denom - betai;

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
// Computes fast magnetosonic eigenvalues for the GR MHD system.
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
                           Real Wlor,
                           Real alpha,
                           Real betai,
                           Real gammaii,
                           Real* plambda_plus,
                           Real* plambda_minus,
                           bool /*Wlor_is_precomputed*/)
{
  // Compute contravariant 4-metric and 4-velocity from pre-computed W
  const Real oo_alpha2 = 1.0 / (alpha * alpha);
  const Real u0        = Wlor / alpha;
  const Real g00       = -oo_alpha2;
  const Real g01       = betai * oo_alpha2;
  const Real u1        = (vi * alpha - betai) * u0;
  const Real g11       = gammaii - betai * betai * oo_alpha2;

  Real va_sq  = bsq / (bsq + rho_h);
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Solve quadratic for fast magnetosonic eigenvalues
  const Real u0sq  = SQR(u0);
  const Real u1sq  = SQR(u1);
  Real a           = u0sq - (g00 + u0sq) * cms_sq;
  Real b           = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
  Real c           = u1sq - (g11 + u1sq) * cms_sq;
  Real d           = std::max(SQR(b) - 4.0 * a * c, 0.0);
  Real d_sqrt      = std::sqrt(d);
  const Real oo_2a = 0.5 / a;
  Real root_1      = (-b + d_sqrt) * oo_2a;
  Real root_2      = (-b - d_sqrt) * oo_2a;

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
// Takes v^2 instead of W and reconstructs the Lorentz factor internally.

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
  // Reconstruct Lorentz factor from v^2, delegate to optimized overload
  Real Wlor = 1.0 / std::sqrt(1.0 - v2);
  MHDEigenvalues(cs_sq,
                 rho_h,
                 bsq,
                 vi,
                 Wlor,
                 alpha,
                 betai,
                 gammaii,
                 plambda_plus,
                 plambda_minus,
                 true);
}

}  // namespace Eigenvalues

#endif  // HYDRO_RSOLVERS_EIGENVALUES_HPP_
