#ifndef HYDRO_RSOLVERS_EIGENVALUES_HPP_
#define HYDRO_RSOLVERS_EIGENVALUES_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eigenvalues.hpp
//  \brief Characteristic wavespeed (eigenvalue) computations for GR hydro and MHD
//         Riemann solvers.
//
//  These functions compute the eigenvalues (characteristic wavespeeds) of the
//  GR hydrodynamics and MHD systems.  Despite their former names (SoundSpeedsGR,
//  FastMagnetosonicSpeedsGR), they output the plus/minus eigenvalues lambda_+/-,
//  not sound speeds.
//
//  Hydro follows the Font (2008) / HARM vchar() procedure.
//  MHD solves the quadratic for fast magnetosonic eigenvalues.
//
//  Two overloads each:
//    - Overload 1: computes cs^2 internally from (n, T, Y) via GetSoundSpeed()
//    - Overload 2: takes a pre-computed cs^2 (from auxiliary/reconstruction)

// C++ headers
#include <algorithm>  // std::min, std::max
#include <cmath>      // std::sqrt, std::isfinite, std::isnan
#include <cstdio>     // std::printf

// Athena++ headers
#include "../../athena.hpp"       // Real, NSCALARS, MAX_SPECIES
#include "../../defs.hpp"         // SQR
#include "../../eos/eos.hpp"      // EquationOfState

namespace Eigenvalues {

//----------------------------------------------------------------------------------------
// HydroEigenvalues - overload 1 (computes cs^2 internally)
//
// Inputs:
//   peos:    pointer to EquationOfState (for EOS access and speed-clamping params)
//   n:       baryon number density
//   T:       temperature
//   vi:      Eulerian 3-velocity component v^i in the normal direction
//   v2:      norm-squared of Eulerian 3-velocity  gamma_{ab} v^a v^b
//   alpha:   lapse
//   betai:   shift component beta^i in the normal direction
//   gammaii: inverse 3-metric component gamma^{ii} in the normal direction
//   prim_scalar: passive scalar array (particle fractions Y)
// Outputs:
//   plambda_plus:  most positive eigenvalue
//   plambda_minus: most negative eigenvalue

inline void HydroEigenvalues(
  EquationOfState *peos,
  Real n, Real T, Real vi, Real v2, Real alpha,
  Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus,
  Real prim_scalar[NSCALARS])
{
  Real Y[MAX_SPECIES] = {0.0};
  for (int l = 0; l < NSCALARS; l++) {
    Y[l] = prim_scalar[l];
  }

  Real cs = peos->GetEOS().GetSoundSpeed(n, T, Y);
  Real cs_sq = cs * cs;

  if ((cs_sq > peos->max_cs2) && peos->warn_unrestricted_cs2) {
    std::printf("Warning: cs_sq exceeds max_cs2");
  }

  cs_sq = std::min(cs_sq, peos->max_cs2);
  cs = std::sqrt(cs_sq);

  const Real sqrt_term = std::sqrt(
    (1 - v2) * (gammaii * (1.0 - v2 * cs_sq) - vi * vi * (1.0 - cs_sq))
  );

  Real root_1 = alpha * (vi * (1.0 - cs_sq) + cs * sqrt_term) / (1.0 - v2 * cs_sq)
                - betai;
  Real root_2 = alpha * (vi * (1.0 - cs_sq) - cs * sqrt_term) / (1.0 - v2 * cs_sq)
                - betai;

  if (!std::isfinite(root_1 + root_2)) {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
}

//----------------------------------------------------------------------------------------
// HydroEigenvalues - overload 2 (pre-computed cs^2)

inline void HydroEigenvalues(
  EquationOfState *peos,
  Real cs_2, Real n, Real T, Real vi, Real v2, Real alpha,
  Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus,
  Real prim_scalar[NSCALARS])
{
  Real Y[MAX_SPECIES] = {0.0};
  for (int l = 0; l < NSCALARS; l++) {
    Y[l] = prim_scalar[l];
  }

  if ((cs_2 > peos->max_cs2) && peos->warn_unrestricted_cs2) {
    std::printf("Warning: cs_sq exceeds max_cs2");
  }

  cs_2 = std::min(cs_2, peos->max_cs2);
  const Real cs = std::sqrt(cs_2);

  const Real sqrt_term = std::sqrt(
    (1 - v2) * (gammaii * (1.0 - v2 * cs_2) - vi * vi * (1.0 - cs_2))
  );

  Real root_1 = alpha * (vi * (1.0 - cs_2) + cs * sqrt_term) / (1.0 - v2 * cs_2)
                - betai;
  Real root_2 = alpha * (vi * (1.0 - cs_2) - cs * sqrt_term) / (1.0 - v2 * cs_2)
                - betai;

  if (!std::isfinite(root_1 + root_2)) {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
}

//----------------------------------------------------------------------------------------
// MHDEigenvalues
//
// Additional inputs beyond HydroEigenvalues:
//   bsq: covariant magnetic field squared  b_mu b^mu

inline void MHDEigenvalues(
  EquationOfState *peos,
  Real n, Real T, Real bsq,
  Real vi, Real v2, Real alpha,
  Real betai, Real gammaii,
  Real *plambda_plus, Real *plambda_minus,
  Real prim_scalar[NSCALARS])
{
  // Reconstruct 4-velocity and contravariant metric components
  Real Wlor = std::sqrt(1.0 - v2);
  Wlor = 1.0 / Wlor;
  Real u0 = Wlor / alpha;
  Real g00 = -1.0 / (alpha * alpha);
  Real g01 = betai / (alpha * alpha);
  Real u1 = (vi - betai / alpha) * Wlor;
  Real g11 = gammaii - betai * betai / (alpha * alpha);

  Real Y[MAX_SPECIES] = {0.0};
  for (int l = 0; l < NSCALARS; l++)
    Y[l] = prim_scalar[l];

  Real cs = peos->GetEOS().GetSoundSpeed(n, T, Y);
  Real cs_sq = cs * cs;

  if ((cs_sq > peos->max_cs2) && peos->warn_unrestricted_cs2) {
    std::printf("Warning: cs_sq exceeds max_cs2");
  }

  cs_sq = std::min(cs_sq, peos->max_cs2);
  cs = std::sqrt(cs_sq);

  Real mb = peos->GetEOS().GetBaryonMass();
  Real va_sq = bsq / (bsq + n * mb * peos->GetEOS().GetEnthalpy(n, T, Y));
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Solve quadratic for fast magnetosonic eigenvalues
  Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real d = std::max(SQR(b) - 4.0 * a * c, 0.0);
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0 * a);
  Real root_2 = (-b - d_sqrt) / (2.0 * a);

  if (std::isnan(root_1) || std::isnan(root_2)) {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
}

//----------------------------------------------------------------------------------------
// MHDEigenvalues - overload 2 (pre-computed cs^2)

inline void MHDEigenvalues(
  EquationOfState *peos,
  Real cs_2, Real n, Real T, Real bsq,
  Real vi, Real v2, Real alpha,
  Real betai, Real gammaii,
  Real *plambda_plus, Real *plambda_minus,
  Real prim_scalar[NSCALARS])
{
  // Reconstruct 4-velocity and contravariant metric components
  Real Wlor = std::sqrt(1.0 - v2);
  Wlor = 1.0 / Wlor;
  Real u0 = Wlor / alpha;
  Real g00 = -1.0 / (alpha * alpha);
  Real g01 = betai / (alpha * alpha);
  Real u1 = (vi - betai / alpha) * Wlor;
  Real g11 = gammaii - betai * betai / (alpha * alpha);

  Real Y[MAX_SPECIES] = {0.0};
  for (int l = 0; l < NSCALARS; l++)
    Y[l] = prim_scalar[l];

  if ((cs_2 > peos->max_cs2) && peos->warn_unrestricted_cs2) {
    std::printf("Warning: cs_sq exceeds max_cs2");
  }

  if (peos->restrict_cs2) {
    cs_2 = std::min(cs_2, peos->max_cs2);
  }

  Real mb = peos->GetEOS().GetBaryonMass();
  Real va_sq = bsq / (bsq + n * mb * peos->GetEOS().GetEnthalpy(n, T, Y));
  Real cms_sq = cs_2 + va_sq - cs_2 * va_sq;

  // Solve quadratic for fast magnetosonic eigenvalues
  Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real d = std::max(SQR(b) - 4.0 * a * c, 0.0);
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0 * a);
  Real root_2 = (-b - d_sqrt) / (2.0 * a);

  if (!std::isfinite(root_1 + root_2)) {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
}

}  // namespace Eigenvalues

#endif // HYDRO_RSOLVERS_EIGENVALUES_HPP_
