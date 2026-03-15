#ifndef EOS_PRIMITIVE_SOLVER_HELPER_HPP_
#define EOS_PRIMITIVE_SOLVER_HELPER_HPP_
//========================================================================================
//! \file primitive_solver_helper.hpp
//  \brief Inline helpers that marshal between Athena's AthenaArray layout and
//         the flat Real[] arrays used by PrimitiveSolver.
//
//  Two indexing families are provided:
//    - 3-D  (k, j, i)  - for full grid arrays   prim(n, k, j, i)
//    - 1-D  (i)         - for pencil/slice arrays prim(n, i)
//
//  All functions are header-only (inline) so there is zero link-time overhead.
//
//  This header is only included when FLUID_ENABLED - every symbol it uses
//  (NPRIM, ITM, IYF, PrimitiveSolver EOS types, ...) requires fluid support.
//========================================================================================

#include "../athena.hpp"         // Real, NHYDRO, NSCALARS, IDN, ...
#include "../athena_arrays.hpp"  // AthenaArray

// PrimitiveSolver headers (available because we are gated behind FLUID_ENABLED)
#include "../z4c/primitive/eos.hpp"
#include "include_eos.hpp"

namespace PrimHelper {

// -------------------------------------------------------------------------
// Gather: AthenaArrays  -->  flat prim_pt[NPRIM]
// -------------------------------------------------------------------------

/// 3-D gather: hydro prims + scalars --> prim_pt[NPRIM]
///   density is converted from mass density to number density (rho / mb)
inline void GatherPrim(const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &prim_scalar,
                       Real prim_pt[NPRIM],
                       int k, int j, int i,
                       Real mb)
{
  prim_pt[IDN] = prim(IDN, k, j, i) / mb;
  prim_pt[IVX] = prim(IVX, k, j, i);
  prim_pt[IVY] = prim(IVY, k, j, i);
  prim_pt[IVZ] = prim(IVZ, k, j, i);
  prim_pt[IPR] = prim(IPR, k, j, i);
  for (int n = 0; n < NSCALARS; ++n) {
    prim_pt[IYF + n] = prim_scalar(n, k, j, i);
  }
}

/// 1-D (pencil) gather
inline void GatherPrim(const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &prim_scalar,
                       Real prim_pt[NPRIM],
                       int i,
                       Real mb)
{
  prim_pt[IDN] = prim(IDN, i) / mb;
  prim_pt[IVX] = prim(IVX, i);
  prim_pt[IVY] = prim(IVY, i);
  prim_pt[IVZ] = prim(IVZ, i);
  prim_pt[IPR] = prim(IPR, i);
  for (int n = 0; n < NSCALARS; ++n) {
    prim_pt[IYF + n] = prim_scalar(n, i);
  }
}

// -------------------------------------------------------------------------
// Scatter: flat prim_pt[NPRIM]  -->  AthenaArrays
// -------------------------------------------------------------------------

/// 3-D scatter: prim_pt[NPRIM] --> hydro prims + scalars
///   density is converted from number density to mass density (n * mb)
inline void ScatterPrim(const Real prim_pt[NPRIM],
                        AthenaArray<Real> &prim,
                        AthenaArray<Real> &prim_scalar,
                        int k, int j, int i,
                        Real mb)
{
  prim(IDN, k, j, i) = prim_pt[IDN] * mb;
  prim(IVX, k, j, i) = prim_pt[IVX];
  prim(IVY, k, j, i) = prim_pt[IVY];
  prim(IVZ, k, j, i) = prim_pt[IVZ];
  prim(IPR, k, j, i) = prim_pt[IPR];
  for (int n = 0; n < NSCALARS; ++n) {
    prim_scalar(n, k, j, i) = prim_pt[IYF + n];
  }
}

/// 1-D (pencil) scatter
inline void ScatterPrim(const Real prim_pt[NPRIM],
                        AthenaArray<Real> &prim,
                        AthenaArray<Real> &prim_scalar,
                        int i,
                        Real mb)
{
  prim(IDN, i) = prim_pt[IDN] * mb;
  prim(IVX, i) = prim_pt[IVX];
  prim(IVY, i) = prim_pt[IVY];
  prim(IVZ, i) = prim_pt[IVZ];
  prim(IPR, i) = prim_pt[IPR];
  for (int n = 0; n < NSCALARS; ++n) {
    prim_scalar(n, i) = prim_pt[IYF + n];
  }
}

// -------------------------------------------------------------------------
// Scalar-only gather / scatter  (used by PassiveScalars)
// -------------------------------------------------------------------------

/// 3-D gather: scalars only --> Y[MAX_SPECIES]
inline void GatherScalars(const AthenaArray<Real> &prim_scalar,
                          Real Y[MAX_SPECIES],
                          int k, int j, int i)
{
  for (int n = 0; n < NSCALARS; ++n) {
    Y[n] = prim_scalar(n, k, j, i);
  }
}

/// 1-D gather: scalars only --> Y[MAX_SPECIES]
inline void GatherScalars(const AthenaArray<Real> &prim_scalar,
                          Real Y[MAX_SPECIES],
                          int i)
{
  for (int n = 0; n < NSCALARS; ++n) {
    Y[n] = prim_scalar(n, i);
  }
}

/// 3-D scatter: Y[MAX_SPECIES] --> scalars
inline void ScatterScalars(const Real Y[MAX_SPECIES],
                           AthenaArray<Real> &prim_scalar,
                           int k, int j, int i)
{
  for (int n = 0; n < NSCALARS; ++n) {
    prim_scalar(n, k, j, i) = Y[n];
  }
}

/// 1-D scatter: Y[MAX_SPECIES] --> scalars
inline void ScatterScalars(const Real Y[MAX_SPECIES],
                           AthenaArray<Real> &prim_scalar,
                           int i)
{
  for (int n = 0; n < NSCALARS; ++n) {
    prim_scalar(n, i) = Y[n];
  }
}

// -------------------------------------------------------------------------
// SetPrimAtmo: set atmosphere (failure-response) values at a grid point
// -------------------------------------------------------------------------

using EOS_t = Primitive::EOS<Primitive::EOS_POLICY, Primitive::ERROR_POLICY>;

/// 3-D: set atmosphere at grid point (k, j, i).
/// Optionally writes temperature if ptemperature != nullptr.
inline void SetPrimAtmo(
  EOS_t &eos,
  AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_scalar,
  const int k, const int j, const int i,
  AthenaArray<Real> *ptemperature = nullptr)
{
  Real prim_pt[NPRIM] = {0.0};
  eos.DoFailureResponse(prim_pt);
  ScatterPrim(prim_pt, prim, prim_scalar, k, j, i,
              eos.GetBaryonMass());
  if (ptemperature) (*ptemperature)(k,j,i) = prim_pt[ITM];
}

/// 1-D (pencil): set atmosphere at slice index i.
/// Optionally writes temperature if ptemperature != nullptr.
inline void SetPrimAtmo(
  EOS_t &eos,
  AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_scalar,
  const int i,
  AthenaArray<Real> *ptemperature = nullptr)
{
  Real prim_pt[NPRIM] = {0.0};
  eos.DoFailureResponse(prim_pt);
  ScatterPrim(prim_pt, prim, prim_scalar, i,
              eos.GetBaryonMass());
  if (ptemperature) (*ptemperature)(i) = prim_pt[ITM];
}

// -------------------------------------------------------------------------
// ApplyPrimitiveFloors: apply density, species, and pressure floors
//   at a single grid point.  Handles both 3-D and 1-D (pencil) arrays
//   automatically via prim.GetDim4().
// -------------------------------------------------------------------------

/// Apply primitive floors at grid point (k, j, i) - or at pencil index i
/// when the array is 1-D (prim.GetDim4()==1).
inline void ApplyPrimitiveFloors(
  EOS_t &eos,
  AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_scalar,
  int k, int j, int i)
{
  Real Y[MAX_SPECIES] = {0.0};
  Real Wvu[3] = {};
  Real P;
  Real n;

  Real mb = eos.GetBaryonMass();

  if (prim.GetDim4() == 1)
  {
    GatherScalars(prim_scalar, Y, i);
    n = prim(IDN, i) / mb;
    P = prim(IPR, i);
    for (int a = 0; a < 3; ++a) Wvu[a] = prim(IVX + a, i);
  }
  else
  {
    GatherScalars(prim_scalar, Y, k, j, i);
    n = prim(IDN, k, j, i) / mb;
    P = prim(IPR, k, j, i);
    for (int a = 0; a < 3; ++a) Wvu[a] = prim(IVX + a, k, j, i);
  }

  eos.ApplyDensityLimits(n);
  eos.ApplySpeciesLimits(Y);
  Real T = eos.GetTemperatureFromP(n, P, Y);
  eos.ApplyPrimitiveFloor(n, Wvu, P, T, Y);

  // Push updated quantities back to Athena arrays.
  if (prim.GetDim4() == 1)
  {
    prim(IDN, i) = n * mb;
    prim(IVX, i) = Wvu[0];
    prim(IVY, i) = Wvu[1];
    prim(IVZ, i) = Wvu[2];
    prim(IPR, i) = P;
    ScatterScalars(Y, prim_scalar, i);
  }
  else
  {
    prim(IDN, k, j, i) = n * mb;
    prim(IVX, k, j, i) = Wvu[0];
    prim(IVY, k, j, i) = Wvu[1];
    prim(IVZ, k, j, i) = Wvu[2];
    prim(IPR, k, j, i) = P;
    ScatterScalars(Y, prim_scalar, k, j, i);
  }
}

} // namespace PrimHelper

#endif // EOS_PRIMITIVE_SOLVER_HELPER_HPP_
