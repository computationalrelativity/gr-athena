// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
namespace
{

/*
// BAM conventions (e.g.):
// zl(n,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
// zr(n,i) = rec1d_m_weno5(zimt,zimo,zi,zipo,zipt);
//
// or- flip the arguments and write one function
//
// zl(n,i) = rec1d_p_weno5(zimt,zimo,zi,zipo,zipt);
// zr(n,i) = rec1d_p_weno5(zipt,zipo,zi,zimo,zimt);
*/

static const Real alpha = 0.7;  // CENO3 coef

#pragma omp declare simd
Real rec1d_p_ceno3(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt);

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructCeno3X1(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k, j, i - 2);
    const Real zimo = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i + 1);
    const Real zipt = z(n_src, k, j, i + 2);

    zl_(n_tar, i + 1) = rec1d_p_ceno3(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i)     = rec1d_p_ceno3(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructCeno3X2(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k, j - 2, i);
    const Real zimo = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j + 1, i);
    const Real zipt = z(n_src, k, j + 2, i);

    zl_(n_tar, i) = rec1d_p_ceno3(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_ceno3(zipt, zipo, zi, zimo, zimt);
  }
}

void Reconstruction::ReconstructCeno3X3(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k - 2, j, i);
    const Real zimo = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k + 1, j, i);
    const Real zipt = z(n_src, k + 2, j, i);

    zl_(n_tar, i) = rec1d_p_ceno3(zimt, zimo, zi, zipo, zipt);
    zr_(n_tar, i) = rec1d_p_ceno3(zipt, zipo, zi, zimo, zimt);
  }
}

// impl -----------------------------------------------------------------------
namespace
{

Real ceno3lim(Real d[3])
{
  Real o3term = 0.0;
  Real absd[3];
  int kmin;

  if (((d[0] >= 0.) && (d[1] >= 0.) && (d[2] >= 0.)) ||
      ((d[0] < 0.) && (d[1] < 0.) && (d[2] < 0.)))
  {
    absd[0] = std::abs(d[0]);
    absd[1] = std::abs(alpha * d[1]);
    absd[2] = std::abs(d[2]);

    kmin = 0;
    if (absd[1] < absd[kmin])
      kmin = 1;
    if (absd[2] < absd[kmin])
      kmin = 2;

    o3term = d[kmin];
  }

  return (o3term);
}

#pragma omp declare simd
Real rec1d_p_ceno3(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt)
{
  /*
  // Computes u[i + 1/2]
  Real uipt = u[i+2];
  Real uipo = u[i+1];
  Real ui   = u[i];
  Real uimo = u[i-1];
  Real uimt = u[i-2];
  */
  using namespace reconstruction::utils;

  static const Real oocc2 = 1.0 / 2.0;
  static const Real oocc8 = 1.0 / 8.0;

  static const Real cc2  = 2.0;
  static const Real cc3  = 3.0;
  static const Real cc6  = 6.0;
  static const Real cc10 = 10.0;
  static const Real cc15 = 15.0;

  const Real slope = oocc2 * MC2((ui - uimo), (uipo - ui));

  Real tmpL;
  Real tmpd[3];  // these are d^k_i with k = -1,0,1

  tmpL    = ui + slope;
  tmpd[0] = (cc3 * uimt - cc10 * uimo + cc15 * ui) * oocc8 - tmpL;
  tmpd[1] = (-uimo + cc6 * ui + cc3 * uipo) * oocc8 - tmpL;
  tmpd[2] = (cc3 * ui + cc6 * uipo - uipt) * oocc8 - tmpL;

  return tmpL + ceno3lim(tmpd);
}

}  // namespace
// ----------------------------------------------------------------------------

//
// :D
//

// ============================================================================
// CENO5 -- Central ENO 5th order reconstruction
//
// Same philosophy as CENO3 but uses three quartic (5-point) sub-stencils
// from a 7-point stencil {i-3,...,i+3}.  Achieves 5th order in smooth
// regions; falls back to 2nd-order TVD near discontinuities.
//
// Sub-stencil coefficients (Lagrange interpolation at x_{i+1/2}),
// denominator 128:
//   k=-1  {i-3..i+1}:  -5,  28, -70, 140, 35
//   k= 0  {i-2..i+2}:   3, -20,  90,  60, -5
//   k=+1  {i-1..i+3}:  -5,  60,  90, -20,  3
//
// Requires NGHOST >= 4 (same as MP7).
// ============================================================================

// ----------------------------------------------------------------------------
namespace
{

// CENO5 uses the same limiter structure as CENO3 (3 candidates, same-sign
// check, alpha-weighted central bias, select minimum magnitude).  Reuse
// ceno3lim directly -- it already does exactly what we need.

// Paired L+R kernel: computes both left and right states in a single call,
// sharing the MC2 slope computation.
inline void rec1d_p_ceno5_LR(const Real uim3,
                             const Real uim2,
                             const Real uim1,
                             const Real ui,
                             const Real uip1,
                             const Real uip2,
                             const Real uip3,
                             Real& uL,
                             Real& uR)
{
  using namespace reconstruction::utils;

  static constexpr Real oo2   = 1.0 / 2.0;
  static constexpr Real oo128 = 1.0 / 128.0;

  // MC2 slope -- shared between L and R (R just flips sign)
  const Real slope = oo2 * MC2((ui - uim1), (uip1 - ui));

  // --- Left state (forward stencil) ---
  const Real baseL = ui + slope;
  Real dL[3];
  dL[0] =
    (-5.0 * uim3 + 28.0 * uim2 - 70.0 * uim1 + 140.0 * ui + 35.0 * uip1) *
      oo128 -
    baseL;
  dL[1] =
    (3.0 * uim2 - 20.0 * uim1 + 90.0 * ui + 60.0 * uip1 - 5.0 * uip2) * oo128 -
    baseL;
  dL[2] = (-5.0 * uim1 + 60.0 * ui + 90.0 * uip1 - 20.0 * uip2 + 3.0 * uip3) *
            oo128 -
          baseL;
  uL = baseL + ceno3lim(dL);

  // --- Right state (reversed stencil) ---
  // Under argument reversal: uim3<->uip3, uim2<->uip2, uim1<->uip1
  // slope flips sign -> baseR = ui - slope
  const Real baseR = ui - slope;
  Real dR[3];
  dR[0] =
    (-5.0 * uip3 + 28.0 * uip2 - 70.0 * uip1 + 140.0 * ui + 35.0 * uim1) *
      oo128 -
    baseR;
  dR[1] =
    (3.0 * uip2 - 20.0 * uip1 + 90.0 * ui + 60.0 * uim1 - 5.0 * uim2) * oo128 -
    baseR;
  dR[2] = (-5.0 * uip1 + 60.0 * ui + 90.0 * uim1 - 20.0 * uim2 + 3.0 * uim3) *
            oo128 -
          baseR;
  uR = baseR + ceno3lim(dR);
}

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void Reconstruction::ReconstructCeno5X1(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim3 = z(n_src, k, j, i - 3);
    const Real zim2 = z(n_src, k, j, i - 2);
    const Real zim1 = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j, i + 1);
    const Real zip2 = z(n_src, k, j, i + 2);
    const Real zip3 = z(n_src, k, j, i + 3);

    Real uL, uR;
    rec1d_p_ceno5_LR(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructCeno5X2(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim3 = z(n_src, k, j - 3, i);
    const Real zim2 = z(n_src, k, j - 2, i);
    const Real zim1 = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k, j + 1, i);
    const Real zip2 = z(n_src, k, j + 2, i);
    const Real zip3 = z(n_src, k, j + 3, i);

    Real uL, uR;
    rec1d_p_ceno5_LR(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructCeno5X3(AthenaArray<Real>& z,
                                        AthenaArray<Real>& zl_,
                                        AthenaArray<Real>& zr_,
                                        const int n_tar,
                                        const int n_src,
                                        const int k,
                                        const int j,
                                        const int il,
                                        const int iu)
{
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zim3 = z(n_src, k - 3, j, i);
    const Real zim2 = z(n_src, k - 2, j, i);
    const Real zim1 = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zip1 = z(n_src, k + 1, j, i);
    const Real zip2 = z(n_src, k + 2, j, i);
    const Real zip3 = z(n_src, k + 3, j, i);

    Real uL, uR;
    rec1d_p_ceno5_LR(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------

//
// :D
//
