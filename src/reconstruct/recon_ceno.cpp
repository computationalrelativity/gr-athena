// C/C++ headers

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
namespace
{

static constexpr Real alpha = 0.7;  // CENO3 bias

#pragma omp declare simd
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

// ---------------------------------------------------------------------------
// CENO3 -- Central ENO 3rd order reconstruction
//
// Paired L+R kernel: computes both left and right states in a single call,
// sharing the MC2 slope computation.
//
// pw = false : FV cell-average stencil coefficients (denom 6)
// pw = true  : PW pointwise stencil coefficients (denom 8)
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_ceno3_LR(const Real uimt,
                              const Real uimo,
                              const Real ui,
                              const Real uipo,
                              const Real uipt,
                              Real& uL,
                              Real& uR)
{
  using namespace reconstruction::utils;

  const Real slope = 0.5 * MC2((ui - uimo), (uipo - ui));

  const Real baseL = ui + slope;
  Real dL[3];
  if constexpr (pw)
  {
    constexpr Real oo8 = 1.0 / 8.0;
    dL[0] = ( 3.0 * uimt - 10.0 * uimo + 15.0 * ui) * oo8 - baseL;
    dL[1] = (           -uimo +  6.0 * ui +  3.0 * uipo) * oo8 - baseL;
    dL[2] = (           3.0 * ui +  6.0 * uipo -       uipt) * oo8 - baseL;
  }
  else
  {
    constexpr Real oo6 = 1.0 / 6.0;
    dL[0] = ( 2.0 * uimt -  7.0 * uimo + 11.0 * ui) * oo6 - baseL;
    dL[1] = (           -uimo +  5.0 * ui +  2.0 * uipo) * oo6 - baseL;
    dL[2] = (           2.0 * ui +  5.0 * uipo -       uipt) * oo6 - baseL;
  }
  uL = baseL + ceno3lim(dL);

  const Real baseR = ui - slope;
  Real dR[3];
  if constexpr (pw)
  {
    constexpr Real oo8 = 1.0 / 8.0;
    dR[0] = ( 3.0 * uipt - 10.0 * uipo + 15.0 * ui) * oo8 - baseR;
    dR[1] = (           -uipo +  6.0 * ui +  3.0 * uimo) * oo8 - baseR;
    dR[2] = (           3.0 * ui +  6.0 * uimo -       uimt) * oo8 - baseR;
  }
  else
  {
    constexpr Real oo6 = 1.0 / 6.0;
    dR[0] = ( 2.0 * uipt -  7.0 * uipo + 11.0 * ui) * oo6 - baseR;
    dR[1] = (           -uipo +  5.0 * ui +  2.0 * uimo) * oo6 - baseR;
    dR[2] = (           2.0 * ui +  5.0 * uimo -       uimt) * oo6 - baseR;
  }
  uR = baseR + ceno3lim(dR);
}

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
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k, j, i - 2);
    const Real zimo = z(n_src, k, j, i - 1);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j, i + 1);
    const Real zipt = z(n_src, k, j, i + 2);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_ceno3_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_ceno3_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
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
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k, j - 2, i);
    const Real zimo = z(n_src, k, j - 1, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k, j + 1, i);
    const Real zipt = z(n_src, k, j + 2, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_ceno3_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_ceno3_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
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
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i = il; i <= iu; ++i)
  {
    const Real zimt = z(n_src, k - 2, j, i);
    const Real zimo = z(n_src, k - 1, j, i);
    const Real zi   = z(n_src, k, j, i);
    const Real zipo = z(n_src, k + 1, j, i);
    const Real zipt = z(n_src, k + 2, j, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_ceno3_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_ceno3_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

//
// :D
//

// ============================================================================
// CENO5 -- Central ENO 5th order reconstruction
//
// Three quartic (5-point) sub-stencils from a 7-point stencil {i-3,...,i+3}.
// Achieves 5th order in smooth regions; falls back to 2nd-order TVD near
// discontinuities.
//
// pw = false : FV cell-average stencil coefficients (denom 60)
// pw = true  : PW pointwise stencil coefficients (denom 128)
//
// Requires NGHOST >= 4.
// ============================================================================

// ----------------------------------------------------------------------------
namespace
{

// Paired L+R kernel: computes both left and right states in a single call,
// sharing the MC2 slope computation.
#pragma omp declare simd
template <bool pw>
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

  const Real slope = 0.5 * MC2((ui - uim1), (uip1 - ui));

  // --- Left state (forward stencil) ---
  const Real baseL = ui + slope;
  Real dL[3];
  if constexpr (pw)
  {
    constexpr Real oo128 = 1.0 / 128.0;
    dL[0] = (-5.0*uim3 +  28.0*uim2 -  70.0*uim1 + 140.0*ui +  35.0*uip1) * oo128 - baseL;
    dL[1] = ( 3.0*uim2 -  20.0*uim1 +  90.0*ui   +  60.0*uip1 -  5.0*uip2) * oo128 - baseL;
    dL[2] = (-5.0*uim1 +  60.0*ui   +  90.0*uip1 -  20.0*uip2 +  3.0*uip3) * oo128 - baseL;
  }
  else
  {
    constexpr Real oo60 = 1.0 / 60.0;
    dL[0] = (-3.0*uim3 +  17.0*uim2 -  43.0*uim1 +  77.0*ui +  12.0*uip1) * oo60 - baseL;
    dL[1] = ( 2.0*uim2 -  13.0*uim1 +  47.0*ui   +  27.0*uip1 -  3.0*uip2) * oo60 - baseL;
    dL[2] = (-3.0*uim1 +  27.0*ui   +  47.0*uip1 -  13.0*uip2 +  2.0*uip3) * oo60 - baseL;
  }
  uL = baseL + ceno3lim(dL);

  // --- Right state (reversed stencil) ---
  const Real baseR = ui - slope;
  Real dR[3];
  if constexpr (pw)
  {
    constexpr Real oo128 = 1.0 / 128.0;
    dR[0] = (-5.0*uip3 +  28.0*uip2 -  70.0*uip1 + 140.0*ui +  35.0*uim1) * oo128 - baseR;
    dR[1] = ( 3.0*uip2 -  20.0*uip1 +  90.0*ui   +  60.0*uim1 -  5.0*uim2) * oo128 - baseR;
    dR[2] = (-5.0*uip1 +  60.0*ui   +  90.0*uim1 -  20.0*uim2 +  3.0*uim3) * oo128 - baseR;
  }
  else
  {
    constexpr Real oo60 = 1.0 / 60.0;
    dR[0] = (-3.0*uip3 +  17.0*uip2 -  43.0*uip1 +  77.0*ui +  12.0*uim1) * oo60 - baseR;
    dR[1] = ( 2.0*uip2 -  13.0*uip1 +  47.0*ui   +  27.0*uim1 -  3.0*uim2) * oo60 - baseR;
    dR[2] = (-3.0*uip1 +  27.0*ui   +  47.0*uim1 -  13.0*uim2 +  2.0*uim3) * oo60 - baseR;
  }
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
    if (xorder_pointwise)
      rec1d_p_ceno5_LR<true>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    else
      rec1d_p_ceno5_LR<false>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
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
    if (xorder_pointwise)
      rec1d_p_ceno5_LR<true>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    else
      rec1d_p_ceno5_LR<false>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
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
    if (xorder_pointwise)
      rec1d_p_ceno5_LR<true>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    else
      rec1d_p_ceno5_LR<false>(zim3, zim2, zim1, zi, zip1, zip2, zip3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------

//
// :D
//
