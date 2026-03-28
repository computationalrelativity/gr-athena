// C/C++ headers
#include <cmath>

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

// ---------------------------------------------------------------------------
// File-scope constexpr constants (Change 1 & 2)
// ---------------------------------------------------------------------------

// Smoothness indicator coefficients (Jiang & Shu '96)
static constexpr Real kThreeHalves = 3.0 / 2.0;
static constexpr Real kOneQuarter  = 1.0 / 4.0;

// Stencil polynomial coefficients
static constexpr Real kOneSixth = 1.0 / 6.0;

// WENO5 optimal weights & epsilon
static constexpr Real optimw[3] = { 1. / 10., 3. / 5., 3. / 10. };
static constexpr Real EPSL      = 1e-40;  // 1e-6

// See:
// A novel and robust scale-invariant WENO scheme for hyperbolic conservation
// laws; 2022, Don et. al.
//
// D-SI parameters (int types for p,s enable constexpr specialization)
static constexpr Real W5D_SI_EPSL = 1e-12;
static constexpr int W5D_SI_p     = 2;
static constexpr int W5D_SI_s     = 1;
static constexpr Real W5D_SI_mu_0 = 1e-40;

// Constexpr integer power -- resolves at compile time.
// For W5D_SI_p=2: ipow<2>(x) -> x*x  (eliminates std::pow)
// For W5D_SI_s=1: ipow<1>(x) -> x    (eliminates std::pow)
template <int N>
inline Real ipow(Real x)
{
  static_assert(N >= 0, "ipow requires non-negative exponent");
  if constexpr (N == 0)
    return 1.0;
  else if constexpr (N == 1)
    return x;
  else if constexpr (N == 2)
    return x * x;
  else
    return x * ipow<N - 1>(x);
}

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_JS_smoothness(Real& b_0,
                                  Real& b_1,
                                  Real& b_2,
                                  const Real uimt,
                                  const Real uimo,
                                  const Real ui,
                                  const Real uipo,
                                  const Real uipt);

#pragma omp declare simd
inline void rec1d_p_weno5stencils(Real& u_0,
                                  Real& u_1,
                                  Real& u_2,
                                  const Real uimt,
                                  const Real uimo,
                                  const Real ui,
                                  const Real uipo,
                                  const Real uipt);

// Single-directional functions (kept for reference / standalone use)
#pragma omp declare simd
Real rec1d_p_weno5(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt);

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt);

#pragma omp declare simd
Real rec1d_p_weno5d_si(const Real uimt,
                       const Real uimo,
                       const Real ui,
                       const Real uipo,
                       const Real uipt);

// Paired L+R functions (optimized: single beta computation per cell)
#pragma omp declare simd
inline void rec1d_p_weno5_LR(const Real uimt,
                             const Real uimo,
                             const Real ui,
                             const Real uipo,
                             const Real uipt,
                             Real& uL,
                             Real& uR);

#pragma omp declare simd
inline void rec1d_p_weno5z_LR(const Real uimt,
                              const Real uimo,
                              const Real ui,
                              const Real uipo,
                              const Real uipt,
                              Real& uL,
                              Real& uR);

#pragma omp declare simd
inline void rec1d_p_weno5d_si_LR(const Real uimt,
                                 const Real uimo,
                                 const Real ui,
                                 const Real uipo,
                                 const Real uipt,
                                 Real& uL,
                                 Real& uR);

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// WENO5-JS (Jiang-Shu)
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5X1(AthenaArray<Real>& z,
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
    rec1d_p_weno5_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno5X2(AthenaArray<Real>& z,
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
    rec1d_p_weno5_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno5X3(AthenaArray<Real>& z,
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
    rec1d_p_weno5_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------
// WENO5-Z (Borges et al.)
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5ZX1(AthenaArray<Real>& z,
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
    rec1d_p_weno5z_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno5ZX2(AthenaArray<Real>& z,
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
    rec1d_p_weno5z_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno5ZX3(AthenaArray<Real>& z,
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
    rec1d_p_weno5z_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------
// WENO5-D-SI (Don et al. 2022, scale-invariant)
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno5dsiX1(AthenaArray<Real>& z,
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
    rec1d_p_weno5d_si_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno5dsiX2(AthenaArray<Real>& z,
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
    rec1d_p_weno5d_si_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno5dsiX3(AthenaArray<Real>& z,
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
    rec1d_p_weno5d_si_LR(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// impl -----------------------------------------------------------------------
namespace
{

// ---------------------------------------------------------------------------
// Shared building blocks
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_JS_smoothness(Real& b_0,
                                  Real& b_1,
                                  Real& b_2,
                                  const Real uimt,
                                  const Real uimo,
                                  const Real ui,
                                  const Real uipo,
                                  const Real uipt)
{
  // Smoothness coefficients, Jiang & Shu '96
  b_0 = kThreeHalves * SQR((uimt - 2.0 * uimo + ui)) +
        kOneQuarter * SQR((uimt - 4.0 * uimo + 3.0 * ui));
  b_1 = kThreeHalves * SQR((uimo - 2.0 * ui + uipo)) +
        kOneQuarter * SQR((uimo - uipo));
  b_2 = kThreeHalves * SQR((ui - 2.0 * uipo + uipt)) +
        kOneQuarter * SQR((3.0 * ui - 4.0 * uipo + uipt));
}

#pragma omp declare simd
inline void rec1d_p_weno5stencils(Real& u_0,
                                  Real& u_1,
                                  Real& u_2,
                                  const Real uimt,
                                  const Real uimo,
                                  const Real ui,
                                  const Real uipo,
                                  const Real uipt)
{
  u_0 = kOneSixth * (2.0 * uimt - 7.0 * uimo + 11.0 * ui);
  u_1 = kOneSixth * (-uimo + 5.0 * ui + 2.0 * uipo);
  u_2 = kOneSixth * (2.0 * ui + 5.0 * uipo - uipt);
}

// ---------------------------------------------------------------------------
// Single-directional functions (kept for reference / standalone use)
// ---------------------------------------------------------------------------

#pragma omp declare simd
Real rec1d_p_weno5(const Real uimt,
                   const Real uimo,
                   const Real ui,
                   const Real uipo,
                   const Real uipt)
{
  /*
  // Computes u[i + 1/2]
  Real uimt = u [i-2];
  Real uimo = u [i-1];
  Real ui   = u [i];
  Real uipo = u [i+1];
  Real uipt = u [i+2];
  */
  Real uk[3], b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real a_0 = optimw[0] / SQR((EPSL + b[0]));
  const Real a_1 = optimw[1] / SQR((EPSL + b[1]));
  const Real a_2 = optimw[2] / SQR((EPSL + b[2]));

  const Real dsa = 1.0 / (a_0 + a_1 + a_2);

  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);
  return dsa * (a_0 * uk[0] + a_1 * uk[1] + a_2 * uk[2]);
}

#pragma omp declare simd
Real rec1d_p_weno5z(const Real uimt,
                    const Real uimo,
                    const Real ui,
                    const Real uipo,
                    const Real uipt)
{
  /*
  // Computes u[i + 1/2]
  Real uimt = u [i-2];
  Real uimo = u [i-1];
  Real ui   = u [i];
  Real uipo = u [i+1];
  Real uipt = u [i+2];
  */
  Real uk[3], b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real db = std::abs(b[0] - b[2]);

  const Real a_0 = optimw[0] * (1.0 + db / (EPSL + b[0]));
  const Real a_1 = optimw[1] * (1.0 + db / (EPSL + b[1]));
  const Real a_2 = optimw[2] * (1.0 + db / (EPSL + b[2]));

  const Real dsa = 1.0 / (a_0 + a_1 + a_2);

  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);
  return dsa * (a_0 * uk[0] + a_1 * uk[1] + a_2 * uk[2]);
}

#pragma omp declare simd
Real rec1d_p_weno5d_si(const Real uimt,
                       const Real uimo,
                       const Real ui,
                       const Real uipo,
                       const Real uipt)
{
  /*
    // Computes u[i + 1/2]
    Real uimt = u [i-2];
    Real uimo = u [i-1];
    Real ui   = u [i];
    Real uipo = u [i+1];
    Real uipt = u [i+2];
  */
  Real uk[3], a[3], b[3];

  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  const Real phi = std::sqrt(std::abs(b[0] - 2. * b[1] + b[2]));
  const Real tau = std::abs(b[0] - b[2]);

  // local descaling function
  const int r   = 3;
  const Real xi = (std::abs(uimt) + std::abs(uimo) + std::abs(ui) +
                   std::abs(uipo) + std::abs(uipt)) /
                  (2. * (r)-1.);

  const Real mu  = xi + W5D_SI_mu_0;
  const Real mu2 = SQR(mu);
  const Real Phi = std::min(1., phi / mu);

  const Real eps_mu2 = W5D_SI_EPSL * mu2;

  // Unrolled (was for j=0..2); uses ipow to eliminate std::pow
  {
    const Real Z_0   = ipow<W5D_SI_p>(tau / (b[0] + eps_mu2));
    const Real Gam_0 = Phi * Z_0;
    a[0]             = optimw[0] * ipow<W5D_SI_s>(1. + Gam_0);
  }
  {
    const Real Z_1   = ipow<W5D_SI_p>(tau / (b[1] + eps_mu2));
    const Real Gam_1 = Phi * Z_1;
    a[1]             = optimw[1] * ipow<W5D_SI_s>(1. + Gam_1);
  }
  {
    const Real Z_2   = ipow<W5D_SI_p>(tau / (b[2] + eps_mu2));
    const Real Gam_2 = Phi * Z_2;
    a[2]             = optimw[2] * ipow<W5D_SI_s>(1. + Gam_2);
  }

  const Real dsa = 1.0 / (a[0] + a[1] + a[2]);
  rec1d_p_weno5stencils(uk[0], uk[1], uk[2], uimt, uimo, ui, uipo, uipt);

  return dsa * (a[0] * uk[0] + a[1] * uk[1] + a[2] * uk[2]);
}

// ---------------------------------------------------------------------------
// Paired L+R functions (optimized: single beta computation per cell)
//
// Exploits the Jiang-Shu smoothness indicator symmetry under argument
// reversal:
//   b_0(uipt,uipo,ui,uimo,uimt) = b_2(uimt,uimo,ui,uipo,uipt)
//   b_1(uipt,uipo,ui,uimo,uimt) = b_1(uimt,uimo,ui,uipo,uipt)
//   b_2(uipt,uipo,ui,uimo,uimt) = b_0(uimt,uimo,ui,uipo,uipt)
//
// So we compute betas once (forward) and derive L weights from (b0,b1,b2),
// R weights from (b2,b1,b0). Stencil polynomials do NOT share this symmetry
// and are computed separately for L and R.
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_weno5_LR(const Real uimt,
                             const Real uimo,
                             const Real ui,
                             const Real uipo,
                             const Real uipt,
                             Real& uL,
                             Real& uR)
{
  Real b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  // --- Left state: weights from (b[0], b[1], b[2]) ---
  const Real aL_0 = optimw[0] / SQR((EPSL + b[0]));
  const Real aL_1 = optimw[1] / SQR((EPSL + b[1]));
  const Real aL_2 = optimw[2] / SQR((EPSL + b[2]));
  const Real dsaL = 1.0 / (aL_0 + aL_1 + aL_2);

  Real ukL[3];
  rec1d_p_weno5stencils(ukL[0], ukL[1], ukL[2], uimt, uimo, ui, uipo, uipt);
  uL = dsaL * (aL_0 * ukL[0] + aL_1 * ukL[1] + aL_2 * ukL[2]);

  // --- Right state: weights from (b[2], b[1], b[0]) (symmetry) ---
  const Real aR_0 = optimw[0] / SQR((EPSL + b[2]));
  const Real aR_1 = optimw[1] / SQR((EPSL + b[1]));
  const Real aR_2 = optimw[2] / SQR((EPSL + b[0]));
  const Real dsaR = 1.0 / (aR_0 + aR_1 + aR_2);

  Real ukR[3];
  rec1d_p_weno5stencils(ukR[0], ukR[1], ukR[2], uipt, uipo, ui, uimo, uimt);
  uR = dsaR * (aR_0 * ukR[0] + aR_1 * ukR[1] + aR_2 * ukR[2]);
}

#pragma omp declare simd
inline void rec1d_p_weno5z_LR(const Real uimt,
                              const Real uimo,
                              const Real ui,
                              const Real uipo,
                              const Real uipt,
                              Real& uL,
                              Real& uR)
{
  Real b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  // tau = |b0 - b2| is invariant under the L/R swap
  const Real db = std::abs(b[0] - b[2]);

  // --- Left state: weights from (b[0], b[1], b[2]) ---
  const Real aL_0 = optimw[0] * (1.0 + db / (EPSL + b[0]));
  const Real aL_1 = optimw[1] * (1.0 + db / (EPSL + b[1]));
  const Real aL_2 = optimw[2] * (1.0 + db / (EPSL + b[2]));
  const Real dsaL = 1.0 / (aL_0 + aL_1 + aL_2);

  Real ukL[3];
  rec1d_p_weno5stencils(ukL[0], ukL[1], ukL[2], uimt, uimo, ui, uipo, uipt);
  uL = dsaL * (aL_0 * ukL[0] + aL_1 * ukL[1] + aL_2 * ukL[2]);

  // --- Right state: weights from (b[2], b[1], b[0]) (symmetry) ---
  const Real aR_0 = optimw[0] * (1.0 + db / (EPSL + b[2]));
  const Real aR_1 = optimw[1] * (1.0 + db / (EPSL + b[1]));
  const Real aR_2 = optimw[2] * (1.0 + db / (EPSL + b[0]));
  const Real dsaR = 1.0 / (aR_0 + aR_1 + aR_2);

  Real ukR[3];
  rec1d_p_weno5stencils(ukR[0], ukR[1], ukR[2], uipt, uipo, ui, uimo, uimt);
  uR = dsaR * (aR_0 * ukR[0] + aR_1 * ukR[1] + aR_2 * ukR[2]);
}

#pragma omp declare simd
inline void rec1d_p_weno5d_si_LR(const Real uimt,
                                 const Real uimo,
                                 const Real ui,
                                 const Real uipo,
                                 const Real uipt,
                                 Real& uL,
                                 Real& uR)
{
  Real b[3];
  rec1d_p_JS_smoothness(b[0], b[1], b[2], uimt, uimo, ui, uipo, uipt);

  // All of the following are invariant under the L/R argument swap:
  //   phi = sqrt(|b0 - 2*b1 + b2|)  -- symmetric in b0,b2
  //   tau = |b0 - b2|               -- symmetric
  //   xi  = (sum |u_k|) / 5         -- symmetric (same stencil values)
  //   mu, mu2, Phi, eps_mu2          -- derived from the above
  const Real phi = std::sqrt(std::abs(b[0] - 2. * b[1] + b[2]));
  const Real tau = std::abs(b[0] - b[2]);

  // local descaling function
  const int r   = 3;
  const Real xi = (std::abs(uimt) + std::abs(uimo) + std::abs(ui) +
                   std::abs(uipo) + std::abs(uipt)) /
                  (2. * (r)-1.);

  const Real mu  = xi + W5D_SI_mu_0;
  const Real mu2 = SQR(mu);
  const Real Phi = std::min(1., phi / mu);

  const Real eps_mu2 = W5D_SI_EPSL * mu2;

  // --- Left state: weights from (b[0], b[1], b[2]) ---
  // Unrolled, with ipow to eliminate std::pow
  const Real ZL_0 = ipow<W5D_SI_p>(tau / (b[0] + eps_mu2));
  const Real ZL_1 = ipow<W5D_SI_p>(tau / (b[1] + eps_mu2));
  const Real ZL_2 = ipow<W5D_SI_p>(tau / (b[2] + eps_mu2));
  const Real aL_0 = optimw[0] * ipow<W5D_SI_s>(1. + Phi * ZL_0);
  const Real aL_1 = optimw[1] * ipow<W5D_SI_s>(1. + Phi * ZL_1);
  const Real aL_2 = optimw[2] * ipow<W5D_SI_s>(1. + Phi * ZL_2);
  const Real dsaL = 1.0 / (aL_0 + aL_1 + aL_2);

  Real ukL[3];
  rec1d_p_weno5stencils(ukL[0], ukL[1], ukL[2], uimt, uimo, ui, uipo, uipt);
  uL = dsaL * (aL_0 * ukL[0] + aL_1 * ukL[1] + aL_2 * ukL[2]);

  // --- Right state: weights from (b[2], b[1], b[0]) (symmetry) ---
  // Z values reuse those already computed (just swapped indices)
  const Real aR_0 = optimw[0] * ipow<W5D_SI_s>(1. + Phi * ZL_2);
  const Real aR_1 = optimw[1] * ipow<W5D_SI_s>(1. + Phi * ZL_1);
  const Real aR_2 = optimw[2] * ipow<W5D_SI_s>(1. + Phi * ZL_0);
  const Real dsaR = 1.0 / (aR_0 + aR_1 + aR_2);

  Real ukR[3];
  rec1d_p_weno5stencils(ukR[0], ukR[1], ukR[2], uipt, uipo, ui, uimo, uimt);
  uR = dsaR * (aR_0 * ukR[0] + aR_1 * ukR[1] + aR_2 * ukR[2]);
}

}  // namespace
// ----------------------------------------------------------------------------

//
// :D
//
