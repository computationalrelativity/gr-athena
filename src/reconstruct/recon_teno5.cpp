// C/C++ headers
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "recon_koren.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
namespace
{

// Paper: Takagi, Fu, Wakimura, Xiao (2022), J. Comput. Phys. 452, 110899
//
// TENO5 smoothness indicators (Eq. 10):
//   B0(central)  = 1/4*(u_{i-1}-u_{i+1})^2 + 13/12*(u_{i-1}-2*u_i+u_{i+1})^2
//   B1(forward)  = 1/4*(3*u_i-4*u_{i+1}+u_{i+2})^2 +
//   13/12*(u_i-2*u_{i+1}+u_{i+2})^2 B2(backward) =
//   1/4*(u_{i-2}-4*u_{i-1}+3*u_i)^2 + 13/12*(u_{i-2}-2*u_{i-1}+u_i)^2
//
// Candidate stencils (Eq. 7):
//   u0 = 1/6*(-u_{i-1} + 5*u_i + 2*u_{i+1})      [central, d0=6/10]
//   u1 = 1/6*( 2*u_i + 5*u_{i+1} - u_{i+2})      [forward, d1=3/10]
//   u2 = 1/6*( 2*u_{i-2} - 7*u_{i-1} + 11*u_i)   [backward, d2=1/10]
//
// TENO weighting:
//   tau = |B0 - B1| + |B0 - B2|             N.B. this differs from ref.
//   gamma_k = (1 + tau/(B_k + eps))^6       (Eq. 8, q=6)
//   chi_k = gamma_k / sum(gamma_k)          (Eq. 11)
//   delta_k = 1 if chi_k >= C_T else 0      (Eq. 12, C_T=1e-5)
//   w_k = d_k*delta_k / sum(d_j*delta_j)    (Eq. 13)

// ---------------------------------------------------------------------------
// File-scope constants
// ---------------------------------------------------------------------------

static constexpr Real kOneSixth   = 1.0 / 6.0;
static constexpr Real kOneQuarter = 1.0 / 4.0;
static constexpr Real k13Over12   = 13.0 / 12.0;

static constexpr Real dteno[3]    = { 6.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0 };
static constexpr Real dteno_pw[3] = { 5.0 / 8.0, 5.0 / 16.0, 1.0 / 16.0 };
static constexpr Real EPSL        = 1e-40;
static constexpr Real C_T      = 1e-5;
static constexpr Real q_teno   = 6.0;

// ---------------------------------------------------------------------------
// Paper smoothness indicators
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline Real teno_B0(const Real im1, const Real i, const Real ip1)
{
  return kOneQuarter * SQR(im1 - ip1) + k13Over12 * SQR(im1 - 2.0 * i + ip1);
}

#pragma omp declare simd
inline Real teno_B1(const Real i, const Real ip1, const Real ip2)
{
  return kOneQuarter * SQR(3.0 * i - 4.0 * ip1 + ip2) +
         k13Over12 * SQR(i - 2.0 * ip1 + ip2);
}

#pragma omp declare simd
inline Real teno_B2(const Real im2, const Real im1, const Real i)
{
  return kOneQuarter * SQR(im2 - 4.0 * im1 + 3.0 * i) +
         k13Over12 * SQR(im2 - 2.0 * im1 + i);
}

// ---------------------------------------------------------------------------
// TENO5 binary cutoff  (Eqs. 8, 11, 12)
//
// Input:  b0,b1,b2  -- smoothness indicators in paper order
// Output: d0,d1,d2  -- binary stencil flags (0.0 or 1.0)
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void teno5_cutoff(const Real b0,
                         const Real b1,
                         const Real b2,
                         Real& d0,
                         Real& d1,
                         Real& d2)
{
  const Real tau = std::abs(b0 - b1) + std::abs(b0 - b2);
  const Real x0  = 1.0 + tau / (b0 + EPSL);
  const Real x1  = 1.0 + tau / (b1 + EPSL);
  const Real x2  = 1.0 + tau / (b2 + EPSL);
  // x^6 = (x^2)^3
  const Real g0_2      = x0 * x0;
  const Real g1_2      = x1 * x1;
  const Real g2_2      = x2 * x2;
  const Real g0        = g0_2 * g0_2 * g0_2;
  const Real g1        = g1_2 * g1_2 * g1_2;
  const Real g2        = g2_2 * g2_2 * g2_2;
  const Real inv_sum_g = 1.0 / (g0 + g1 + g2);
  const Real chi0      = g0 * inv_sum_g;
  const Real chi1      = g1 * inv_sum_g;
  const Real chi2      = g2 * inv_sum_g;
  d0                   = (chi0 >= C_T) ? 1.0 : 0.0;
  d1                   = (chi1 >= C_T) ? 1.0 : 0.0;
  d2                   = (chi2 >= C_T) ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// MC2 TVD fallback function (used by TENO5 templates below)
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_mc2_LR(const Real a,
                         const Real b,
                         const Real c,
                         Real& uL,
                         Real& uR)
{
  const Real dl = c - b;
  const Real dr = b - a;
  if (dl * dr <= 0.0)
  {
    uL = b;
    uR = b;
    return;
  }
  const Real sgn = (dl > 0.0) ? 1.0 : -1.0;
  const Real adl = std::fabs(dl);
  const Real adr = std::fabs(dr);
  const Real adc = 0.5 * std::fabs(c - a);
  const Real du  = sgn * std::fmin(2.0 * adl, std::fmin(2.0 * adr, adc));
  uL             = b + 0.5 * du;
  uR             = b - 0.5 * du;
}

// ---------------------------------------------------------------------------
// Paired L+R reconstruction
//
// B0 is symmetric: B0(a,b,c) = B0(c,b,a): computed once, shared by L and R.
// B1 and B2 are not symmetric and must be recomputed for the reversed stencil.
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_teno5_LR(const Real uimt,
                             const Real uimo,
                             const Real ui,
                             const Real uipo,
                             const Real uipt,
                             Real& uL,
                             Real& uR)
{
  const auto& dt = pw ? dteno_pw : dteno;

  const Real b0_L = teno_B0(uimo, ui, uipo);
  const Real b1_L = teno_B1(ui, uipo, uipt);
  const Real b2_L = teno_B2(uimt, uimo, ui);

  Real dL[3];
  teno5_cutoff(b0_L, b1_L, b2_L, dL[0], dL[1], dL[2]);

  if (dL[0] == 0.0 && dL[1] == 0.0 && dL[2] == 0.0)
  {
    rec1d_mc2_LR(uimo, ui, uipo, uL, uR);
    return;
  }

  const Real denom_L = dt[0] * dL[0] + dt[1] * dL[1] + dt[2] * dL[2];
  const Real inv_denom_L = 1.0 / std::max(denom_L, EPSL);

  Real ukL0, ukL1, ukL2;
  if constexpr (pw) {
    ukL0 = (-uimo + 6.0 * ui + 3.0 * uipo) * (1.0 / 8.0);
    ukL1 = (3.0 * ui + 6.0 * uipo - uipt) * (1.0 / 8.0);
    ukL2 = (3.0 * uimt - 10.0 * uimo + 15.0 * ui) * (1.0 / 8.0);
  } else {
    ukL0 = kOneSixth * (-uimo + 5.0 * ui + 2.0 * uipo);
    ukL1 = kOneSixth * (2.0 * ui + 5.0 * uipo - uipt);
    ukL2 = kOneSixth * (2.0 * uimt - 7.0 * uimo + 11.0 * ui);
  }
  uL = inv_denom_L *
       (dt[0] * dL[0] * ukL0 + dt[1] * dL[1] * ukL1 + dt[2] * dL[2] * ukL2);

  const Real b1_R = teno_B1(ui, uimo, uimt);
  const Real b2_R = teno_B2(uipt, uipo, ui);

  Real dR[3];
  teno5_cutoff(b0_L, b1_R, b2_R, dR[0], dR[1], dR[2]);

  if (dR[0] == 0.0 && dR[1] == 0.0 && dR[2] == 0.0)
  {
    Real uL_tmp;
    rec1d_mc2_LR(uimo, ui, uipo, uL_tmp, uR);
    return;
  }

  const Real denom_R = dt[0] * dR[0] + dt[1] * dR[1] + dt[2] * dR[2];
  const Real inv_denom_R = 1.0 / std::max(denom_R, EPSL);

  Real ukR0, ukR1, ukR2;
  if constexpr (pw) {
    ukR0 = (-uipo + 6.0 * ui + 3.0 * uimo) * (1.0 / 8.0);
    ukR1 = (3.0 * ui + 6.0 * uimo - uimt) * (1.0 / 8.0);
    ukR2 = (3.0 * uipt - 10.0 * uipo + 15.0 * ui) * (1.0 / 8.0);
  } else {
    ukR0 = kOneSixth * (-uipo + 5.0 * ui + 2.0 * uimo);
    ukR1 = kOneSixth * (2.0 * ui + 5.0 * uimo - uimt);
    ukR2 = kOneSixth * (2.0 * uipt - 7.0 * uipo + 11.0 * ui);
  }
  uR = inv_denom_R *
       (dt[0] * dR[0] * ukR0 + dt[1] * dR[1] * ukR1 + dt[2] * dR[2] * ukR2);
}

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_teno5_mc2_LR(const Real uimt,
                                 const Real uimo,
                                 const Real ui,
                                 const Real uipo,
                                 const Real uipt,
                                 Real& uL,
                                 Real& uR)
{
  const auto& dt = pw ? dteno_pw : dteno;

  const Real b0_L = teno_B0(uimo, ui, uipo);
  const Real b1_L = teno_B1(ui, uipo, uipt);
  const Real b2_L = teno_B2(uimt, uimo, ui);

  Real dL[3];
  teno5_cutoff(b0_L, b1_L, b2_L, dL[0], dL[1], dL[2]);

  if (dL[0] > 0.0 && dL[1] > 0.0 && dL[2] > 0.0)
  {
    const Real denom_L = dt[0] * dL[0] + dt[1] * dL[1] + dt[2] * dL[2];
    const Real inv_denom_L = 1.0 / std::max(denom_L, EPSL);

    Real ukL0, ukL1, ukL2;
    if constexpr (pw) {
      ukL0 = (-uimo + 6.0 * ui + 3.0 * uipo) * (1.0 / 8.0);
      ukL1 = (3.0 * ui + 6.0 * uipo - uipt) * (1.0 / 8.0);
      ukL2 = (3.0 * uimt - 10.0 * uimo + 15.0 * ui) * (1.0 / 8.0);
    } else {
      ukL0 = kOneSixth * (-uimo + 5.0 * ui + 2.0 * uipo);
      ukL1 = kOneSixth * (2.0 * ui + 5.0 * uipo - uipt);
      ukL2 = kOneSixth * (2.0 * uimt - 7.0 * uimo + 11.0 * ui);
    }
    uL = inv_denom_L *
         (dt[0] * dL[0] * ukL0 + dt[1] * dL[1] * ukL1 + dt[2] * dL[2] * ukL2);
  }
  else
  {
    rec1d_mc2_LR(uimo, ui, uipo, uL, uR);
  }

  const Real b1_R = teno_B1(ui, uimo, uimt);
  const Real b2_R = teno_B2(uipt, uipo, ui);

  Real dR[3];
  teno5_cutoff(b0_L, b1_R, b2_R, dR[0], dR[1], dR[2]);

  if (dR[0] > 0.0 && dR[1] > 0.0 && dR[2] > 0.0)
  {
    const Real denom_R = dt[0] * dR[0] + dt[1] * dR[1] + dt[2] * dR[2];
    const Real inv_denom_R = 1.0 / std::max(denom_R, EPSL);

    Real ukR0, ukR1, ukR2;
    if constexpr (pw) {
      ukR0 = (-uipo + 6.0 * ui + 3.0 * uimo) * (1.0 / 8.0);
      ukR1 = (3.0 * ui + 6.0 * uimo - uimt) * (1.0 / 8.0);
      ukR2 = (3.0 * uipt - 10.0 * uipo + 15.0 * ui) * (1.0 / 8.0);
    } else {
      ukR0 = kOneSixth * (-uipo + 5.0 * ui + 2.0 * uimo);
      ukR1 = kOneSixth * (2.0 * ui + 5.0 * uimo - uimt);
      ukR2 = kOneSixth * (2.0 * uipt - 7.0 * uipo + 11.0 * ui);
    }
    uR = inv_denom_R *
         (dt[0] * dR[0] * ukR0 + dt[1] * dR[1] * ukR1 + dt[2] * dR[2] * ukR2);
  }
  else
  {
    Real uR_dummy;
    rec1d_mc2_LR(uimo, ui, uipo, uR_dummy, uR);
  }
}

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_teno5_koren_LR(const Real uimt,
                                   const Real uimo,
                                   const Real ui,
                                   const Real uipo,
                                   const Real uipt,
                                   Real& uL,
                                   Real& uR)
{
  const auto& dt = pw ? dteno_pw : dteno;

  const Real b0_L = teno_B0(uimo, ui, uipo);
  const Real b1_L = teno_B1(ui, uipo, uipt);
  const Real b2_L = teno_B2(uimt, uimo, ui);

  Real dL[3];
  teno5_cutoff(b0_L, b1_L, b2_L, dL[0], dL[1], dL[2]);

  if (dL[0] > 0.0 && dL[1] > 0.0 && dL[2] > 0.0)
  {
    const Real denom_L = dt[0] * dL[0] + dt[1] * dL[1] + dt[2] * dL[2];
    const Real inv_denom_L = 1.0 / std::max(denom_L, EPSL);

    Real ukL0, ukL1, ukL2;
    if constexpr (pw) {
      ukL0 = (-uimo + 6.0 * ui + 3.0 * uipo) * (1.0 / 8.0);
      ukL1 = (3.0 * ui + 6.0 * uipo - uipt) * (1.0 / 8.0);
      ukL2 = (3.0 * uimt - 10.0 * uimo + 15.0 * ui) * (1.0 / 8.0);
    } else {
      ukL0 = kOneSixth * (-uimo + 5.0 * ui + 2.0 * uipo);
      ukL1 = kOneSixth * (2.0 * ui + 5.0 * uipo - uipt);
      ukL2 = kOneSixth * (2.0 * uimt - 7.0 * uimo + 11.0 * ui);
    }
    uL = inv_denom_L *
         (dt[0] * dL[0] * ukL0 + dt[1] * dL[1] * ukL1 + dt[2] * dL[2] * ukL2);
  }
  else
  {
    reconstruction::koren::ReconstructLR(uimo, ui, uipo, uL, uR);
  }

  const Real b1_R = teno_B1(ui, uimo, uimt);
  const Real b2_R = teno_B2(uipt, uipo, ui);

  Real dR[3];
  teno5_cutoff(b0_L, b1_R, b2_R, dR[0], dR[1], dR[2]);

  if (dR[0] > 0.0 && dR[1] > 0.0 && dR[2] > 0.0)
  {
    const Real denom_R = dt[0] * dR[0] + dt[1] * dR[1] + dt[2] * dR[2];
    const Real inv_denom_R = 1.0 / std::max(denom_R, EPSL);

    Real ukR0, ukR1, ukR2;
    if constexpr (pw) {
      ukR0 = (-uipo + 6.0 * ui + 3.0 * uimo) * (1.0 / 8.0);
      ukR1 = (3.0 * ui + 6.0 * uimo - uimt) * (1.0 / 8.0);
      ukR2 = (3.0 * uipt - 10.0 * uipo + 15.0 * ui) * (1.0 / 8.0);
    } else {
      ukR0 = kOneSixth * (-uipo + 5.0 * ui + 2.0 * uimo);
      ukR1 = kOneSixth * (2.0 * ui + 5.0 * uimo - uimt);
      ukR2 = kOneSixth * (2.0 * uipt - 7.0 * uipo + 11.0 * ui);
    }
    uR = inv_denom_R *
         (dt[0] * dR[0] * ukR0 + dt[1] * dR[1] * ukR1 + dt[2] * dR[2] * ukR2);
  }
  else
  {
    Real uR_dummy;
    reconstruction::koren::ReconstructLR(uimo, ui, uipo, uR_dummy, uR);
  }
}

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// X1-direction
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructTeno5X1(AthenaArray<Real>& z,
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
      rec1d_p_teno5_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructTeno5X2(AthenaArray<Real>& z,
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
      rec1d_p_teno5_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructTeno5X3(AthenaArray<Real>& z,
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
      rec1d_p_teno5_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructTeno5mc2X1(AthenaArray<Real>& z,
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
      rec1d_p_teno5_mc2_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_mc2_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructTeno5mc2X2(AthenaArray<Real>& z,
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
      rec1d_p_teno5_mc2_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_mc2_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructTeno5mc2X3(AthenaArray<Real>& z,
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
      rec1d_p_teno5_mc2_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_mc2_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructTeno5korenX1(AthenaArray<Real>& z,
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
      rec1d_p_teno5_koren_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_koren_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructTeno5korenX2(AthenaArray<Real>& z,
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
      rec1d_p_teno5_koren_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_koren_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructTeno5korenX3(AthenaArray<Real>& z,
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
      rec1d_p_teno5_koren_LR<true>(zimt, zimo, zi, zipo, zipt, uL, uR);
    else
      rec1d_p_teno5_koren_LR<false>(zimt, zimo, zi, zipo, zipt, uL, uR);
     zl_(n_tar, i) = uL;
     zr_(n_tar, i) = uR;
   }
 }
