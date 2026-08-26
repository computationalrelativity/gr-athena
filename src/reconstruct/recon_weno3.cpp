// C/C++ headers
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
// WENO3 and WENO3-Z
//
// 3rd-order, 3-point stencil {i-1,i,i+1}, 2 sub-stencils (r=2):
//
//   u0 = (-f_{i-1} + 3 f_i) / 2   [backward,  S0 = {i-1,i}]
//   u1 = ( f_i + f_{i+1}) / 2     [forward,   S1 = {i,i+1}]
//
// Smoothness indicators:
//   b0 = (f_i - f_{i-1})^2
//   b1 = (f_{i+1} - f_i)^2
//
// Ideal weights:  FV = {1/3, 2/3},  PW = {1/4, 3/4}
//
// WENO3-Z:  alpha_k = d_k * (1 + tau/(eps + b_k)),  tau = |b0 - b1|
//
// Paired L+R: 4-point window {i-1,i,i+1,i+2}
//   L from {i-1,i,i+1}, R from {i,i+1,i+2}
//   b1 = (f_{i+1}-f_i)^2 shared, tau computed from L betas only
// ----------------------------------------------------------------------------
namespace
{

// ---------------------------------------------------------------------------
// File-scope constants
// ---------------------------------------------------------------------------

static constexpr Real dw3_fv[2] = { 1. / 3., 2. / 3. };
static constexpr Real dw3_pw[2] = { 1. / 4., 3. / 4. };
static constexpr Real EPSL      = 1e-40;

// ---------------------------------------------------------------------------
// Paired L+R - WENO3-JS
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_weno3_LR(const Real fm1,
                             const Real f0,
                             const Real fp1,
                             const Real fp2,
                             Real& uL,
                             Real& uR)
{
  const auto& dw = pw ? dw3_pw : dw3_fv;

  const Real b0_L = SQR(f0 - fm1);
  const Real b1_L = SQR(fp1 - f0);

  const Real a0_L  = dw[0] / SQR(EPSL + b0_L);
  const Real a1_L  = dw[1] / SQR(EPSL + b1_L);
  const Real dsaL  = 1.0 / (a0_L + a1_L);
  const Real uk0_L = (-fm1 + 3.0 * f0) * 0.5;
  const Real uk1_L = (f0 + fp1) * 0.5;
  uL              = dsaL * (a0_L * uk0_L + a1_L * uk1_L);

  const Real b0_R = b0_L;
  const Real b1_R = b1_L;

  const Real a0_R  = dw[1] / SQR(EPSL + b0_R);
  const Real a1_R  = dw[0] / SQR(EPSL + b1_R);
  const Real dsaR  = 1.0 / (a0_R + a1_R);
  const Real uk0_R = (fm1 + f0) * 0.5;
  const Real uk1_R = (3.0 * f0 - fp1) * 0.5;
  uR              = dsaR * (a0_R * uk0_R + a1_R * uk1_R);
}

// ---------------------------------------------------------------------------
// Paired L+R - WENO3-Z
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_weno3z_LR(const Real fm1,
                              const Real f0,
                              const Real fp1,
                              const Real fp2,
                              Real& uL,
                              Real& uR)
{
  const auto& dw = pw ? dw3_pw : dw3_fv;

  const Real b0_L = SQR(f0 - fm1);
  const Real b1_L = SQR(fp1 - f0);
  const Real tau  = std::abs(b0_L - b1_L);

  const Real a0_L  = dw[0] * (1.0 + tau / (EPSL + b0_L));
  const Real a1_L  = dw[1] * (1.0 + tau / (EPSL + b1_L));
  const Real dsaL  = 1.0 / (a0_L + a1_L);
  const Real uk0_L = (-fm1 + 3.0 * f0) * 0.5;
  const Real uk1_L = (f0 + fp1) * 0.5;
  uL              = dsaL * (a0_L * uk0_L + a1_L * uk1_L);

  const Real b0_R = b0_L;
  const Real b1_R = b1_L;

  const Real a0_R  = dw[1] * (1.0 + tau / (EPSL + b0_R));
  const Real a1_R  = dw[0] * (1.0 + tau / (EPSL + b1_R));
  const Real dsaR  = 1.0 / (a0_R + a1_R);
  const Real uk0_R = (fm1 + f0) * 0.5;
  const Real uk1_R = (3.0 * f0 - fp1) * 0.5;
  uR              = dsaR * (a0_R * uk0_R + a1_R * uk1_R);
}

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// WENO3-JS direction wrappers
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno3X1(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k, j, i - 1);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j, i + 1);
    const Real fp2 = z(n_src, k, j, i + 2);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno3X2(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k, j - 1, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j + 1, i);
    const Real fp2 = z(n_src, k, j + 2, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno3X3(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k - 1, j, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k + 1, j, i);
    const Real fp2 = z(n_src, k + 2, j, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------
// WENO3-Z direction wrappers
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno3ZX1(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k, j, i - 1);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j, i + 1);
    const Real fp2 = z(n_src, k, j, i + 2);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3z_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3z_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno3ZX2(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k, j - 1, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j + 1, i);
    const Real fp2 = z(n_src, k, j + 2, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3z_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3z_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno3ZX3(AthenaArray<Real>& z,
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
    const Real fm1 = z(n_src, k - 1, j, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k + 1, j, i);
    const Real fp2 = z(n_src, k + 2, j, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno3z_LR<true>(fm1, f0, fp1, fp2, uL, uR);
    else
      rec1d_p_weno3z_LR<false>(fm1, f0, fp1, fp2, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}
