// C/C++ headers
#include <cmath>

// Athena++ classes headers
#include "../athena.hpp"
#include "reconstruction.hpp"
#include "reconstruction_utils.hpp"

// ----------------------------------------------------------------------------
// WENO7 and WENO7-Z
//
// 7th-order, 7-point global stencil {i-3..i+3}, 4 sub-stencils (r=4):
//
// Sub-stencils (each 4 points -> face i+1/2):
//   S0 : {i-3, i-2, i-1, i  }
//   S1 : {i-2, i-1, i  , i+1}
//   S2 : {i-1, i  , i+1, i+2}
//   S3 : {i  , i+1, i+2, i+3}
//
// FV ideal weights:         {  1/35,  12/35,  18/35,   4/35}
// PW ideal weights:         {  1/64,  21/64,  35/64,   7/64}
//
// WENO7-Z:  alpha_k = d_k * (1 + tau/(eps + b_k)),  tau = |b0 - b3|
// tau is shared between L and R (computed from L betas only).
//
// Paired L+R: single 7-point window {i-3..i+3} fed to the LR template.
// R state is computed from reversed data with β mapped {3,2,1,0}.
//
// Requires NGHOST >= 4.  Not compatible with xorder_use_fb = true
// (num_enlarge_layer=1 pushes the 7-point stencil out of bounds).
// ----------------------------------------------------------------------------
namespace
{

// ---------------------------------------------------------------------------
// File-scope constants
// ---------------------------------------------------------------------------

static constexpr Real kOneTwelfth    = 1.0 / 12.0;
static constexpr Real kOneSixteenth  = 1.0 / 16.0;

static constexpr Real ow7_fv[4] = { 1. / 35., 12. / 35., 18. / 35., 4. / 35. };
static constexpr Real ow7_pw[4] = { 1. / 64., 21. / 64., 35. / 64., 7. / 64. };
static constexpr Real EPSL       = 1e-40;

// ---------------------------------------------------------------------------
// JS smoothness indicators
//
// Exploits mirror symmetry:  b1 <-> b2 and b0 <-> b3 under reversal of all
// indices.
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_weno7_JS_smoothness(Real& b0,
                                        Real& b1,
                                        Real& b2,
                                        Real& b3,
                                        const Real fm3,
                                        const Real fm2,
                                        const Real fm1,
                                        const Real f0,
                                        const Real fp1,
                                        const Real fp2,
                                        const Real fp3)
{
  b0 = fm3 * (2107. * fm3 - 9402. * fm2 + 7042. * fm1 - 1854. * f0) +
       fm2 * (11003. * fm2 - 17246. * fm1 + 4642. * f0) +
       fm1 * (7043. * fm1 - 3882. * f0) +
       f0 * (547. * f0);

  b1 = fm2 * (267. * fm2 - 1098. * fm1 + 974. * f0 - 262. * fp1) +
       fm1 * (1203. * fm1 - 2278. * f0 + 642. * fp1) +
       f0 * (1203. * f0 - 702. * fp1) +
       fp1 * (107. * fp1);

  b2 = fm1 * (107. * fm1 - 702. * f0 + 642. * fp1 - 262. * fp2) +
       f0 * (1203. * f0 - 2278. * fp1 + 974. * fp2) +
       fp1 * (1203. * fp1 - 1098. * fp2) +
       fp2 * (267. * fp2);

  b3 = f0 * (547. * f0 - 3882. * fp1 + 4642. * fp2 - 1854. * fp3) +
       fp1 * (7043. * fp1 - 17246. * fp2 + 7042. * fp3) +
       fp2 * (11003. * fp2 - 9402. * fp3) +
       fp3 * (2107. * fp3);
}

// ---------------------------------------------------------------------------
// FV (cell-average) stencil polynomials
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_weno7stencils(Real& u0,
                                  Real& u1,
                                  Real& u2,
                                  Real& u3,
                                  const Real fm3,
                                  const Real fm2,
                                  const Real fm1,
                                  const Real f0,
                                  const Real fp1,
                                  const Real fp2,
                                  const Real fp3)
{
  u0 = kOneTwelfth *
       (-3. * fm3 + 13. * fm2 - 23. * fm1 + 25. * f0);
  u1 = kOneTwelfth *
       (fm2 - 5. * fm1 + 13. * f0 + 3. * fp1);
  u2 = kOneTwelfth *
       (-fm1 + 7. * f0 + 7. * fp1 - fp2);
  u3 = kOneTwelfth *
       (3. * f0 + 13. * fp1 - 5. * fp2 + fp3);
}

// ---------------------------------------------------------------------------
// PW (pointwise) stencil polynomials
// ---------------------------------------------------------------------------

#pragma omp declare simd
inline void rec1d_p_weno7stencils_pw(Real& u0,
                                     Real& u1,
                                     Real& u2,
                                     Real& u3,
                                     const Real fm3,
                                     const Real fm2,
                                     const Real fm1,
                                     const Real f0,
                                     const Real fp1,
                                     const Real fp2,
                                     const Real fp3)
{
  u0 = kOneSixteenth *
       (-5. * fm3 + 21. * fm2 - 35. * fm1 + 35. * f0);
  u1 = kOneSixteenth *
       (fm2 - 5. * fm1 + 15. * f0 + 5. * fp1);
  u2 = kOneSixteenth *
       (-fm1 + 9. * f0 + 9. * fp1 - fp2);
  u3 = kOneSixteenth *
       (5. * f0 + 15. * fp1 - 5. * fp2 + fp3);
}

// ---------------------------------------------------------------------------
// Paired L+R — WENO7-JS
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_weno7_LR(const Real fm3,
                             const Real fm2,
                             const Real fm1,
                             const Real f0,
                             const Real fp1,
                             const Real fp2,
                             const Real fp3,
                             Real& uL,
                             Real& uR)
{
  const auto& ow = pw ? ow7_pw : ow7_fv;

  Real b[4];
  rec1d_p_weno7_JS_smoothness(
    b[0], b[1], b[2], b[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);

  const Real a0_L = ow[0] / SQR(EPSL + b[0]);
  const Real a1_L = ow[1] / SQR(EPSL + b[1]);
  const Real a2_L = ow[2] / SQR(EPSL + b[2]);
  const Real a3_L = ow[3] / SQR(EPSL + b[3]);
  const Real dsaL = 1.0 / (a0_L + a1_L + a2_L + a3_L);

  Real ukL[4];
  if constexpr (pw)
    rec1d_p_weno7stencils_pw(
      ukL[0], ukL[1], ukL[2], ukL[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);
  else
    rec1d_p_weno7stencils(
      ukL[0], ukL[1], ukL[2], ukL[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);
  uL = dsaL * (a0_L * ukL[0] + a1_L * ukL[1] + a2_L * ukL[2] +
               a3_L * ukL[3]);

  const Real a0_R = ow[0] / SQR(EPSL + b[3]);
  const Real a1_R = ow[1] / SQR(EPSL + b[2]);
  const Real a2_R = ow[2] / SQR(EPSL + b[1]);
  const Real a3_R = ow[3] / SQR(EPSL + b[0]);
  const Real dsaR = 1.0 / (a0_R + a1_R + a2_R + a3_R);

  Real ukR[4];
  if constexpr (pw)
    rec1d_p_weno7stencils_pw(
      ukR[0], ukR[1], ukR[2], ukR[3], fp3, fp2, fp1, f0, fm1, fm2, fm3);
  else
    rec1d_p_weno7stencils(
      ukR[0], ukR[1], ukR[2], ukR[3], fp3, fp2, fp1, f0, fm1, fm2, fm3);
  uR = dsaR * (a0_R * ukR[0] + a1_R * ukR[1] + a2_R * ukR[2] +
               a3_R * ukR[3]);
}

// ---------------------------------------------------------------------------
// Paired L+R — WENO7-Z
// ---------------------------------------------------------------------------

#pragma omp declare simd
template <bool pw>
inline void rec1d_p_weno7z_LR(const Real fm3,
                              const Real fm2,
                              const Real fm1,
                              const Real f0,
                              const Real fp1,
                              const Real fp2,
                              const Real fp3,
                              Real& uL,
                              Real& uR)
{
  const auto& ow = pw ? ow7_pw : ow7_fv;

  Real b[4];
  rec1d_p_weno7_JS_smoothness(
    b[0], b[1], b[2], b[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);
  const Real tau = std::abs(b[0] - b[3]);

  const Real a0_L = ow[0] * (1.0 + tau / (EPSL + b[0]));
  const Real a1_L = ow[1] * (1.0 + tau / (EPSL + b[1]));
  const Real a2_L = ow[2] * (1.0 + tau / (EPSL + b[2]));
  const Real a3_L = ow[3] * (1.0 + tau / (EPSL + b[3]));
  const Real dsaL = 1.0 / (a0_L + a1_L + a2_L + a3_L);

  Real ukL[4];
  if constexpr (pw)
    rec1d_p_weno7stencils_pw(
      ukL[0], ukL[1], ukL[2], ukL[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);
  else
    rec1d_p_weno7stencils(
      ukL[0], ukL[1], ukL[2], ukL[3], fm3, fm2, fm1, f0, fp1, fp2, fp3);
  uL = dsaL * (a0_L * ukL[0] + a1_L * ukL[1] + a2_L * ukL[2] +
               a3_L * ukL[3]);

  const Real a0_R = ow[0] * (1.0 + tau / (EPSL + b[3]));
  const Real a1_R = ow[1] * (1.0 + tau / (EPSL + b[2]));
  const Real a2_R = ow[2] * (1.0 + tau / (EPSL + b[1]));
  const Real a3_R = ow[3] * (1.0 + tau / (EPSL + b[0]));
  const Real dsaR = 1.0 / (a0_R + a1_R + a2_R + a3_R);

  Real ukR[4];
  if constexpr (pw)
    rec1d_p_weno7stencils_pw(
      ukR[0], ukR[1], ukR[2], ukR[3], fp3, fp2, fp1, f0, fm1, fm2, fm3);
  else
    rec1d_p_weno7stencils(
      ukR[0], ukR[1], ukR[2], ukR[3], fp3, fp2, fp1, f0, fm1, fm2, fm3);
  uR = dsaR * (a0_R * ukR[0] + a1_R * ukR[1] + a2_R * ukR[2] +
               a3_R * ukR[3]);
}

}  // namespace
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// WENO7-JS direction wrappers
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno7X1(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k, j, i - 3);
    const Real fm2 = z(n_src, k, j, i - 2);
    const Real fm1 = z(n_src, k, j, i - 1);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j, i + 1);
    const Real fp2 = z(n_src, k, j, i + 2);
    const Real fp3 = z(n_src, k, j, i + 3);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno7X2(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k, j - 3, i);
    const Real fm2 = z(n_src, k, j - 2, i);
    const Real fm1 = z(n_src, k, j - 1, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j + 1, i);
    const Real fp2 = z(n_src, k, j + 2, i);
    const Real fp3 = z(n_src, k, j + 3, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno7X3(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k - 3, j, i);
    const Real fm2 = z(n_src, k - 2, j, i);
    const Real fm1 = z(n_src, k - 1, j, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k + 1, j, i);
    const Real fp2 = z(n_src, k + 2, j, i);
    const Real fp3 = z(n_src, k + 3, j, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

// ----------------------------------------------------------------------------
// WENO7-Z direction wrappers
// ----------------------------------------------------------------------------

void Reconstruction::ReconstructWeno7ZX1(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k, j, i - 3);
    const Real fm2 = z(n_src, k, j, i - 2);
    const Real fm1 = z(n_src, k, j, i - 1);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j, i + 1);
    const Real fp2 = z(n_src, k, j, i + 2);
    const Real fp3 = z(n_src, k, j, i + 3);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7z_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7z_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i + 1) = uL;
    zr_(n_tar, i)     = uR;
  }
}

void Reconstruction::ReconstructWeno7ZX2(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k, j - 3, i);
    const Real fm2 = z(n_src, k, j - 2, i);
    const Real fm1 = z(n_src, k, j - 1, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k, j + 1, i);
    const Real fp2 = z(n_src, k, j + 2, i);
    const Real fp3 = z(n_src, k, j + 3, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7z_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7z_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}

void Reconstruction::ReconstructWeno7ZX3(AthenaArray<Real>& z,
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
    const Real fm3 = z(n_src, k - 3, j, i);
    const Real fm2 = z(n_src, k - 2, j, i);
    const Real fm1 = z(n_src, k - 1, j, i);
    const Real f0  = z(n_src, k, j, i);
    const Real fp1 = z(n_src, k + 1, j, i);
    const Real fp2 = z(n_src, k + 2, j, i);
    const Real fp3 = z(n_src, k + 3, j, i);

    Real uL, uR;
    if (xorder_pointwise)
      rec1d_p_weno7z_LR<true>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    else
      rec1d_p_weno7z_LR<false>(fm3, fm2, fm1, f0, fp1, fp2, fp3, uL, uR);
    zl_(n_tar, i) = uL;
    zr_(n_tar, i) = uR;
  }
}
