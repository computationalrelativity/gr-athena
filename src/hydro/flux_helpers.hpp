#ifndef HYDRO_FLUX_HELPERS_HPP_
#define HYDRO_FLUX_HELPERS_HPP_

#include <algorithm>
#include <cmath>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../z4c/z4c.hpp"
#include "hydro.hpp"

// #ifndef CORRECTION_TENO
// #define CORRECTION_TENO
// #endif

namespace {

static constexpr Real kOneQuarter       = 1.0 / 4.0;
static constexpr Real kThirteenTwelfths = 13.0 / 12.0;

#pragma omp declare simd
inline void JS_smoothness(Real& b0,
                          Real& b1,
                          Real& b2,
                          const Real um2,
                          const Real um1,
                          const Real u0,
                          const Real up1,
                          const Real up2)
{
  b0 = kThirteenTwelfths * SQR((um2 - 2.0 * um1 + u0)) +
       kOneQuarter * SQR((um2 - 4.0 * um1 + 3.0 * u0));
  b1 = kThirteenTwelfths * SQR((um1 - 2.0 * u0 + up1)) +
       kOneQuarter * SQR((um1 - up1));
  b2 = kThirteenTwelfths * SQR((u0 - 2.0 * up1 + up2)) +
       kOneQuarter * SQR((3.0 * u0 - 4.0 * up1 + up2));
}

static constexpr Real EPSL         = 1e-40;
static constexpr Real c_zcp[3]     = { 9. / 8., 9. / 4., 9. / 8. };
static constexpr Real optimw_pw[3] = { 1. / 16., 5. / 8., 5. / 16. };

static constexpr Real k13Over12 = 13.0 / 12.0;
static constexpr Real TENO_C_T  = 1e-5;

#pragma omp declare simd
static inline Real teno_B0(const Real im1, const Real i, const Real ip1)
{
  return kOneQuarter * SQR(im1 - ip1) + k13Over12 * SQR(im1 - 2.0 * i + ip1);
}
#pragma omp declare simd
static inline Real teno_B1(const Real i, const Real ip1, const Real ip2)
{
  return kOneQuarter * SQR(3.0 * i - 4.0 * ip1 + ip2) +
         k13Over12 * SQR(i - 2.0 * ip1 + ip2);
}
#pragma omp declare simd
static inline Real teno_B2(const Real im2, const Real im1, const Real i)
{
  return kOneQuarter * SQR(im2 - 4.0 * im1 + 3.0 * i) +
         k13Over12 * SQR(im2 - 2.0 * im1 + i);
}

#pragma omp declare simd
static inline Real weno5_zcp_pw_sensor(const Real um2,
                                       const Real um1,
                                       const Real u0,
                                       const Real up1,
                                       const Real up2)
{
  Real b[3];
  JS_smoothness(b[0], b[1], b[2], um2, um1, u0, up1, up2);
  const Real tau    = std::abs(b[0] - b[2]);
  const Real bbar   = (b[0] + b[1] + b[2]) * (1.0 / 3.0);
  const Real d_plus = tau + bbar + EPSL;
  const Real tf     = tau / d_plus;
  Real a0 = optimw_pw[0] * (1.0 + c_zcp[0] * (tau / (EPSL + b[0])) * tf +
                             b[0] / d_plus);
  Real a1 = optimw_pw[1] * (1.0 + c_zcp[1] * (tau / (EPSL + b[1])) * tf +
                             b[1] / d_plus);
  Real a2 = optimw_pw[2] * (1.0 + c_zcp[2] * (tau / (EPSL + b[2])) * tf +
                             b[2] / d_plus);
  const Real s = 1.0 / (a0 + a1 + a2);
  return std::abs(a0 * s - optimw_pw[0]) + std::abs(a1 * s - optimw_pw[1]) +
         std::abs(a2 * s - optimw_pw[2]);
}

static void PhysicalFluxPoint(const int d,
                              const int k,
                              const int j,
                              const int i,
                              Z4c* pz4c,
                              AA& w,
                              AA& derived_ms,
                              Real f_out[NHYDRO])
{
  const Real rho      = w(IDN, k, j, i);
  const Real p        = w(IPR, k, j, i);
  const Real util_u_x = w(IVX, k, j, i);
  const Real util_u_y = w(IVY, k, j, i);
  const Real util_u_z = w(IVZ, k, j, i);
  const Real W        = derived_ms(IX_LOR, k, j, i);
  const Real h        = derived_ms(IX_ETH, k, j, i);

  const Real util_u_d = (d == 0) ? util_u_x : ((d == 1) ? util_u_y : util_u_z);

  const Real adm_gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
  const Real adm_gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
  const Real adm_gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
  const Real adm_gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
  const Real adm_gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
  const Real adm_gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);

  const Real u_d0 =
      adm_gxx * util_u_x + adm_gxy * util_u_y + adm_gxz * util_u_z;
  const Real u_d1 =
      adm_gxy * util_u_x + adm_gyy * util_u_y + adm_gyz * util_u_z;
  const Real u_d2 =
      adm_gxz * util_u_x + adm_gyz * util_u_y + adm_gzz * util_u_z;

  const Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
  const Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);
  const Real sdg =
      pz4c->storage.aux_extended(
          Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma, k, j, i);

  const Real v_d       = util_u_d / W;
  const Real alpha_vmb = alpha * v_d - beta_d;

  const Real rhoW  = rho * W;
  const Real hrhoW = rho * h * W;
  const Real D     = rhoW;
  const Real tau   = rho * h * SQR(W) - rhoW - p;

  f_out[IDN]     = D * sdg * alpha_vmb;
  f_out[IVX]     = hrhoW * u_d0 * sdg * alpha_vmb;
  f_out[IVY]     = hrhoW * u_d1 * sdg * alpha_vmb;
  f_out[IVZ]     = hrhoW * u_d2 * sdg * alpha_vmb;
  f_out[IEN]     = (tau * alpha_vmb + alpha * p * v_d) * sdg;
  f_out[IVX + d] += alpha * p * sdg;
}

#ifdef CORRECTION_TENO
static Real ComputePhi_5pt(const Real f[5][NHYDRO])
{
  Real phi = 1.0;
  for (int n = 0; n < NHYDRO; ++n) {
    Real b0c   = teno_B0(f[1][n], f[2][n], f[3][n]);
    Real b1c   = teno_B1(f[2][n], f[3][n], f[4][n]);
    Real b2c   = teno_B2(f[0][n], f[1][n], f[2][n]);
    Real tau   = std::abs(b0c - b1c) + std::abs(b0c - b2c);
    Real g0    = 1.0 + tau / (b0c + EPSL);
    g0         = g0 * g0;
    g0         = g0 * g0 * g0;
    Real g1    = 1.0 + tau / (b1c + EPSL);
    g1         = g1 * g1;
    g1         = g1 * g1 * g1;
    Real g2    = 1.0 + tau / (b2c + EPSL);
    g2         = g2 * g2;
    g2         = g2 * g2 * g2;
    Real inv_s = 1.0 / (g0 + g1 + g2);
    int ns     = ((g0 * inv_s >= TENO_C_T) ? 1 : 0) +
                 ((g1 * inv_s >= TENO_C_T) ? 1 : 0) +
                 ((g2 * inv_s >= TENO_C_T) ? 1 : 0);
    phi = std::min(phi, Real(ns) / 3.0);
  }
  return phi;
}
#else
static Real ComputePhi_5pt(const Real f[5][NHYDRO])
{
  const Real Cphi = 8.0;
  Real S = 0.0;
  for (int n = 0; n < NHYDRO; ++n)
    S = std::max(
        S, weno5_zcp_pw_sensor(f[0][n], f[1][n], f[2][n], f[3][n], f[4][n]));
  return std::exp(-Cphi * S);
}
#endif

#pragma omp declare simd
static inline Real FDCorrectionCell(Real phi,
                                    Real Fm1,
                                    Real F0,
                                    Real Fp1)
{
  return phi * (Fm1 - 2.0 * F0 + Fp1) * (1.0 / 24.0);
}

#if NSCALARS > 0
static Real ScalarPhi_5pt(const Real sf[5])
{
#ifdef CORRECTION_TENO
  Real b0c   = teno_B0(sf[1], sf[2], sf[3]);
  Real b1c   = teno_B1(sf[2], sf[3], sf[4]);
  Real b2c   = teno_B2(sf[0], sf[1], sf[2]);
  Real tau   = std::abs(b0c - b1c) + std::abs(b0c - b2c);
  Real g0    = 1.0 + tau / (b0c + EPSL);
  g0         = g0 * g0;
  g0         = g0 * g0 * g0;
  Real g1    = 1.0 + tau / (b1c + EPSL);
  g1         = g1 * g1;
  g1         = g1 * g1 * g1;
  Real g2    = 1.0 + tau / (b2c + EPSL);
  g2         = g2 * g2;
  g2         = g2 * g2 * g2;
  Real inv_s = 1.0 / (g0 + g1 + g2);
  int ns     = ((g0 * inv_s >= TENO_C_T) ? 1 : 0) +
               ((g1 * inv_s >= TENO_C_T) ? 1 : 0) +
               ((g2 * inv_s >= TENO_C_T) ? 1 : 0);
  return Real(ns) / 3.0;
#else
  const Real Cphi = 8.0;
  Real Ss = weno5_zcp_pw_sensor(sf[0], sf[1], sf[2], sf[3], sf[4]);
  return std::exp(-Cphi * Ss);
#endif
}
#endif

}  // namespace

#endif  // HYDRO_FLUX_HELPERS_HPP_
