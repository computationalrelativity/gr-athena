// Flux correction:
// Removes the O(dx^2) error term involving d^3F/dx^3 from
// the flux-difference formula.
//
// A 4-point linear FD stencil on cell-centered physical fluxes computes F''
// at each face.
//
// WENO5-ZC+-PW (or TENO5 with -DCORRECTION_TENO) smoothness sensor,
// phi in [0,1], gates the correction.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "hydro.hpp"

// Debug:
// #ifndef CORRECTION_DIAG
// #define CORRECTION_DIAG
// #endif

// #ifndef CORRECTION_TENO
// #define CORRECTION_TENO
// #endif

namespace
{

static constexpr Real kOneQuarter       = 1.0 / 4.0;
static constexpr Real kThirteenTwelfths = 13.0 / 12.0;

#pragma omp declare simd
void JS_smoothness(Real& b0,
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

// TENO5 smoothness sensor.
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

// WENO5-ZC+-PW smoothness sensor.
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
  Real a0      = optimw_pw[0] *
                 (1.0 + c_zcp[0] * (tau / (EPSL + b[0])) * tf + b[0] / d_plus);
  Real a1      = optimw_pw[1] *
                 (1.0 + c_zcp[1] * (tau / (EPSL + b[1])) * tf + b[1] / d_plus);
  Real a2      = optimw_pw[2] *
                 (1.0 + c_zcp[2] * (tau / (EPSL + b[2])) * tf + b[2] / d_plus);
  const Real s = 1.0 / (a0 + a1 + a2);
  return std::abs(a0 * s - optimw_pw[0]) + std::abs(a1 * s - optimw_pw[1]) +
         std::abs(a2 * s - optimw_pw[2]);
}

// Physical flux F^d at cell (k,j,i), using LLF Riemann solver conventions.
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
    pz4c->storage.aux_extended(Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma, k, j, i);

  const Real v_d       = util_u_d / W;
  const Real alpha_vmb = alpha * v_d - beta_d;

  const Real rhoW  = rho * W;
  const Real hrhoW = rho * h * W;
  const Real D     = rhoW;
  const Real tau   = rho * h * SQR(W) - rhoW - p;

  f_out[IDN] = D * sdg * alpha_vmb;
  f_out[IVX] = hrhoW * u_d0 * sdg * alpha_vmb;
  f_out[IVY] = hrhoW * u_d1 * sdg * alpha_vmb;
  f_out[IVZ] = hrhoW * u_d2 * sdg * alpha_vmb;
  f_out[IEN] = (tau * alpha_vmb + alpha * p * v_d) * sdg;
  f_out[IVX + d] += alpha * p * sdg;
}

}  // namespace

// ---------------------------------------------------------------------------
// CorrectFluxX1  --  face x_{i+1/2}
//   F''     = (F_{i-1} - F_i - F_{i+1} + F_{i+2}) / (2 Delta x^2)
//   hflux  -= phi * F'' * Delta x^2 / 24
//           = phi * (F_{i-1} - F_i - F_{i+1} + F_{i+2}) / 48
// ---------------------------------------------------------------------------
void Hydro::CorrectFluxX1(AA& hflux,
                          AA& w,
                          AA& derived_ms,
                          AA& r_scalar,
                          AA& sflux,
                          int k,
                          int j,
                          int il,
                          int iu)
{
  MeshBlock* pmb     = pmy_block;
  Z4c* pz4c          = pmb->pz4c;
  const int d        = 0;
  const Real mb      = pmb->peos->GetEOS().GetBaryonMass();
  const Real rho_cut = mb * pmb->peos->GetEOS().GetDensityFloor() * 10.0;
  const Real Cphi    = 8;
  if (il - 3 < 0 || iu + 1 >= pmy_block->ncells1)
    return;

#ifdef CORRECTION_DIAG
  static int x1_calls = 0;
  static int x1_tot = 0, x1_dens = 0;
  static Real x1_phi_sum = 0, x1_corr_max[NHYDRO] = {};
#endif

  for (int i = il - 1; i <= iu - 1; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k, j, i - 2 + s, pz4c, w, derived_ms, f[s]);

#ifdef CORRECTION_DIAG
    x1_tot++;
#endif
    if (w(IDN, k, j, i - 2) <= rho_cut || w(IDN, k, j, i - 1) <= rho_cut ||
        w(IDN, k, j, i) <= rho_cut || w(IDN, k, j, i + 1) <= rho_cut ||
        w(IDN, k, j, i + 2) <= rho_cut)
      continue;

#ifdef CORRECTION_DIAG
    x1_dens++;
#endif
#ifdef CORRECTION_TENO
    Real phi = 1.0;
    for (int n = 0; n < NHYDRO; ++n)
    {
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
      phi        = std::min(phi, Real(ns) / 3.0);
    }
#else
    Real S = 0.0;
    for (int n = 0; n < NHYDRO; ++n)
      S = std::max(
        S, weno5_zcp_pw_sensor(f[0][n], f[1][n], f[2][n], f[3][n], f[4][n]));
    Real phi = std::exp(-Cphi * S);
#endif

#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k, j, i - 2 + s);
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
      int ns2    = ((g0 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g1 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g2 * inv_s >= TENO_C_T) ? 1 : 0);
      phi        = std::min(phi, Real(ns2) / 3.0);
#else
      Real Ss = weno5_zcp_pw_sensor(sf[0], sf[1], sf[2], sf[3], sf[4]);
      phi     = std::min(phi, std::exp(-Cphi * Ss));
#endif
    }
#endif
    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i + 1) -= phi * corr;
    }
#ifdef CORRECTION_DIAG
    x1_phi_sum += phi;
    for (int nn = 0; nn < NHYDRO; ++nn)
    {
      Real abs_c = std::abs(phi * (f[1][nn] - f[2][nn] - f[3][nn] + f[4][nn]) *
                            (1.0 / 48.0));
      if (abs_c > x1_corr_max[nn])
        x1_corr_max[nn] = abs_c;
    }
    if (++x1_calls % 2000 == 0)
    {
#pragma omp critical(corr_diag)
      {
        printf(
          "[CORR_DIAG] X1 faces=%d gated=%d(%.1f%%) phi_avg=%.4f |corr|_max:",
          x1_tot,
          x1_dens,
          100.0 * x1_dens / std::max(1, x1_tot),
          x1_phi_sum / std::max(1, x1_dens));
        for (int nn = 0; nn < NHYDRO; ++nn)
          printf(" %.2e", x1_corr_max[nn]);
        printf("\n");
      }
      x1_tot     = 0;
      x1_dens    = 0;
      x1_phi_sum = 0;
      for (int nn = 0; nn < NHYDRO; ++nn)
        x1_corr_max[nn] = 0;
    }
#endif
#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      const Real corr = phi *
                        (f[1][IDN] * r_scalar(ns, k, j, i - 1) -
                         f[2][IDN] * r_scalar(ns, k, j, i) -
                         f[3][IDN] * r_scalar(ns, k, j, i + 1) +
                         f[4][IDN] * r_scalar(ns, k, j, i + 2)) *
                        (1.0 / 48.0);
      sflux(ns, k, j, i + 1) -= corr;
    }
#endif
  }
}

// ---------------------------------------------------------------------------
// CorrectFluxX2  --  face y_{j-1/2}
//   F''     = (F_{j-2} - F_{j-1} - F_j + F_{j+1}) / (2 Delta y^2)
//   hflux  -= phi * F'' * Delta y^2 / 24
//           = phi * (F_{j-2} - F_{j-1} - F_j + F_{j+1}) / 48
// ---------------------------------------------------------------------------
void Hydro::CorrectFluxX2(AA& hflux,
                          AA& w,
                          AA& derived_ms,
                          AA& r_scalar,
                          AA& sflux,
                          int k,
                          int j,
                          int il,
                          int iu)
{
  MeshBlock* pmb     = pmy_block;
  Z4c* pz4c          = pmb->pz4c;
  const int d        = 1;
  const Real mb      = pmb->peos->GetEOS().GetBaryonMass();
  const Real rho_cut = mb * pmb->peos->GetEOS().GetDensityFloor() * 10.0;
  const Real Cphi    = 8;
  if (j - 3 < 0 || j + 1 >= pmy_block->ncells2)
    return;

#ifdef CORRECTION_DIAG
  static int x2_calls = 0;
  static int x2_tot = 0, x2_dens = 0;
  static Real x2_phi_sum = 0, x2_corr_max[NHYDRO] = {};
#endif

  for (int i = il; i <= iu; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k, j - 3 + s, i, pz4c, w, derived_ms, f[s]);

#ifdef CORRECTION_DIAG
    x2_tot++;
#endif
    if (w(IDN, k, j - 3, i) <= rho_cut || w(IDN, k, j - 2, i) <= rho_cut ||
        w(IDN, k, j - 1, i) <= rho_cut || w(IDN, k, j, i) <= rho_cut ||
        w(IDN, k, j + 1, i) <= rho_cut)
      continue;

#ifdef CORRECTION_DIAG
    x2_dens++;
#endif
#ifdef CORRECTION_TENO
    Real phi = 1.0;
    for (int n = 0; n < NHYDRO; ++n)
    {
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
      phi        = std::min(phi, Real(ns) / 3.0);
    }
#else
    Real S = 0.0;
    for (int n = 0; n < NHYDRO; ++n)
      S = std::max(
        S, weno5_zcp_pw_sensor(f[0][n], f[1][n], f[2][n], f[3][n], f[4][n]));
    Real phi = std::exp(-Cphi * S);
#endif
#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k, j - 3 + s, i);
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
      int ns2    = ((g0 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g1 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g2 * inv_s >= TENO_C_T) ? 1 : 0);
      phi        = std::min(phi, Real(ns2) / 3.0);
#else
      Real Ss = weno5_zcp_pw_sensor(sf[0], sf[1], sf[2], sf[3], sf[4]);
      phi     = std::min(phi, std::exp(-Cphi * Ss));
#endif
    }
#endif

    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i) -= phi * corr;
    }
#ifdef CORRECTION_DIAG
    x2_phi_sum += phi;
    for (int nn = 0; nn < NHYDRO; ++nn)
    {
      Real abs_c = std::abs(phi * (f[1][nn] - f[2][nn] - f[3][nn] + f[4][nn]) *
                            (1.0 / 48.0));
      if (abs_c > x2_corr_max[nn])
        x2_corr_max[nn] = abs_c;
    }
    if (++x2_calls % 2000 == 0)
    {
#pragma omp critical(corr_diag)
      {
        printf(
          "[CORR_DIAG] X2 faces=%d gated=%d(%.1f%%) phi_avg=%.4f |corr|_max:",
          x2_tot,
          x2_dens,
          100.0 * x2_dens / std::max(1, x2_tot),
          x2_phi_sum / std::max(1, x2_dens));
        for (int nn = 0; nn < NHYDRO; ++nn)
          printf(" %.2e", x2_corr_max[nn]);
        printf("\n");
      }
      x2_tot     = 0;
      x2_dens    = 0;
      x2_phi_sum = 0;
      for (int nn = 0; nn < NHYDRO; ++nn)
        x2_corr_max[nn] = 0;
    }
#endif
#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      const Real corr = phi *
                        (f[1][IDN] * r_scalar(ns, k, j - 2, i) -
                         f[2][IDN] * r_scalar(ns, k, j - 1, i) -
                         f[3][IDN] * r_scalar(ns, k, j, i) +
                         f[4][IDN] * r_scalar(ns, k, j + 1, i)) *
                        (1.0 / 48.0);
      sflux(ns, k, j, i) -= corr;
    }
#endif
  }
}

// ---------------------------------------------------------------------------
// CorrectFluxX3  --  face z_{k-1/2}
//   F''     = (F_{k-2} - F_{k-1} - F_k + F_{k+1}) / (2 Delta z^2)
//   hflux  -= phi * F'' * Delta z^2 / 24
//           = phi * (F_{k-2} - F_{k-1} - F_k + F_{k+1}) / 48
// ---------------------------------------------------------------------------
void Hydro::CorrectFluxX3(AA& hflux,
                          AA& w,
                          AA& derived_ms,
                          AA& r_scalar,
                          AA& sflux,
                          int k,
                          int j,
                          int il,
                          int iu)
{
  MeshBlock* pmb     = pmy_block;
  Z4c* pz4c          = pmb->pz4c;
  const int d        = 2;
  const Real mb      = pmb->peos->GetEOS().GetBaryonMass();
  const Real rho_cut = mb * pmb->peos->GetEOS().GetDensityFloor() * 10.0;
  const Real Cphi    = 8;
  if (k - 3 < 0 || k + 1 >= pmy_block->ncells3)
    return;

#ifdef CORRECTION_DIAG
  static int x3_calls = 0;
  static int x3_tot = 0, x3_dens = 0;
  static Real x3_phi_sum = 0, x3_corr_max[NHYDRO] = {};
#endif

  for (int i = il; i <= iu; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k - 3 + s, j, i, pz4c, w, derived_ms, f[s]);

#ifdef CORRECTION_DIAG
    x3_tot++;
#endif
    if (w(IDN, k - 3, j, i) <= rho_cut || w(IDN, k - 2, j, i) <= rho_cut ||
        w(IDN, k - 1, j, i) <= rho_cut || w(IDN, k, j, i) <= rho_cut ||
        w(IDN, k + 1, j, i) <= rho_cut)
      continue;

#ifdef CORRECTION_DIAG
    x3_dens++;
#endif
#ifdef CORRECTION_TENO
    Real phi = 1.0;
    for (int n = 0; n < NHYDRO; ++n)
    {
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
      phi        = std::min(phi, Real(ns) / 3.0);
    }
#else
    Real S = 0.0;
    for (int n = 0; n < NHYDRO; ++n)
      S = std::max(
        S, weno5_zcp_pw_sensor(f[0][n], f[1][n], f[2][n], f[3][n], f[4][n]));
    Real phi = std::exp(-Cphi * S);
#endif
#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k - 3 + s, j, i);
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
      int ns2    = ((g0 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g1 * inv_s >= TENO_C_T) ? 1 : 0) +
                   ((g2 * inv_s >= TENO_C_T) ? 1 : 0);
      phi        = std::min(phi, Real(ns2) / 3.0);
#else
      Real Ss = weno5_zcp_pw_sensor(sf[0], sf[1], sf[2], sf[3], sf[4]);
      phi     = std::min(phi, std::exp(-Cphi * Ss));
#endif
    }
#endif

    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i) -= phi * corr;
    }
#ifdef CORRECTION_DIAG
    x3_phi_sum += phi;
    for (int nn = 0; nn < NHYDRO; ++nn)
    {
      Real abs_c = std::abs(phi * (f[1][nn] - f[2][nn] - f[3][nn] + f[4][nn]) *
                            (1.0 / 48.0));
      if (abs_c > x3_corr_max[nn])
        x3_corr_max[nn] = abs_c;
    }
    if (++x3_calls % 2000 == 0)
    {
#pragma omp critical(corr_diag)
      {
        printf(
          "[CORR_DIAG] X3 faces=%d gated=%d(%.1f%%) phi_avg=%.4f |corr|_max:",
          x3_tot,
          x3_dens,
          100.0 * x3_dens / std::max(1, x3_tot),
          x3_phi_sum / std::max(1, x3_dens));
        for (int nn = 0; nn < NHYDRO; ++nn)
          printf(" %.2e", x3_corr_max[nn]);
        printf("\n");
      }
      x3_tot     = 0;
      x3_dens    = 0;
      x3_phi_sum = 0;
      for (int nn = 0; nn < NHYDRO; ++nn)
        x3_corr_max[nn] = 0;
    }
#endif
#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      const Real corr = phi *
                        (f[1][IDN] * r_scalar(ns, k - 2, j, i) -
                         f[2][IDN] * r_scalar(ns, k - 1, j, i) -
                         f[3][IDN] * r_scalar(ns, k, j, i) +
                         f[4][IDN] * r_scalar(ns, k + 1, j, i)) *
                        (1.0 / 48.0);
      sflux(ns, k, j, i) -= corr;
    }
#endif
  }
}
