// ---------------------------------------------------------------------------
// calculate_fluxes_split.cpp
//   GRHD flux-split path:
//
//   1. Compute cell-centered physical flux F_i = F(U_i, g_i)
//   2. Compute per-cell local LLF eigenvalue lambda_max_i
//   3. Split: F^+ = 0.5*(F + lambda_max*U), F^- = 0.5*(F - lambda_max*U)
//   4. Apply FD correction at cell centers (3-point, phi-gated)
//   5. WENO5-reconstruct corrected split fluxes to faces
//      - F^+ : left-bias  -> face i-1/2
//      - F^- : right-bias -> face i-1/2
//   6. Scalars are split uniformly with the same lambda_max
//
//   MHD is not supported (deferred).  Selecting split_llf when MHD is
//   enabled silently falls back to the Riemann-solver path via the
//   dispatch in CalculateFluxes.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../utils/linear_algebra.hpp"
#include "../z4c/z4c.hpp"
#include "hydro.hpp"
#include "flux_helpers.hpp"
#include "rsolvers/eigenvalues.hpp"

void Hydro::CalculateFluxesSplit(AA& w,
                                 AA& r,
                                 FaceField& b,
                                 AA& bcc,
                                 AA (&hflux)[3],
                                 AA (&sflux)[3],
                                 Reconstruction::ReconstructionVariant rv,
                                 const int num_enlarge_layer,
                                 ThreadCache* cache)
{
  using namespace LinearAlgebra;

  MeshBlock* pmb       = pmy_block;
  Z4c* pz4c            = pmb->pz4c;
  Reconstruction* pr   = pmb->precon;
  const int N1         = pmb->ncells1;
  const int N2         = pmb->ncells2;
  const int N3         = pmb->ncells3;
  const int NV         = pmb->nverts1;

  hflux[X1DIR].ZeroClear();
  hflux[X2DIR].ZeroClear();
  hflux[X3DIR].ZeroClear();
#if NSCALARS > 0
  sflux[X1DIR].ZeroClear();
  sflux[X2DIR].ZeroClear();
  sflux[X3DIR].ZeroClear();
#endif

  int il, iu, jl, ju, kl, ku;

  // -- X1 direction --------------------------------------------------------
  {
    const int d = 0;
    pr->SetIndicialLimitsCalculateFluxes(
        IVX, il, iu, jl, ju, kl, ku, num_enlarge_layer);

    // 1. Pre-compute cell-centered physical flux + eigenvalues + split
    AA f(NHYDRO, N3, N2, N1);
    AA sf_p(NHYDRO, N3, N2, N1);
    AA sf_m(NHYDRO, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real f_tmp[NHYDRO];
          PhysicalFluxPoint(d, k, j, i, pz4c, w, derived_ms, f_tmp);
          for (int n = 0; n < NHYDRO; ++n)
            f(n, k, j, i) = f_tmp[n];

          Real cs2     = derived_ms(IX_CS2, k, j, i);
          Real W_lcl   = derived_ms(IX_LOR, k, j, i);
          Real util_d  = w(IVX, k, j, i);
          Real v_d     = util_d / W_lcl;

          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);

          Real vx = w(IVX, k, j, i) / W_lcl;
          Real vy = w(IVY, k, j, i) / W_lcl;
          Real vz = w(IVZ, k, j, i) / W_lcl;
          Real v2 = gxx * vx * vx + gyy * vy * vy + gzz * vz * vz +
                    2.0 * (gxy * vx * vy + gxz * vx * vz + gyz * vy * vz);

          Real detg = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);
          Real oodetg = 1.0 / detg;
          Real guu_dd =
              Inv3MetricDiag(oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);

          Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);

          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p, &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          for (int n = 0; n < NHYDRO; ++n) {
            Real U_c = u(n, k, j, i);
            sf_p(n, k, j, i) =
                0.5 * (f(n, k, j, i) + lambda_max_i * U_c);
            sf_m(n, k, j, i) =
                0.5 * (f(n, k, j, i) - lambda_max_i * U_c);
          }
        }
      }
    }

    // Scalar split fluxes
#if NSCALARS > 0
    AA sf_s_p(NSCALARS, N3, N2, N1);
    AA sf_s_m(NSCALARS, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real util_d = w(IVX, k, j, i);
          Real W_lcl  = derived_ms(IX_LOR, k, j, i);
          Real v_d    = util_d / W_lcl;
          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);
          Real vx = w(IVX, k, j, i) / W_lcl;
          Real vy = w(IVY, k, j, i) / W_lcl;
          Real vz = w(IVZ, k, j, i) / W_lcl;
          Real v2 = gxx * vx * vx + gyy * vy * vy + gzz * vz * vz +
                    2.0 * (gxy * vx * vy + gxz * vx * vz + gyz * vy * vz);
          Real detg = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);
          Real oodetg = 1.0 / detg;
          Real guu_dd =
              Inv3MetricDiag(oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);
          Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);
          Real cs2 = derived_ms(IX_CS2, k, j, i);
          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p, &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          Real F_D = f(IDN, k, j, i);
          Real D_c = u(IDN, k, j, i);
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real Y_s = r(ns, k, j, i);
            Real f_s = Y_s * F_D;
            Real u_s = Y_s * D_c;
            sf_s_p(ns, k, j, i) =
                0.5 * (f_s + lambda_max_i * u_s);
            sf_s_m(ns, k, j, i) =
                0.5 * (f_s - lambda_max_i * u_s);
          }
        }
      }
    }
#endif

    // 2. Store raw split fluxes in ThreadCache for LO hybridization fallback
    if (cache) {
      for (int k = 0; k < N3; ++k) {
        for (int j = 0; j < N2; ++j) {
          for (int i = 0; i < N1; ++i) {
            for (int n = 0; n < NHYDRO; ++n) {
              cache->split_sf_p_[0](n, k, j, i) = sf_p(n, k, j, i);
              cache->split_sf_m_[0](n, k, j, i) = sf_m(n, k, j, i);
            }
#if NSCALARS > 0
            for (int ns = 0; ns < NSCALARS; ++ns) {
              cache->split_sf_s_p_[0](ns, k, j, i) = sf_s_p(ns, k, j, i);
              cache->split_sf_s_m_[0](ns, k, j, i) = sf_s_m(ns, k, j, i);
            }
#endif
          }
        }
      }
    }

    // 3. Apply FD correction at cell centers (HO only)
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 1; i < N1 - 1; ++i) {
          int i0 = std::max(0, std::min(N1 - 5, i - 2));
          Real f_5[5][NHYDRO];
          for (int s = 0; s < 5; ++s)
            for (int n = 0; n < NHYDRO; ++n)
              f_5[s][n] = f(n, k, j, i0 + s);
          Real phi = ComputePhi_5pt(f_5);

          for (int n = 0; n < NHYDRO; ++n) {
            Real corr_p =
                FDCorrectionCell(phi, sf_p(n,k,j,i-1), sf_p(n,k,j,i),
                                 sf_p(n,k,j,i+1));
            Real corr_m =
                FDCorrectionCell(phi, sf_m(n,k,j,i-1), sf_m(n,k,j,i),
                                 sf_m(n,k,j,i+1));
            sf_p(n, k, j, i) += corr_p;
            sf_m(n, k, j, i) += corr_m;
          }
#if NSCALARS > 0
          Real phi_scl = phi;
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real sf_5[5];
            for (int s = 0; s < 5; ++s)
              sf_5[s] = f_5[s][IDN] * r(ns, k, j, i0 + s);
            Real phi_s = ScalarPhi_5pt(sf_5);
            phi_scl = std::min(phi_scl, phi_s);
          }
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real cs_p = FDCorrectionCell(
                phi_scl, sf_s_p(ns,k,j,i-1), sf_s_p(ns,k,j,i),
                sf_s_p(ns,k,j,i+1));
            Real cs_m = FDCorrectionCell(
                phi_scl, sf_s_m(ns,k,j,i-1), sf_s_m(ns,k,j,i),
                sf_s_m(ns,k,j,i+1));
            sf_s_p(ns, k, j, i) += cs_p;
            sf_s_m(ns, k, j, i) += cs_m;
          }
#endif
        }
      }
    }

    // 4. WENO-reconstruct to X1 faces
    AA& x1flux = hflux[X1DIR];
    AA fl_(NHYDRO, pmb->nverts1);
    AA fr_(NHYDRO, pmb->nverts1);

    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        // F^+ : left-bias -> fl_(n,i) = L at face i-1/2
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX1(
              rv, sf_p, fl_, fr_, n, n, k, j, il - 1, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x1flux(n, k, j, i) += fl_(n, i);

        // F^- : right-bias -> fr_(n,i) = R at face i-1/2
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX1(
              rv, sf_m, fl_, fr_, n, n, k, j, il - 1, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x1flux(n, k, j, i) += fr_(n, i);
      }
    }

#if NSCALARS > 0
    AA& s_x1flux = sflux[X1DIR];
    AA sfl_(NSCALARS, pmb->nverts1);
    AA sfr_(NSCALARS, pmb->nverts1);
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX1(
              rv, sf_s_p, sfl_, sfr_, ns, ns, k, j, il - 1, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x1flux(ns, k, j, i) += sfl_(ns, i);

        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX1(
              rv, sf_s_m, sfl_, sfr_, ns, ns, k, j, il - 1, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x1flux(ns, k, j, i) += sfr_(ns, i);
      }
    }
#endif
  }

  // -- X2 direction --------------------------------------------------------
  if (pmb->pmy_mesh->f2) {
    const int d = 1;
    pr->SetIndicialLimitsCalculateFluxes(
        IVY, il, iu, jl, ju, kl, ku, num_enlarge_layer);

    // 1. Pre-compute
    AA f(NHYDRO, N3, N2, N1);
    AA sf_p(NHYDRO, N3, N2, N1);
    AA sf_m(NHYDRO, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real f_tmp[NHYDRO];
          PhysicalFluxPoint(d, k, j, i, pz4c, w, derived_ms, f_tmp);
          for (int n = 0; n < NHYDRO; ++n)
            f(n, k, j, i) = f_tmp[n];

          Real cs2    = derived_ms(IX_CS2, k, j, i);
          Real W_lcl  = derived_ms(IX_LOR, k, j, i);
          Real util_d = w(IVY, k, j, i);
          Real v_d    = util_d / W_lcl;

          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);

          Real vx = w(IVX, k, j, i) / W_lcl;
          Real vy = w(IVY, k, j, i) / W_lcl;
          Real vz = w(IVZ, k, j, i) / W_lcl;
          Real v2 = gxx*vx*vx + gyy*vy*vy + gzz*vz*vz +
                    2.0*(gxy*vx*vy + gxz*vx*vz + gyz*vy*vz);

          Real detg = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);
          Real oodetg = 1.0 / detg;
          Real guu_dd =
              Inv3MetricDiag(oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);

          Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);

          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p, &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          for (int n = 0; n < NHYDRO; ++n) {
            Real U_c = u(n, k, j, i);
            sf_p(n, k, j, i) =
                0.5 * (f(n, k, j, i) + lambda_max_i * U_c);
            sf_m(n, k, j, i) =
                0.5 * (f(n, k, j, i) - lambda_max_i * U_c);
          }
        }
      }
    }

#if NSCALARS > 0
    AA sf_s_p(NSCALARS, N3, N2, N1);
    AA sf_s_m(NSCALARS, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real W_lcl  = derived_ms(IX_LOR, k, j, i);
          Real util_d = w(IVY, k, j, i);
          Real v_d    = util_d / W_lcl;
          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);
          Real vx = w(IVX,k,j,i)/W_lcl, vy=w(IVY,k,j,i)/W_lcl,
               vz=w(IVZ,k,j,i)/W_lcl;
          Real v2 = gxx*vx*vx+gyy*vy*vy+gzz*vz*vz +
                    2.0*(gxy*vx*vy+gxz*vx*vz+gyz*vy*vz);
          Real detg = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);
          Real oodetg = 1.0/detg;
          Real guu_dd =
              Inv3MetricDiag(oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);
          Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);
          Real cs2 = derived_ms(IX_CS2, k, j, i);
          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p, &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          Real F_D = f(IDN, k, j, i);
          Real D_c = u(IDN, k, j, i);
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real Y_s = r(ns, k, j, i);
            Real f_s = Y_s * F_D;
            Real u_s = Y_s * D_c;
            sf_s_p(ns, k, j, i) =
                0.5 * (f_s + lambda_max_i * u_s);
            sf_s_m(ns, k, j, i) =
                0.5 * (f_s - lambda_max_i * u_s);
          }
        }
      }
    }
#endif

    // 2. Store raw split fluxes in ThreadCache for LO hybridization fallback
    if (cache) {
      for (int k = 0; k < N3; ++k) {
        for (int j = 0; j < N2; ++j) {
          for (int i = 0; i < N1; ++i) {
            for (int n = 0; n < NHYDRO; ++n) {
              cache->split_sf_p_[1](n, k, j, i) = sf_p(n, k, j, i);
              cache->split_sf_m_[1](n, k, j, i) = sf_m(n, k, j, i);
            }
#if NSCALARS > 0
            for (int ns = 0; ns < NSCALARS; ++ns) {
              cache->split_sf_s_p_[1](ns, k, j, i) = sf_s_p(ns, k, j, i);
              cache->split_sf_s_m_[1](ns, k, j, i) = sf_s_m(ns, k, j, i);
            }
#endif
          }
        }
      }
    }

    // 3. FD correction (HO only) — along j for X2 direction
    for (int k = 0; k < N3; ++k) {
      for (int i = 0; i < N1; ++i) {
        for (int j = 1; j < N2 - 1; ++j) {
          int j0 = std::max(0, std::min(N2 - 5, j - 2));
          Real f_5[5][NHYDRO];
          for (int s = 0; s < 5; ++s)
            for (int n = 0; n < NHYDRO; ++n)
              f_5[s][n] = f(n, k, j0 + s, i);
          Real phi = ComputePhi_5pt(f_5);
          for (int n = 0; n < NHYDRO; ++n) {
            sf_p(n,k,j,i) += FDCorrectionCell(
                phi, sf_p(n,k,j-1,i), sf_p(n,k,j,i), sf_p(n,k,j+1,i));
            sf_m(n,k,j,i) += FDCorrectionCell(
                phi, sf_m(n,k,j-1,i), sf_m(n,k,j,i), sf_m(n,k,j+1,i));
          }
#if NSCALARS > 0
          Real phi_scl = phi;
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real sf_5[5];
            for (int s = 0; s < 5; ++s)
              sf_5[s] = f_5[s][IDN] * r(ns, k, j0 + s, i);
            Real phi_s = ScalarPhi_5pt(sf_5);
            phi_scl = std::min(phi_scl, phi_s);
          }
          for (int ns = 0; ns < NSCALARS; ++ns) {
            sf_s_p(ns,k,j,i) += FDCorrectionCell(
                phi_scl, sf_s_p(ns,k,j-1,i), sf_s_p(ns,k,j,i),
                sf_s_p(ns,k,j+1,i));
            sf_s_m(ns,k,j,i) += FDCorrectionCell(
                phi_scl, sf_s_m(ns,k,j-1,i), sf_s_m(ns,k,j,i),
                sf_s_m(ns,k,j+1,i));
          }
#endif
        }
      }
    }

    // 4. WENO-reconstruct with swap buffers
    AA& x2flux = hflux[X2DIR];
    AA fl_(NHYDRO, NV);
    AA flb_(NHYDRO, NV);
    AA fr_(NHYDRO, NV);

    for (int k = kl; k <= ku; ++k) {
      // F^+ : prime at jl-1
      for (int n = 0; n < NHYDRO; ++n)
        pr->ReconstructFieldX2(
            rv, sf_p, fl_, fr_, n, n, k, jl - 1, il, iu);
      for (int j = jl; j <= ju; ++j) {
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX2(
              rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x2flux(n, k, j, i) += fl_(n, i);
        fl_.SwapAthenaArray(flb_);
      }
      // F^- : prime at jl-1
      for (int n = 0; n < NHYDRO; ++n)
        pr->ReconstructFieldX2(
            rv, sf_m, fl_, fr_, n, n, k, jl - 1, il, iu);
      for (int j = jl; j <= ju; ++j) {
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX2(
              rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x2flux(n, k, j, i) += fr_(n, i);
        fl_.SwapAthenaArray(flb_);
      }
    }

#if NSCALARS > 0
    AA& s_x2flux = sflux[X2DIR];
    AA sfl_(NSCALARS, NV);
    AA sflb_(NSCALARS, NV);
    AA sfr_(NSCALARS, NV);
    for (int k = kl; k <= ku; ++k) {
      for (int ns = 0; ns < NSCALARS; ++ns)
        pr->ReconstructFieldX2(
            rv, sf_s_p, sfl_, sfr_, ns, ns, k, jl - 1, il, iu);
      for (int j = jl; j <= ju; ++j) {
        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX2(
              rv, sf_s_p, sflb_, sfr_, ns, ns, k, j, il, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x2flux(ns, k, j, i) += sfl_(ns, i);
        sfl_.SwapAthenaArray(sflb_);
      }
      for (int ns = 0; ns < NSCALARS; ++ns)
        pr->ReconstructFieldX2(
            rv, sf_s_m, sfl_, sfr_, ns, ns, k, jl - 1, il, iu);
      for (int j = jl; j <= ju; ++j) {
        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX2(
              rv, sf_s_m, sflb_, sfr_, ns, ns, k, j, il, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x2flux(ns, k, j, i) += sfr_(ns, i);
        sfl_.SwapAthenaArray(sflb_);
      }
    }
#endif
  }

  // -- X3 direction ----------------------------------------------------------
  if (pmb->pmy_mesh->f3) {
    const int d = 2;
    pr->SetIndicialLimitsCalculateFluxes(
        IVZ, il, iu, jl, ju, kl, ku, num_enlarge_layer);

    // 1. Pre-compute
    AA f(NHYDRO, N3, N2, N1);
    AA sf_p(NHYDRO, N3, N2, N1);
    AA sf_m(NHYDRO, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real f_tmp[NHYDRO];
          PhysicalFluxPoint(d, k, j, i, pz4c, w, derived_ms, f_tmp);
          for (int n = 0; n < NHYDRO; ++n)
            f(n, k, j, i) = f_tmp[n];

          Real cs2    = derived_ms(IX_CS2, k, j, i);
          Real W_lcl  = derived_ms(IX_LOR, k, j, i);
          Real util_d = w(IVZ, k, j, i);
          Real v_d    = util_d / W_lcl;

          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);

          Real vx = w(IVX, k, j, i) / W_lcl;
          Real vy = w(IVY, k, j, i) / W_lcl;
          Real vz = w(IVZ, k, j, i) / W_lcl;
          Real v2 = gxx*vx*vx + gyy*vy*vy + gzz*vz*vz +
                    2.0*(gxy*vx*vy + gxz*vx*vz + gyz*vy*vz);

          Real detg = Det3Metric(gxx, gxy, gxz, gyy, gyz, gzz);
          Real oodetg = 1.0 / detg;
          Real guu_dd =
              Inv3MetricDiag(oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);

          Real alpha  = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax + d, k, j, i);

          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p, &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          for (int n = 0; n < NHYDRO; ++n) {
            Real U_c = u(n, k, j, i);
            sf_p(n, k, j, i) =
                0.5 * (f(n, k, j, i) + lambda_max_i * U_c);
            sf_m(n, k, j, i) =
                0.5 * (f(n, k, j, i) - lambda_max_i * U_c);
          }
        }
      }
    }

#if NSCALARS > 0
    AA sf_s_p(NSCALARS, N3, N2, N1);
    AA sf_s_m(NSCALARS, N3, N2, N1);
    for (int k = 0; k < N3; ++k) {
      for (int j = 0; j < N2; ++j) {
        for (int i = 0; i < N1; ++i) {
          Real W_lcl = derived_ms(IX_LOR, k, j, i);
          Real util_d = w(IVZ, k, j, i), v_d = util_d / W_lcl;
          Real gxx = pz4c->storage.adm(Z4c::I_ADM_gxx, k, j, i);
          Real gxy = pz4c->storage.adm(Z4c::I_ADM_gxy, k, j, i);
          Real gxz = pz4c->storage.adm(Z4c::I_ADM_gxz, k, j, i);
          Real gyy = pz4c->storage.adm(Z4c::I_ADM_gyy, k, j, i);
          Real gyz = pz4c->storage.adm(Z4c::I_ADM_gyz, k, j, i);
          Real gzz = pz4c->storage.adm(Z4c::I_ADM_gzz, k, j, i);
          Real vx = w(IVX,k,j,i)/W_lcl, vy=w(IVY,k,j,i)/W_lcl,
               vz=w(IVZ,k,j,i)/W_lcl;
          Real v2 = gxx*vx*vx+gyy*vy*vy+gzz*vz*vz +
                    2.0*(gxy*vx*vy+gxz*vx*vz+gyz*vy*vz);
          Real detg = Det3Metric(gxx,gxy,gxz,gyy,gyz,gzz);
          Real oodetg = 1.0/detg;
          Real guu_dd = Inv3MetricDiag(
              oodetg, gxx, gxy, gxz, gyy, gyz, gzz, d);
          Real alpha = pz4c->storage.adm(Z4c::I_ADM_alpha, k, j, i);
          Real beta_d = pz4c->storage.adm(Z4c::I_ADM_betax+d, k, j, i);
          Real cs2 = derived_ms(IX_CS2, k, j, i);
          Real lambda_p, lambda_m;
          Eigenvalues::HydroEigenvalues(
              cs2, v_d, v2, alpha, beta_d, guu_dd, &lambda_p,
              &lambda_m);
          Real lambda_max_i =
              std::max(std::abs(lambda_m), std::abs(lambda_p));

          Real F_D = f(IDN, k, j, i);
          Real D_c = u(IDN, k, j, i);
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real Y_s = r(ns, k, j, i);
            Real f_s = Y_s * F_D;
            Real u_s = Y_s * D_c;
            sf_s_p(ns, k, j, i) =
                0.5 * (f_s + lambda_max_i * u_s);
            sf_s_m(ns, k, j, i) =
                0.5 * (f_s - lambda_max_i * u_s);
          }
        }
      }
    }
#endif

    // 2. Store raw split fluxes in ThreadCache for LO hybridization fallback
    if (cache) {
      for (int k = 0; k < N3; ++k) {
        for (int j = 0; j < N2; ++j) {
          for (int i = 0; i < N1; ++i) {
            for (int n = 0; n < NHYDRO; ++n) {
              cache->split_sf_p_[2](n, k, j, i) = sf_p(n, k, j, i);
              cache->split_sf_m_[2](n, k, j, i) = sf_m(n, k, j, i);
            }
#if NSCALARS > 0
            for (int ns = 0; ns < NSCALARS; ++ns) {
              cache->split_sf_s_p_[2](ns, k, j, i) = sf_s_p(ns, k, j, i);
              cache->split_sf_s_m_[2](ns, k, j, i) = sf_s_m(ns, k, j, i);
            }
#endif
          }
        }
      }
    }

    // 3. FD correction (HO only) — along k for X3 direction
    for (int j = 0; j < N2; ++j) {
      for (int i = 0; i < N1; ++i) {
        for (int k = 1; k < N3 - 1; ++k) {
          int k0 = std::max(0, std::min(N3 - 5, k - 2));
          Real f_5[5][NHYDRO];
          for (int s = 0; s < 5; ++s)
            for (int n = 0; n < NHYDRO; ++n)
              f_5[s][n] = f(n, k0 + s, j, i);
          Real phi = ComputePhi_5pt(f_5);
          for (int n = 0; n < NHYDRO; ++n) {
            sf_p(n,k,j,i) += FDCorrectionCell(
                phi, sf_p(n,k-1,j,i), sf_p(n,k,j,i), sf_p(n,k+1,j,i));
            sf_m(n,k,j,i) += FDCorrectionCell(
                phi, sf_m(n,k-1,j,i), sf_m(n,k,j,i), sf_m(n,k+1,j,i));
          }
#if NSCALARS > 0
          Real phi_scl = phi;
          for (int ns = 0; ns < NSCALARS; ++ns) {
            Real sf_5[5];
            for (int s = 0; s < 5; ++s)
              sf_5[s] = f_5[s][IDN] * r(ns, k0 + s, j, i);
            Real phi_s = ScalarPhi_5pt(sf_5);
            phi_scl = std::min(phi_scl, phi_s);
          }
          for (int ns = 0; ns < NSCALARS; ++ns) {
            sf_s_p(ns,k,j,i) += FDCorrectionCell(
                phi_scl, sf_s_p(ns,k-1,j,i), sf_s_p(ns,k,j,i),
                sf_s_p(ns,k+1,j,i));
            sf_s_m(ns,k,j,i) += FDCorrectionCell(
                phi_scl, sf_s_m(ns,k-1,j,i), sf_s_m(ns,k,j,i),
                sf_s_m(ns,k+1,j,i));
          }
#endif
        }
      }
    }

    // 4. WENO-reconstruct X3 (j outer, k inner)
    AA& x3flux = hflux[X3DIR];
    AA fl_(NHYDRO, NV);
    AA flb_(NHYDRO, NV);
    AA fr_(NHYDRO, NV);

    for (int j = jl; j <= ju; ++j) {
      for (int n = 0; n < NHYDRO; ++n)
        pr->ReconstructFieldX3(
            rv, sf_p, fl_, fr_, n, n, kl - 1, j, il, iu);
      for (int k = kl; k <= ku; ++k) {
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX3(
              rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x3flux(n, k, j, i) += fl_(n, i);
        fl_.SwapAthenaArray(flb_);
      }
      for (int n = 0; n < NHYDRO; ++n)
        pr->ReconstructFieldX3(
            rv, sf_m, fl_, fr_, n, n, kl - 1, j, il, iu);
      for (int k = kl; k <= ku; ++k) {
        for (int n = 0; n < NHYDRO; ++n)
          pr->ReconstructFieldX3(
              rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        for (int n = 0; n < NHYDRO; ++n)
          for (int i = il; i <= iu; ++i)
            x3flux(n, k, j, i) += fr_(n, i);
        fl_.SwapAthenaArray(flb_);
      }
    }

#if NSCALARS > 0
    AA& s_x3flux = sflux[X3DIR];
    AA sfl_(NSCALARS, NV);
    AA sflb_(NSCALARS, NV);
    AA sfr_(NSCALARS, NV);
    for (int j = jl; j <= ju; ++j) {
      for (int ns = 0; ns < NSCALARS; ++ns)
        pr->ReconstructFieldX3(
            rv, sf_s_p, sfl_, sfr_, ns, ns, kl - 1, j, il, iu);
      for (int k = kl; k <= ku; ++k) {
        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX3(
              rv, sf_s_p, sflb_, sfr_, ns, ns, k, j, il, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x3flux(ns, k, j, i) += sfl_(ns, i);
        sfl_.SwapAthenaArray(sflb_);
      }
      for (int ns = 0; ns < NSCALARS; ++ns)
        pr->ReconstructFieldX3(
            rv, sf_s_m, sfl_, sfr_, ns, ns, kl - 1, j, il, iu);
      for (int k = kl; k <= ku; ++k) {
        for (int ns = 0; ns < NSCALARS; ++ns)
          pr->ReconstructFieldX3(
              rv, sf_s_m, sflb_, sfr_, ns, ns, k, j, il, iu);
        for (int ns = 0; ns < NSCALARS; ++ns)
          for (int i = il; i <= iu; ++i)
            s_x3flux(ns, k, j, i) += sfr_(ns, i);
        sfl_.SwapAthenaArray(sflb_);
      }
    }
#endif
  }
}

// ---------------------------------------------------------------------------
// Low-order fallback for split_llf hybridization.  Recovers cell-centred
// physical flux F = F^+ + F^- from the ThreadCache, computes cell-centred
// eigenvalues lambda from primitives, and reconstructs F and U to faces
// independently.  Face flux is assembled via standard LLF:
//   F_face = 0.5*(F_L + F_R) - 0.5*max(lambda_L, lambda_R)*(U_R - U_L)
// guaranteeing non-negative dissipation at every face.
// ---------------------------------------------------------------------------

void Hydro::CalculateFluxesSplitCached(
    AA& w, AA& r, FaceField& b, AA& bcc,
    AA (&lo_hflux)[3], AA (&lo_sflux)[3],
    Reconstruction::ReconstructionVariant rv_lo,
    const int num_enlarge_layer,
    ThreadCache& cache,
    const AA_B& mask)
{
    CalculateFluxesCombined(w, r, b, bcc,
                            lo_hflux, lo_sflux,
                            rv_lo, num_enlarge_layer, &cache);
}
