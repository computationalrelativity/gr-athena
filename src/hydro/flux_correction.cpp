// Flux correction:
// Removes the O(dx^2) error term involving d^3F/dx^3 from
// the flux-difference formula.
//
// A 4-point linear FD stencil on cell-centered physical fluxes computes F''
// at each face. Applied post-Riemann-solver to the face fluxes.
//
// Shared helpers (PhysicalFluxPoint, sensors, FDCorrectionCell)
// live in flux_helpers.hpp.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "hydro.hpp"
#include "flux_helpers.hpp"

// Debug:
// #ifndef CORRECTION_DIAG
// #define CORRECTION_DIAG
// #endif

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
  if (il - 3 < 0 || iu + 1 >= pmy_block->ncells1)
    return;

  for (int i = il - 1; i <= iu - 1; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k, j, i - 2 + s, pz4c, w, derived_ms, f[s]);

    if (w(IDN, k, j, i - 2) <= rho_cut || w(IDN, k, j, i - 1) <= rho_cut ||
        w(IDN, k, j, i) <= rho_cut || w(IDN, k, j, i + 1) <= rho_cut ||
        w(IDN, k, j, i + 2) <= rho_cut)
      continue;

    Real phi = ComputePhi_5pt(f);

#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k, j, i - 2 + s);
      Real phi_s = ScalarPhi_5pt(sf);
      phi = std::min(phi, phi_s);
    }
#endif
    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i + 1) -= phi * corr;
    }
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
  if (j - 3 < 0 || j + 1 >= pmy_block->ncells2)
    return;

  for (int i = il; i <= iu; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k, j - 3 + s, i, pz4c, w, derived_ms, f[s]);

    if (w(IDN, k, j - 3, i) <= rho_cut || w(IDN, k, j - 2, i) <= rho_cut ||
        w(IDN, k, j - 1, i) <= rho_cut || w(IDN, k, j, i) <= rho_cut ||
        w(IDN, k, j + 1, i) <= rho_cut)
      continue;

    Real phi = ComputePhi_5pt(f);

#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k, j - 3 + s, i);
      Real phi_s = ScalarPhi_5pt(sf);
      phi = std::min(phi, phi_s);
    }
#endif
    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i) -= phi * corr;
    }
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
  if (k - 3 < 0 || k + 1 >= pmy_block->ncells3)
    return;

  for (int i = il; i <= iu; ++i)
  {
    Real f[5][NHYDRO];
    for (int s = 0; s < 5; ++s)
      PhysicalFluxPoint(d, k - 3 + s, j, i, pz4c, w, derived_ms, f[s]);

    if (w(IDN, k - 3, j, i) <= rho_cut || w(IDN, k - 2, j, i) <= rho_cut ||
        w(IDN, k - 1, j, i) <= rho_cut || w(IDN, k, j, i) <= rho_cut ||
        w(IDN, k + 1, j, i) <= rho_cut)
      continue;

    Real phi = ComputePhi_5pt(f);

#if NSCALARS > 0
    for (int ns = 0; ns < NSCALARS; ++ns)
    {
      Real sf[5];
      for (int s = 0; s < 5; ++s)
        sf[s] = f[s][IDN] * r_scalar(ns, k - 3 + s, j, i);
      Real phi_s = ScalarPhi_5pt(sf);
      phi = std::min(phi, phi_s);
    }
#endif
    for (int n = 0; n < NHYDRO; ++n)
    {
      const Real corr = (f[1][n] - f[2][n] - f[3][n] + f[4][n]) * (1.0 / 48.0);
      hflux(n, k, j, i) -= phi * corr;
    }
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
