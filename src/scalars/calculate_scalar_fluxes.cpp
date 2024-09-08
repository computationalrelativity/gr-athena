//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_scalar_fluxes.cpp
//  \brief Calculate passive scalar fluxes

// C headers

// C++ headers
#include <algorithm>   // min,max

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../hydro/hydro.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "scalars.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// For reduction of reconstruction order
bool PassiveScalars::SpeciesWithinLimits(AthenaArray<Real> & z_, const int i)
{
  bool within_limits = true;

#if USETM
  EquationOfState *peos = pmy_block->peos;

  for (int n=0; n<NSCALARS; ++n)
  {
    const Real fr_max = peos->GetEOS().GetMaximumSpeciesFraction(n);
    const Real fr_min = peos->GetEOS().GetMinimumSpeciesFraction(n);

    within_limits = (fr_min <= z_(n,i)) &&
                    (z_(n,i) <= fr_max) &&
                    within_limits;

    if (!within_limits)
      return false;
  }
#endif
  // BD: TODO - worth to fix if not USETM?
  return within_limits;
}

void PassiveScalars::ApplySpeciesLimits(AthenaArray<Real> & z_,
                                        const int il,
                                        const int iu)
{
#if USETM
  EquationOfState *peos = pmy_block->peos;

  Real Y[MAX_SPECIES] = {0.0};
  for (int i=il; i<=iu; ++i)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      Y[n] = z_(n,i);
    }

    peos->GetEOS().ApplySpeciesLimits(Y);

    for (int n=0; n<NSCALARS; ++n)
    {
      z_(n,i) = Y[n];
    }
  }
#endif
  // BD: TODO - worth to fix if not USETM?

}

void PassiveScalars::ApplySpeciesLimits(AA & z,
                                        const int i,
                                        const int j,
                                        const int k)
{
#if USETM
  EquationOfState *peos = pmy_block->peos;

  Real Y[MAX_SPECIES] = {0.0};
  for (int n=0; n<NSCALARS; ++n)
  {
    Y[n] = z(n,k,j,i);
  }

  peos->GetEOS().ApplySpeciesLimits(Y);

  for (int n=0; n<NSCALARS; ++n)
  {
    z(n,k,j,i) = Y[n];
  }
#endif
  // BD: TODO - worth to fix if not USETM?
}

#ifdef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX1_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX1_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  static const int I = DGB_RECON_X1_OFFSET;
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      if ((zl_(n,i+I) < 0) || (zr_(n,i) < 0))
      {
        zl_(n,i+I) = f_zl_(n,i+I);
        zr_(n,i  ) = f_zr_(n,i  );
      }
    }
  }
}

#ifdef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX2_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX2_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      if ((zl_(n,i) < 0) || (zr_(n,i) < 0))
      {
        zl_(n,i) = f_zl_(n,i);
        zr_(n,i) = f_zr_(n,i);
      }
    }
  }
}

#ifdef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX3_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX3_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    for (int n=0; n<NSCALARS; ++n)
    {
      if ((zl_(n,i) < 0) || (zr_(n,i) < 0))
      {
        zl_(n,i) = f_zl_(n,i);
        zr_(n,i) = f_zr_(n,i);
      }
    }
  }
}

#ifndef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX1_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX1_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  static const int I = DGB_RECON_X1_OFFSET;
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if (!SpeciesWithinLimits(zl_,i+I) || !SpeciesWithinLimits(zr_,i))
    for (int n=0; n<NSCALARS; ++n)
    {
      zl_(n,i+I) = f_zl_(n,i+I);
      zr_(n,i  ) = f_zr_(n,i  );
    }
  }
}

#ifndef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX2_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX2_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if (!SpeciesWithinLimits(zl_,i) || !SpeciesWithinLimits(zr_,i))
    for (int n=0; n<NSCALARS; ++n)
    {
      zl_(n,i) = f_zl_(n,i);
      zr_(n,i) = f_zr_(n,i);
    }
  }
}

#ifndef DBG_FALLBACK_NO_TABLE_LIMITS
void PassiveScalars::FallbackInadmissibleScalarX3_(
#else
void PassiveScalars::FallbackInadmissibleScalarTableLimitsX3_(
#endif
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu)
{
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    if (!SpeciesWithinLimits(zl_,i) || !SpeciesWithinLimits(zr_,i))
    for (int n=0; n<NSCALARS; ++n)
    {
      zl_(n,i) = f_zl_(n,i);
      zr_(n,i) = f_zr_(n,i);
    }
  }
}

void PassiveScalars::CalculateFluxes(AthenaArray<Real> &r, const int order)
{
#ifdef DBG_COMBINED_HYDPA
  // Taken care of in Hydro:: currently
  return;
#endif

  MeshBlock *pmb = pmy_block;
  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;
  ReconstructionVariant r_rv = pr->xorder_style_fb;

  Hydro &hyd = *(pmb->phydro);
  int il, iu, jl, ju, kl, ku;
  AthenaArray<Real> mass_flux;

  //--------------------------------------------------------------------------------------
  // i-direction
  AthenaArray<Real> &x1flux = s_flux[X1DIR];
  mass_flux.InitWithShallowSlice(hyd.flux[X1DIR], 4, IDN, 1);

  pr->SetIndicialLimitsCalculateFluxes(IVX, il, iu, jl, ju, kl, ku);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    pr->ReconstructPassiveScalarsX1_(rv, r, rl_, rr_,
                                     k, j, il, iu);

    if (pr->xorder_use_fb)
    {
      pr->ReconstructPassiveScalarsX1_(r_rv, r, r_rl_, r_rr_,
                                       k, j, il, iu);
      FallbackInadmissibleScalarX1_(rl_, rr_, r_rl_, r_rr_,
                                    il, iu);
    }
    else
    {
      // Floor here (as needed, always attempted, Cf. CalculateFluxes in Hydro)
      ApplySpeciesLimits(rl_, il, iu);
      ApplySpeciesLimits(rr_, il, iu);
    }

    ComputeUpwindFlux(k, j, il, iu, rl_, rr_, mass_flux, x1flux);
  }

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    AthenaArray<Real> &x2flux = s_flux[X2DIR];
    mass_flux.InitWithShallowSlice(hyd.flux[X2DIR], 4, IDN, 1);

    pr->SetIndicialLimitsCalculateFluxes(IVY, il, iu, jl, ju, kl, ku);

    for (int k=kl; k<=ku; ++k)
    {
      pr->ReconstructPassiveScalarsX2_(rv, r, rl_, rr_,
                                       k, jl-1, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPassiveScalarsX2_(r_rv, r, r_rl_, r_rr_,
                                         k, jl-1, il, iu);
        FallbackInadmissibleScalarX2_(rl_, rr_, r_rl_, r_rr_,
                                      il, iu);
      }
      else
      {
        // Floor here (as needed, Cf. CalculateFluxes in Hydro)
        ApplySpeciesLimits(rl_, il, iu);
        ApplySpeciesLimits(rr_, il, iu);
      }


      for (int j=jl; j<=ju; ++j)
      {
        pr->ReconstructPassiveScalarsX2_(rv, r, rlb_, rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPassiveScalarsX2_(r_rv, r, r_rlb_, r_rr_,
                                           k, j, il, iu);
          FallbackInadmissibleScalarX2_(rlb_, rr_, r_rlb_, r_rr_,
                                        il, iu);
        }
        else
        {
          // Floor here (as needed, Cf. CalculateFluxes in Hydro)
          ApplySpeciesLimits(rlb_, il, iu);
          ApplySpeciesLimits(rr_,  il, iu);
        }

        ComputeUpwindFlux(k, j, il, iu, rl_, rr_, mass_flux, x2flux);

        rl_.SwapAthenaArray(rlb_);
        if (pr->xorder_use_fb)
        {
          r_rl_.SwapAthenaArray(r_rlb_);
        }
      }

    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    AthenaArray<Real> &x3flux = s_flux[X3DIR];
    mass_flux.InitWithShallowSlice(hyd.flux[X3DIR], 4, IDN, 1);

    pr->SetIndicialLimitsCalculateFluxes(IVZ, il, iu, jl, ju, kl, ku);

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional
      pr->ReconstructPassiveScalarsX3_(rv, r, rl_, rr_,
                                       kl-1, j, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPassiveScalarsX3_(r_rv, r, r_rl_, r_rr_,
                                         kl-1, j, il, iu);
        FallbackInadmissibleScalarX3_(rl_, rr_, r_rl_, r_rr_,
                                      il, iu);
      }
      else
      {
        // Floor here (as needed, Cf. CalculateFluxes in Hydro)
        ApplySpeciesLimits(rl_, il, iu);
        ApplySpeciesLimits(rr_, il, iu);
      }


      for (int k=kl; k<=ku; ++k)
      {
        pr->ReconstructPassiveScalarsX3_(rv, r, rlb_, rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPassiveScalarsX3_(r_rv, r, r_rlb_, r_rr_,
                                           k, j, il, iu);
          FallbackInadmissibleScalarX3_(rlb_, rr_, r_rlb_, r_rr_,
                                        il, iu);
        }
        else
        {
          // Floor here (as needed, Cf. CalculateFluxes in Hydro)
          ApplySpeciesLimits(rlb_, il, iu);
          ApplySpeciesLimits(rr_,  il, iu);
        }

        ComputeUpwindFlux(k, j, il, iu, rl_, rr_, mass_flux, x3flux);

        rl_.SwapAthenaArray(rlb_);
        if (pr->xorder_use_fb)
        {
          r_rl_.SwapAthenaArray(r_rlb_);
        }
      }

    }
  }

  return;
}


void PassiveScalars::CalculateFluxes_STS() {
  AddDiffusionFluxes();
}


void PassiveScalars::ComputeUpwindFlux(const int k, const int j, const int il,
                                       const int iu, // CoordinateDirection dir,
                                       AthenaArray<Real> &rl, AthenaArray<Real> &rr, // 2D
                                       AthenaArray<Real> &mass_flx,  // 3D
                                       AthenaArray<Real> &flx_out) { // 4D
  const int nu = NSCALARS - 1;

  for (int n=0; n<=nu; n++) {
#pragma omp simd
    for (int i=il; i<=iu; i++) {
      Real fluid_flx = mass_flx(k,j,i);
      if (fluid_flx >= 0.0)
        flx_out(n,k,j,i) = fluid_flx*rl(n,i);
      else
        flx_out(n,k,j,i) = fluid_flx*rr(n,i);
    }
  }
  return;
}
