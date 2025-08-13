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

void PassiveScalars::FallbackInadmissibleScalarX_(
    AthenaArray<Real> & zl_,
    AthenaArray<Real> & zr_,
    AthenaArray<Real> & f_zl_,
    AthenaArray<Real> & f_zr_,
    const int il, const int iu, const int ivx)
{
  const int I = (ivx == 1) ? DGB_RECON_X1_OFFSET : 0;

  const bool split_lr_fallback = pmy_block->phydro->split_lr_fallback;

#ifdef DBG_FALLBACK_NO_TABLE_LIMITS

  if (!split_lr_fallback)
  {
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
  else
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        if (zl_(n,i+I) < 0)
        {
          zl_(n,i+I) = f_zl_(n,i+I);
        }
      }
    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        if (zr_(n,i) < 0)
        {
          zr_(n,i  ) = f_zr_(n,i  );
        }
      }
    }
  }

#else

  if (!split_lr_fallback)
  {
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
  else
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (!SpeciesWithinLimits(zl_,i+I))
      for (int n=0; n<NSCALARS; ++n)
      {
        zl_(n,i+I) = f_zl_(n,i+I);
      }
    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (!SpeciesWithinLimits(zr_,i))
      for (int n=0; n<NSCALARS; ++n)
      {
        zr_(n,i  ) = f_zr_(n,i  );
      }
    }
  }

#endif // DBG_FALLBACK_NO_TABLE_LIMITS
}

void PassiveScalars::FallbackInadmissibleMaskScalarX_(
  AA_B & mask_l_,
  AA_B & mask_r_,
  AthenaArray<Real> & zl_,
  AthenaArray<Real> & zr_,
  const int il, const int iu, const int ivx)
{
  const int I = (ivx == 1) ? DGB_RECON_X1_OFFSET : 0;

  const bool split_lr_fallback = pmy_block->phydro->split_lr_fallback;

#ifdef DBG_FALLBACK_NO_TABLE_LIMITS

  if (!split_lr_fallback)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        if ((zl_(n,i+I) < 0) || (zr_(n,i) < 0))
        {
          mask_l_(i+I) = false;
          mask_r_(i  ) = false;
        }
      }
    }
  }
  else
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        if (zl_(n,i+I) < 0)
        {
          mask_l_(i+I) = false;
        }
      }
    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      for (int n=0; n<NSCALARS; ++n)
      {
        if (zr_(n,i) < 0)
        {
          mask_r_(i  ) = false;
        }
      }
    }
  }

#else

  if (!split_lr_fallback)
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (!SpeciesWithinLimits(zl_,i+I) || !SpeciesWithinLimits(zr_,i))
      {
        mask_l_(i+I) = false;
        mask_r_(i  ) = false;
      }
    }
  }
  else
  {
    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (!SpeciesWithinLimits(zl_,i+I))
      {
        mask_l_(i+I) = false;
      }
    }

    #pragma omp simd
    for (int i=il; i<=iu; ++i)
    {
      if (!SpeciesWithinLimits(zr_,i))
      {
        mask_r_(i  ) = false;
      }
    }
  }

#endif // DBG_FALLBACK_NO_TABLE_LIMITS
}

void PassiveScalars::CalculateFluxes_STS() {
  AddDiffusionFluxes();
}


void PassiveScalars::ComputeUpwindFlux(const int k, const int j, const int il,
                                       const int iu, // CoordinateDirection dir,
                                       AthenaArray<Real> &rl, AthenaArray<Real> &rr, // 2D
                                       AthenaArray<Real> &mass_flx,  // 3D
                                       AthenaArray<Real> &flx_out) { // 4D
  if (!pmy_block->precon->xorder_upwind_scalars)
  {
    return;
  }

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
