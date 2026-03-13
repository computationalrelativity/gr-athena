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

#if FLUID_ENABLED
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
  return within_limits;
}

void PassiveScalars::ApplySpeciesLimits(AthenaArray<Real> & z_,
                                        const int il,
                                        const int iu)
{
#if FLUID_ENABLED
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
}

void PassiveScalars::ApplySpeciesLimits(AA & z,
                                        const int i,
                                        const int j,
                                        const int k)
{
#if FLUID_ENABLED
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
