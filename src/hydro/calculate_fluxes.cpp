//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_fluxes.cpp
//  \brief Calculate hydro/MHD fluxes

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

#include "../utils/linear_algebra.hpp"
#include <ostream>

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


void Hydro::CalculateFluxes(AthenaArray<Real> &w, FaceField &b,
                            AthenaArray<Real> &bcc, const int order)
{
  if (flux_reconstruction)
  {
    CalculateFluxes_FluxReconstruction(w, b, bcc, order);
    return;
  }

#if (NSCALARS > 0) & defined(DBG_COMBINED_HYDPA)
  SetAtmMask(pmy_block->peos->GetEOS().GetDensityFloor(),
            w, atm_mask);
  //SetAtmMask(pmy_block->peos->GetEOS().GetDensityFloor(),
           // w, atm_mask);
#if EFL_ENABLED
  AthenaArray<Real> &x1flux_HO = flux_HO[X1DIR];
  AthenaArray<Real> &x2flux_HO = flux_HO[X2DIR];
  AthenaArray<Real> &x3flux_HO = flux_HO[X3DIR];
  RusanovFlux(w,u,x1flux_HO,x2flux_HO,x3flux_HO);
#endif

#ifdef DBG_COMBINED_HYDPA
  CalculateFluxesCombined(w,b,bcc,order);
  return;
#endif

  MeshBlock *pmb = pmy_block;
  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;
  ReconstructionVariant r_rv = pr->xorder_style_fb;

  int il, iu, jl, ju, kl, ku;

#if MAGNETIC_FIELDS_ENABLED
  // used only to pass to (up-to) 2x RiemannSolver() calls per dimension:
  // x1:
  AthenaArray<Real> &b1 = b.x1f, &w_x1f = pmb->pfield->wght.x1f,
                  &e3x1 = pmb->pfield->e3_x1f, &e2x1 = pmb->pfield->e2_x1f;
  // x2:
  AthenaArray<Real> &b2 = b.x2f, &w_x2f = pmb->pfield->wght.x2f,
                  &e1x2 = pmb->pfield->e1_x2f, &e3x2 = pmb->pfield->e3_x2f;
  // x3:
  AthenaArray<Real> &b3 = b.x3f, &w_x3f = pmb->pfield->wght.x3f,
                  &e1x3 = pmb->pfield->e1_x3f, &e2x3 = pmb->pfield->e2_x3f;
#endif

  //---------------------------------------------------------------------------
  // i-direction
  AthenaArray<Real> &x1flux = flux[X1DIR];
  pr->SetIndicialLimitsCalculateFluxes(IVX, il, iu, jl, ju, kl, ku);
#if EFL_ENABLED
  AthenaArray<Real> &x1flux_LO = flux_LO[X1DIR];
  AthenaArray<Real> &x1efl = ef_limiter[X1DIR];
#endif //EFL_ENABLED

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    pr->ReconstructPrimitivesX1_(rv, w, wl_, wr_,
                                 k, j, il, iu);
    pr->ReconstructMagneticFieldX1_(rv, bcc, wl_, wr_,
                                    k, j, il, iu);

    if (pr->xorder_use_fb)
    {
      pr->ReconstructPrimitivesX1_(r_rv, w, r_wl_, r_wr_,
                                   k, j, il, iu);
      pr->ReconstructMagneticFieldX1_(r_rv, bcc, r_wl_, r_wr_,
                                      k, j, il, iu);

      FallbackInadmissiblePrimitiveX1_(wl_, wr_, r_wl_, r_wr_,
                                       il, iu);
    }
    else
    {
#if USETM
      PassiveScalars *ps = pmy_block->pscalars;
      pr->ReconstructPassiveScalarsX1_(ReconstructionVariant::donate,
                                       ps->r, rl_, rr_,
                                       k, j, il, iu);
      FloorPrimitiveX1_(wl_, wr_, rl_, rr_,
                        k, j, il, iu);
#else
      FloorPrimitiveX1_(wl_, wr_,
                        k, j, il, iu);
#endif
    }

    pmb->pcoord->CenterWidth1(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED

#if EFL_ENABLED //EFL_ENABLED

    RiemannSolver(k, j, il, iu, IVX, wl_, wr_, x1flux_LO, dxw_);
    CombineFluxes(k, j, il, iu,IVX,x1efl,x1flux_HO,x1flux_LO,x1flux );

#else //EFL not ENABLED

    RiemannSolver(k, j, il, iu, IVX, wl_, wr_, x1flux, dxw_);

#endif// EFL (HYDRO)

#else
    // x1flux(IBY) = (v1*b2 - v2*b1) = -EMFZ
    // x1flux(IBZ) = (v1*b3 - v3*b1) =  EMFY
    RiemannSolver(k, j, il, iu, IVX, b1, wl_, wr_,
                  x1flux, e3x1, e2x1, w_x1f, dxw_);
#endif
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    AthenaArray<Real> &x2flux = flux[X2DIR];
    pr->SetIndicialLimitsCalculateFluxes(IVY, il, iu, jl, ju, kl, ku);
#if EFL_ENABLED
    AthenaArray<Real> &x2flux_LO = flux_LO[X2DIR];
    AthenaArray<Real> &x2efl = ef_limiter[X2DIR];
#endif //EFL_ENABLED
    for (int k=kl; k<=ku; ++k)
    {
      pr->ReconstructPrimitivesX2_(rv, w, wl_, wr_,
                                   k, jl-1, il, iu);
      pr->ReconstructMagneticFieldX2_(rv, bcc, wl_, wr_,
                                      k, jl-1, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX2_(r_rv, w, r_wl_, r_wr_,
                                     k, jl-1, il, iu);
        pr->ReconstructMagneticFieldX2_(r_rv, bcc, r_wl_, r_wr_,
                                        k, jl-1, il, iu);

        FallbackInadmissiblePrimitiveX2_(wl_, wr_, r_wl_, r_wr_,
                                         il, iu);
      }
      else
      {
#if USETM
        PassiveScalars *ps = pmy_block->pscalars;
        pr->ReconstructPassiveScalarsX2_(ReconstructionVariant::donate,
                                         ps->r, rl_, rr_,
                                         k, jl-1, il, iu);
        FloorPrimitiveX2_(wl_, wr_, rl_, rr_,
                          k, jl-1, il, iu);
#else
        FloorPrimitiveX2_(wl_, wr_,
                          k, jl-1, il, iu);
#endif
      }


      for (int j=jl; j<=ju; ++j)
      {

        pr->ReconstructPrimitivesX2_(rv, w, wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX2_(rv, bcc, wlb_, wr_,
                                        k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX2_(r_rv, w, r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX2_(r_rv, bcc, r_wlb_, r_wr_,
                                          k, j, il, iu);

          FallbackInadmissiblePrimitiveX2_(wlb_, wr_, r_wlb_, r_wr_,
                                           il, iu);
        }
        else
        {
#if USETM
          PassiveScalars *ps = pmy_block->pscalars;
          pr->ReconstructPassiveScalarsX2_(ReconstructionVariant::donate,
                                           ps->r, rlb_, rr_,
                                           k, j, il, iu);
          FloorPrimitiveX2_(wlb_, wr_, rlb_, rr_, k, j, il, iu);
#else
          FloorPrimitiveX2_(wlb_, wr_, k, j, il, iu);
#endif
        }

        pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED

#if EFL_ENABLED //EFL_ENABLED

        RiemannSolver(k, j, il, iu, IVY, wl_, wr_, x2flux_LO, dxw_);
        CombineFluxes(k, j, il, iu,IVY,x2efl,x2flux_HO,x2flux_LO,x2flux );

#else //EFL not ENABLED

        RiemannSolver(k, j, il, iu, IVY, wl_, wr_,
                      x2flux, dxw_);

#endif

#else
        // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
        // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ
        RiemannSolver(k, j, il, iu, IVY, b2, wl_, wr_,
                      x2flux, e1x2, e3x2, w_x2f, dxw_);
#endif

        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
        }
        else
        {
#if USETM
          rl_.SwapAthenaArray(rlb_);
#endif
        }
      }
    }
  }

  //---------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    AthenaArray<Real> &x3flux = flux[X3DIR];
    pr->SetIndicialLimitsCalculateFluxes(IVZ, il, iu, jl, ju, kl, ku);
#if EFL_ENABLED
    AthenaArray<Real> &x3flux_LO = flux_LO[X3DIR];
    AthenaArray<Real> &x3efl = ef_limiter[X3DIR];
#endif //EFL_ENABLED
    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional
      pr->ReconstructPrimitivesX3_(rv, w, wl_, wr_,
                                   kl-1, j, il, iu);
      pr->ReconstructMagneticFieldX3_(rv, bcc, wl_, wr_,
                                      kl-1, j, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX3_(r_rv, w, r_wl_, r_wr_,
                                     kl-1, j, il, iu);
        pr->ReconstructMagneticFieldX3_(r_rv, bcc, r_wl_, r_wr_,
                                        kl-1, j, il, iu);

        FallbackInadmissiblePrimitiveX3_(wl_, wr_, r_wl_, r_wr_,
                                         il, iu);
      }
      else
      {
#if USETM
        PassiveScalars *ps = pmy_block->pscalars;
        pr->ReconstructPassiveScalarsX3_(ReconstructionVariant::donate,
                                         ps->r, rl_, rr_,
                                         kl-1, j, il, iu);
        FloorPrimitiveX3_(wl_, wr_, rl_, rr_,
                          kl-1, j, il, iu);
#else
        FloorPrimitiveX3_(wl_, wr_, kl-1, j, il, iu);
#endif
      }


      for (int k=kl; k<=ku; ++k)
      {
        pr->ReconstructPrimitivesX3_(rv, w, wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX3_(rv, bcc, wlb_, wr_,
                                        k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX3_(r_rv, w, r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX3_(r_rv, bcc, r_wlb_, r_wr_,
                                          k, j, il, iu);

          FallbackInadmissiblePrimitiveX3_(wlb_, wr_, r_wlb_, r_wr_,
                                           il, iu);
        }
        else
        {
#if USETM
          PassiveScalars *ps = pmy_block->pscalars;
          pr->ReconstructPassiveScalarsX3_(ReconstructionVariant::donate,
                                           ps->r, rlb_, rr_,
                                           k, j, il, iu);
          FloorPrimitiveX3_(wlb_, wr_, rlb_, rr_,
                            k, j, il, iu);
#else
          FloorPrimitiveX3_(wlb_, wr_,
                            k, j, il, iu);
#endif
        }

        pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED  // Hydro:

#if EFL_ENABLED //EFL_ENABLED

        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_, x3flux_LO, dxw_);
        CombineFluxes(k, j, il, iu,IVZ,x3efl,x3flux_HO,x3flux_LO,x3flux );

#else //EFL not ENABLED

        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_,
                      x3flux, dxw_);
#endif //EFL ENABLED

#else
        // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
        // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX
        RiemannSolver(k, j, il, iu, IVZ, b3, wl_, wr_,
                      x3flux, e2x3, e1x3, w_x3f, dxw_);
#endif
        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
        }
        else
        {
#if USETM
          rl_.SwapAthenaArray(rlb_);
#endif
        }
      }
    }
  }

  return;
}

void Hydro::CalculateFluxesCombined(AthenaArray<Real> &w,
                                    FaceField &b,
                                    AthenaArray<Real> &bcc,
                                    const int order)
{
  MeshBlock *pmb = pmy_block;

  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;
  ReconstructionVariant r_rv = pr->xorder_style_fb;

  ReconstructionVariant p_rv = pr->xorder_style;
  ReconstructionVariant p_r_rv = pr->xorder_style;

  PassiveScalars *ps = pmb->pscalars;

  // For passive-scalar reconstruction
  AthenaArray<Real> mass_flux;
  // AthenaArray<Real> &r = pmb->pscalars->r;

  AA p;
  if (pr->xorder_use_cons_passive)
  {
    p.InitWithShallowSlice(pmb->pscalars->s, 4, 0, NSCALARS);
  }
  else
  {
    p.InitWithShallowSlice(pmb->pscalars->r, 4, 0, NSCALARS);
  }

  int il, iu, jl, ju, kl, ku;

#if MAGNETIC_FIELDS_ENABLED
  // used only to pass to (up-to) 2x RiemannSolver() calls per dimension:
  // x1:
  AthenaArray<Real> &b1 = b.x1f, &w_x1f = pmb->pfield->wght.x1f,
                  &e3x1 = pmb->pfield->e3_x1f, &e2x1 = pmb->pfield->e2_x1f;
  // x2:
  AthenaArray<Real> &b2 = b.x2f, &w_x2f = pmb->pfield->wght.x2f,
                  &e1x2 = pmb->pfield->e1_x2f, &e3x2 = pmb->pfield->e3_x2f;
  // x3:
  AthenaArray<Real> &b3 = b.x3f, &w_x3f = pmb->pfield->wght.x3f,
                  &e1x3 = pmb->pfield->e1_x3f, &e2x3 = pmb->pfield->e2_x3f;
#endif

  //---------------------------------------------------------------------------
  // i-direction
  AthenaArray<Real> &x1flux = flux[X1DIR];
  AthenaArray<Real> &s_x1flux = pmb->pscalars->s_flux[X1DIR];
  mass_flux.InitWithShallowSlice(flux[X1DIR], 4, IDN, 1);

  pr->SetIndicialLimitsCalculateFluxes(IVX, il, iu, jl, ju, kl, ku);

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    pr->ReconstructPrimitivesX1_(rv, w,
                                 wl_, wr_,
                                 k, j, il, iu);
    pr->ReconstructMagneticFieldX1_(rv, bcc,
                                    wl_, wr_,
                                    k, j, il, iu);
    pr->ReconstructPassiveScalarsX1_(p_rv, p,
                                     ps->rl_, ps->rr_,
                                     k, j, il, iu);

    if (pr->xorder_use_cons_passive)
    for (int n=0; n<NSCALARS; ++n)
    for (int i=il; i<=iu; ++i)
    {
      ps->rl_(n,i) /= wl_(IDN,i);
      ps->rr_(n,i) /= wr_(IDN,i);
    }

    if (pr->xorder_use_auxiliaries)
    {
      pr->ReconstructHydroAuxiliariesX1_(rv, derived_ms,
                                         al_, ar_,
                                         k, j, il, iu);
    }

    if (pr->xorder_use_fb)
    {
      pr->ReconstructPrimitivesX1_(r_rv, w, r_wl_, r_wr_,
                                   k, j, il, iu);
      pr->ReconstructMagneticFieldX1_(r_rv, bcc, r_wl_, r_wr_,
                                      k, j, il, iu);
      pr->ReconstructPassiveScalarsX1_(p_r_rv, p,
                                       ps->r_rl_, ps->r_rr_,
                                       k, j, il, iu);

      if (pr->xorder_use_auxiliaries)
      {
        pr->ReconstructHydroAuxiliariesX1_(r_rv, derived_ms,
                                           r_al_, r_ar_,
                                           k, j, il, iu);
      }

      if (pr->xorder_use_fb_mask)
      {
        mask_l_.Fill(true);
        mask_r_.Fill(true);

        FallbackInadmissibleMaskPrimitiveX_(mask_l_, mask_r_,
                                            wl_, wr_,
                                            il, iu, IVX);
        ps->FallbackInadmissibleMaskScalarX_(mask_l_, mask_r_,
                                             ps->rl_, ps->rr_,
                                             il, iu, IVX);

        FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                   wl_, wr_,
                                   r_wl_, r_wr_,
                                   0, NWAVE, il, iu, IVX);

        FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                   ps->rl_, ps->rr_,
                                   ps->r_rl_, ps->r_rr_,
                                   0, NSCALARS, il, iu, IVX);

      }
      else
      {
        FallbackInadmissiblePrimitiveX1_(wl_, wr_,
                                         r_wl_, r_wr_,
                                         il, iu);
        ps->FallbackInadmissibleScalarX_(ps->rl_, ps->rr_,
                                         ps->r_rl_, ps->r_rr_,
                                         il, iu, IVX);
      }

      if (pr->xorder_limit_species)
      {
        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);
      }
    }
    else
    {
      if (pr->xorder_limit_species)
      {
        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);
      }

#if USETM
      FloorPrimitiveX1_(wl_, wr_,
                        ps->rl_, ps->rr_,
                        k, j, il, iu);
#else
      FloorPrimitiveX1_(wl_, wr_,
                        k, j, il, iu);
#endif

      if (pr->xorder_use_auxiliaries)
        LimitAuxiliariesX1_(al_, ar_, il, iu);

    }

    pmb->pcoord->CenterWidth1(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED
    RiemannSolver(k, j, il, iu, IVX, wl_, wr_, x1flux, dxw_);
#else
    // x1flux(IBY) = (v1*b2 - v2*b1) = -EMFZ
    // x1flux(IBZ) = (v1*b3 - v3*b1) =  EMFY
    RiemannSolver(k, j, il, iu, IVX, b1, wl_, wr_,
                  x1flux, e3x1, e2x1, w_x1f, dxw_);
#endif

    ps->ComputeUpwindFlux(k, j, il, iu,
                          ps->rl_, ps->rr_,
                          mass_flux, s_x1flux);

    if (pr->xorder_use_fb && pr->xorder_use_fb_mask)
    {
      // mask_l_.Fill(true);
      // mask_r_.Fill(true);
    }
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    AthenaArray<Real> &x2flux = flux[X2DIR];
    AthenaArray<Real> &s_x2flux = pmb->pscalars->s_flux[X2DIR];
    mass_flux.InitWithShallowSlice(flux[X2DIR], 4, IDN, 1);

    pr->SetIndicialLimitsCalculateFluxes(IVY, il, iu, jl, ju, kl, ku);

    for (int k=kl; k<=ku; ++k)
    {
      pr->ReconstructPrimitivesX2_(rv, w,
                                   wl_, wr_,
                                   k, jl-1, il, iu);
      pr->ReconstructMagneticFieldX2_(rv, bcc,
                                      wl_, wr_,
                                      k, jl-1, il, iu);
      pr->ReconstructPassiveScalarsX2_(p_rv, p,
                                       ps->rl_, ps->rr_,
                                       k, jl-1, il, iu);

      if (pr->xorder_use_cons_passive)
      for (int n=0; n<NSCALARS; ++n)
      for (int i=il; i<=iu; ++i)
      {
        ps->rl_(n,i) /= wl_(IDN,i);
        ps->rr_(n,i) /= wr_(IDN,i);
      }

      if (pr->xorder_use_auxiliaries)
      {
        pr->ReconstructHydroAuxiliariesX2_(rv, derived_ms,
                                           al_, ar_,
                                           k, jl-1, il, iu);
      }

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX2_(r_rv, w,
                                     r_wl_, r_wr_,
                                     k, jl-1, il, iu);
        pr->ReconstructMagneticFieldX2_(r_rv, bcc,
                                        r_wl_, r_wr_,
                                        k, jl-1, il, iu);
        pr->ReconstructPassiveScalarsX2_(p_r_rv, p,
                                         ps->r_rl_, ps->r_rr_,
                                         k, jl-1, il, iu);

        if (pr->xorder_use_auxiliaries)
        {
          pr->ReconstructHydroAuxiliariesX2_(r_rv, derived_ms,
                                             r_al_, r_ar_,
                                             k, jl-1, il, iu);
        }

        if (pr->xorder_use_fb_mask)
        {
          mask_l_.Fill(true);
          mask_r_.Fill(true);

          FallbackInadmissibleMaskPrimitiveX_(mask_l_, mask_r_,
                                              wl_, wr_,
                                              il, iu, IVY);
          ps->FallbackInadmissibleMaskScalarX_(mask_l_, mask_r_,
                                               ps->rl_, ps->rr_,
                                               il, iu, IVY);

          FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                     wl_, wr_,
                                     r_wl_, r_wr_,
                                     0, NWAVE, il, iu, IVY);

          FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                     ps->rl_, ps->rr_,
                                     ps->r_rl_, ps->r_rr_,
                                     0, NSCALARS, il, iu, IVY);

        }
        else
        {
          FallbackInadmissiblePrimitiveX2_(wl_, wr_,
                                           r_wl_, r_wr_,
                                           il, iu);
          ps->FallbackInadmissibleScalarX_(ps->rl_, ps->rr_,
                                           ps->r_rl_, ps->r_rr_,
                                           il, iu, IVY);
        }

        if (pr->xorder_limit_species)
        {
          ps->ApplySpeciesLimits(ps->rl_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_, il, iu);
        }
      }
      else
      {
        if (pr->xorder_limit_species)
        {
          ps->ApplySpeciesLimits(ps->rl_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_, il, iu);
        }

#if USETM
        FloorPrimitiveX2_(wl_, wr_,
                          ps->rl_, ps->rr_,
                          k, jl-1, il, iu);
#else
        FloorPrimitiveX2_(wl_, wr_,
                          k, jl-1, il, iu);
#endif

        if (pr->xorder_use_auxiliaries)
          LimitAuxiliariesX2_(al_, ar_, il, iu);

      }


      for (int j=jl; j<=ju; ++j)
      {

        pr->ReconstructPrimitivesX2_(rv, w,
                                     wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX2_(rv, bcc,
                                        wlb_, wr_,
                                        k, j, il, iu);
        pr->ReconstructPassiveScalarsX2_(p_rv, p,
                                         ps->rlb_, ps->rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_cons_passive)
        for (int n=0; n<NSCALARS; ++n)
        for (int i=il; i<=iu; ++i)
        {
          ps->rlb_(n,i) /= wlb_(IDN,i);
          ps->rr_(n,i) /= wr_(IDN,i);
        }

        if (pr->xorder_use_auxiliaries)
        {
          pr->ReconstructHydroAuxiliariesX2_(rv, derived_ms,
                                             alb_, ar_,
                                             k, j, il, iu);
        }

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX2_(r_rv, w,
                                       r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX2_(r_rv, bcc,
                                          r_wlb_, r_wr_,
                                          k, j, il, iu);
          pr->ReconstructPassiveScalarsX2_(p_r_rv, p,
                                           ps->r_rlb_, ps->r_rr_,
                                           k, j, il, iu);

          if (pr->xorder_use_auxiliaries)
          {
            pr->ReconstructHydroAuxiliariesX2_(r_rv, derived_ms,
                                               r_alb_, r_ar_,
                                               k, j, il, iu);
          }

          if (pr->xorder_use_fb_mask)
          {
            mask_lb_.Fill(true);
            mask_r_.Fill(true);

            FallbackInadmissibleMaskPrimitiveX_(mask_lb_, mask_r_,
                                                wlb_, wr_,
                                                il, iu, IVY);
            ps->FallbackInadmissibleMaskScalarX_(mask_lb_, mask_r_,
                                                 ps->rlb_, ps->rr_,
                                                 il, iu, IVY);

            FallbackInadmissibleMaskX_(mask_lb_, mask_r_,
                                       wlb_, wr_,
                                       r_wlb_, r_wr_,
                                       0, NWAVE, il, iu, IVY);

            FallbackInadmissibleMaskX_(mask_lb_, mask_r_,
                                       ps->rlb_, ps->rr_,
                                       ps->r_rlb_, ps->r_rr_,
                                       0, NSCALARS, il, iu, IVY);
          }
          else
          {
            FallbackInadmissiblePrimitiveX2_(wlb_, wr_,
                                             r_wlb_, r_wr_,
                                             il, iu);
            ps->FallbackInadmissibleScalarX_(ps->rlb_, ps->rr_,
                                             ps->r_rlb_, ps->r_rr_,
                                             il, iu, IVY);
          }

          if (pr->xorder_limit_species)
          {
            ps->ApplySpeciesLimits(ps->rlb_, il, iu);
            ps->ApplySpeciesLimits(ps->rr_, il, iu);
          }
        }
        else
        {
          if (pr->xorder_limit_species)
          {
            ps->ApplySpeciesLimits(ps->rlb_, il, iu);
            ps->ApplySpeciesLimits(ps->rr_, il, iu);
          }

#if USETM
          FloorPrimitiveX2_(wlb_, wr_,
                            ps->rlb_, ps->rr_,
                            k, j, il, iu);
#else
          FloorPrimitiveX2_(wlb_, wr_, k, j, il, iu);
#endif

          if (pr->xorder_use_auxiliaries)
            LimitAuxiliariesX2_(alb_, ar_, il, iu);

        }

        pmb->pcoord->CenterWidth2(k, j, il, iu, dxw_);
#if !MAGNETIC_FIELDS_ENABLED
        RiemannSolver(k, j, il, iu, IVY, wl_, wr_,
                      x2flux, dxw_);
#else
        // flx(IBY) = (v2*b3 - v3*b2) = -EMFX
        // flx(IBZ) = (v2*b1 - v1*b2) =  EMFZ
        RiemannSolver(k, j, il, iu, IVY, b2, wl_, wr_,
                      x2flux, e1x2, e3x2, w_x2f, dxw_);
#endif

        ps->ComputeUpwindFlux(k, j, il, iu,
                              ps->rl_, ps->rr_,
                              mass_flux, s_x2flux);

        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
#if NSCALARS > 0
        ps->rl_.SwapAthenaArray(ps->rlb_);
#endif

        if (pr->xorder_use_auxiliaries)
        {
          al_.SwapAthenaArray(alb_);
        }

        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
#if NSCALARS > 0
          ps->r_rl_.SwapAthenaArray(ps->r_rlb_);
#endif

          if (pr->xorder_use_auxiliaries)
          {
            r_al_.SwapAthenaArray(r_alb_);
          }

          if (pr->xorder_use_fb_mask)
          {
            // mask_lb_.Fill(true);
            // mask_r_.Fill(true);
            mask_l_.SwapAthenaArray(mask_lb_);
          }

        }
      }
    }
  }

  //---------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    AthenaArray<Real> &x3flux = flux[X3DIR];
    AthenaArray<Real> &s_x3flux = pmb->pscalars->s_flux[X3DIR];
    mass_flux.InitWithShallowSlice(flux[X3DIR], 4, IDN, 1);

    pr->SetIndicialLimitsCalculateFluxes(IVZ, il, iu, jl, ju, kl, ku);

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional
      pr->ReconstructPrimitivesX3_(rv, w,
                                   wl_, wr_,
                                   kl-1, j, il, iu);
      pr->ReconstructMagneticFieldX3_(rv, bcc,
                                      wl_, wr_,
                                      kl-1, j, il, iu);
      pr->ReconstructPassiveScalarsX3_(p_rv, p,
                                       ps->rl_, ps->rr_,
                                       kl-1, j, il, iu);

      if (pr->xorder_use_cons_passive)
      for (int n=0; n<NSCALARS; ++n)
      for (int i=il; i<=iu; ++i)
      {
        ps->rl_(n,i) /= wl_(IDN,i);
        ps->rr_(n,i) /= wr_(IDN,i);
      }

      if (pr->xorder_use_auxiliaries)
      {
        pr->ReconstructHydroAuxiliariesX3_(rv, derived_ms,
                                           al_, ar_,
                                           kl-1, j, il, iu);
      }

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX3_(r_rv, w,
                                     r_wl_, r_wr_,
                                     kl-1, j, il, iu);
        pr->ReconstructMagneticFieldX3_(r_rv, bcc,
                                        r_wl_, r_wr_,
                                        kl-1, j, il, iu);
        pr->ReconstructPassiveScalarsX3_(p_r_rv, p,
                                         ps->r_rl_, ps->r_rr_,
                                         kl-1, j, il, iu);

        if (pr->xorder_use_auxiliaries)
        {
          pr->ReconstructHydroAuxiliariesX3_(r_rv, derived_ms,
                                             r_al_, r_ar_,
                                             kl-1, j, il, iu);
        }

        if (pr->xorder_use_fb_mask)
        {
          mask_l_.Fill(true);
          mask_r_.Fill(true);

          FallbackInadmissibleMaskPrimitiveX_(mask_l_, mask_r_,
                                              wl_, wr_,
                                              il, iu, IVZ);
          ps->FallbackInadmissibleMaskScalarX_(mask_l_, mask_r_,
                                               ps->rl_, ps->rr_,
                                               il, iu, IVZ);

          FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                     wl_, wr_,
                                     r_wl_, r_wr_,
                                     0, NWAVE, il, iu, IVZ);

          FallbackInadmissibleMaskX_(mask_l_, mask_r_,
                                     ps->rl_, ps->rr_,
                                     ps->r_rl_, ps->r_rr_,
                                     0, NSCALARS, il, iu, IVZ);

        }
        else
        {
          FallbackInadmissiblePrimitiveX3_(wl_, wr_,
                                           r_wl_, r_wr_,
                                           il, iu);
          ps->FallbackInadmissibleScalarX_(ps->rl_, ps->rr_,
                                           ps->r_rl_, ps->r_rr_,
                                           il, iu, IVZ);
        }

        if (pr->xorder_limit_species)
        {
          ps->ApplySpeciesLimits(ps->rl_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_, il, iu);
        }
      }
      else
      {
        if (pr->xorder_limit_species)
        {
          ps->ApplySpeciesLimits(ps->rl_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_, il, iu);
        }

#if USETM
        FloorPrimitiveX3_(wl_, wr_,
                          ps->rl_, ps->rr_,
                          kl-1, j, il, iu);
#else
        FloorPrimitiveX3_(wl_, wr_, kl-1, j, il, iu);
#endif

        if (pr->xorder_use_auxiliaries)
          LimitAuxiliariesX3_(al_, ar_, il, iu);

      }


      for (int k=kl; k<=ku; ++k)
      {
        pr->ReconstructPrimitivesX3_(rv, w,
                                     wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX3_(rv, bcc,
                                        wlb_, wr_,
                                        k, j, il, iu);
        pr->ReconstructPassiveScalarsX3_(p_rv, p,
                                         ps->rlb_, ps->rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_cons_passive)
        for (int n=0; n<NSCALARS; ++n)
        for (int i=il; i<=iu; ++i)
        {
          ps->rlb_(n,i) /= wlb_(IDN,i);
          ps->rr_(n,i) /= wr_(IDN,i);
        }

        if (pr->xorder_use_auxiliaries)
        {
          pr->ReconstructHydroAuxiliariesX3_(rv, derived_ms,
                                             alb_, ar_,
                                             k, j, il, iu);
        }

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX3_(r_rv, w,
                                       r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX3_(r_rv, bcc,
                                          r_wlb_, r_wr_,
                                          k, j, il, iu);
          pr->ReconstructPassiveScalarsX3_(p_r_rv, p,
                                           ps->r_rlb_, ps->r_rr_,
                                           k, j, il, iu);

          if (pr->xorder_use_auxiliaries)
          {
            pr->ReconstructHydroAuxiliariesX3_(r_rv, derived_ms,
                                               r_alb_, r_ar_,
                                               k, j, il, iu);
          }

          if (pr->xorder_use_fb_mask)
          {
            mask_lb_.Fill(true);
            mask_r_.Fill(true);

            FallbackInadmissibleMaskPrimitiveX_(mask_lb_, mask_r_,
                                                wlb_, wr_,
                                                il, iu, IVZ);
            ps->FallbackInadmissibleMaskScalarX_(mask_lb_, mask_r_,
                                                 ps->rlb_, ps->rr_,
                                                 il, iu, IVZ);

            FallbackInadmissibleMaskX_(mask_lb_, mask_r_,
                                       wlb_, wr_,
                                       r_wlb_, r_wr_,
                                       0, NWAVE, il, iu, IVZ);

            FallbackInadmissibleMaskX_(mask_lb_, mask_r_,
                                       ps->rlb_, ps->rr_,
                                       ps->r_rlb_, ps->r_rr_,
                                       0, NSCALARS, il, iu, IVZ);
          }
          else
          {
            FallbackInadmissiblePrimitiveX3_(wlb_, wr_,
                                             r_wlb_, r_wr_,
                                             il, iu);
            ps->FallbackInadmissibleScalarX_(ps->rlb_, ps->rr_,
                                             ps->r_rlb_, ps->r_rr_,
                                             il, iu, IVZ);
          }

          if (pr->xorder_limit_species)
          {
            ps->ApplySpeciesLimits(ps->rlb_, il, iu);
            ps->ApplySpeciesLimits(ps->rr_, il, iu);
          }

        }
        else
        {
          if (pr->xorder_limit_species)
          {
            ps->ApplySpeciesLimits(ps->rlb_, il, iu);
            ps->ApplySpeciesLimits(ps->rr_, il, iu);
          }
#if USETM
          FloorPrimitiveX3_(wlb_, wr_,
                            ps->rlb_, ps->rr_,
                            k, j, il, iu);
#else
          FloorPrimitiveX3_(wlb_, wr_,
                            k, j, il, iu);
#endif

          if (pr->xorder_use_auxiliaries)
            LimitAuxiliariesX3_(alb_, ar_, il, iu);

        }

        pmb->pcoord->CenterWidth3(k, j, il, iu, dxw_);

#if !MAGNETIC_FIELDS_ENABLED  // Hydro:
        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_,
                      x3flux, dxw_);
#else
        // flx(IBY) = (v3*b1 - v1*b3) = -EMFY
        // flx(IBZ) = (v3*b2 - v2*b3) =  EMFX
        RiemannSolver(k, j, il, iu, IVZ, b3, wl_, wr_,
                      x3flux, e2x3, e1x3, w_x3f, dxw_);
#endif

        ps->ComputeUpwindFlux(k, j, il, iu,
                              ps->rl_, ps->rr_,
                              mass_flux, s_x3flux);
        // swap the arrays for the next step
        wl_.SwapAthenaArray(wlb_);
#if NSCALARS > 0
        ps->rl_.SwapAthenaArray(ps->rlb_);
#endif

        if (pr->xorder_use_auxiliaries)
        {
          al_.SwapAthenaArray(alb_);
        }

        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
#if NSCALARS > 0
          ps->r_rl_.SwapAthenaArray(ps->r_rlb_);
#endif
          if (pr->xorder_use_auxiliaries)
          {
            r_al_.SwapAthenaArray(r_alb_);
          }

          if (pr->xorder_use_fb_mask)
          {
            // mask_lb_.Fill(true);
            // mask_r_.Fill(true);
            mask_l_.SwapAthenaArray(mask_lb_);
          }
        }
      }
    }
  }

  return;

}

void Hydro::CalculateFluxes_FluxReconstruction(
  AthenaArray<Real> &w, FaceField &b,
  AthenaArray<Real> &bcc, const int order)
{
  using namespace fluxes;
  using namespace fluxes::grhd;
  using namespace characteristic::grhd;

  MeshBlock *pmb = pmy_block;
  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;

  ReconstructionVariant r_rv = ReconstructionVariant::lin_vl;

  int il, iu, jl, ju, kl, ku;

  AthenaArray<Real> f(    NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  AthenaArray<Real> sf_m( NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  AthenaArray<Real> sf_p( NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  // AthenaArray<Real> lam_( pmb->ncells1);

  // Shu convention
  AthenaArray<Real> fl_( NHYDRO,pmb->nverts1);
  AthenaArray<Real> flb_(NHYDRO,pmb->nverts1);
  AthenaArray<Real> fr_( NHYDRO,pmb->nverts1);

  AthenaArray<Real> eig_v(NHYDRO,pmb->ncells3,pmb->ncells2,pmb->ncells1);

  flux[X1DIR].ZeroClear();
  flux[X2DIR].ZeroClear();
  flux[X3DIR].ZeroClear();

  int ivx;

  //---------------------------------------------------------------------------
  // i-direction
  {
    ivx = 1;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AthenaArray<Real> &x1flux = flux[X1DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);

    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX1(rv, sf_p, fl_, fr_, n, n, k, j, il-1, iu);
      }

      for (int n=0; n<NHYDRO; ++n)
      for (int i=il; i<=iu; ++i)
      {
        x1flux(n,k,j,i) += fl_(n,i);
      }

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX1(rv, sf_m, fl_, fr_, n, n, k, j, il-1, iu);
      }

      for (int n=0; n<NHYDRO; ++n)
      for (int i=il; i<=iu; ++i)
      {
        x1flux(n,k,j,i) += fr_(n,i);
      }
    }
  }

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    ivx = 2;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AthenaArray<Real> &x2flux = flux[X2DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);

    for (int k=kl; k<=ku; ++k)
    {

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX2(rv, sf_p, fl_, fr_, n, n, k, jl-1, il, iu);
      }

      for (int j=jl; j<=ju; ++j)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX2(rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x2flux(n,k,j,i) += fl_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX2(rv, sf_m, fl_, fr_, n, n, k, jl-1, il, iu);
      }

      for (int j=jl; j<=ju; ++j)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX2(rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x2flux(n,k,j,i) += fr_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }

    }
  }

  //---------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3)
  {
    ivx = 3;

    for (int k=0; k<pmb->ncells3; ++k)
    for (int j=0; j<pmb->ncells2; ++j)
    {
      AssembleFluxes(     pmb, k, j, 0, pmb->ncells1-1, ivx, f,  w, u);
      AssembleEigenvalues(pmb, k, j, 0, pmb->ncells1-1, ivx, eig_v, w, u);
      SplitFluxLLFMax(pmb, k, j, 0, pmb->ncells1-1, ivx, u, f,
                      eig_v, sf_m, sf_p);
    }

    AthenaArray<Real> &x3flux = flux[X3DIR];
    pr->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional

      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX3(rv, sf_p, fl_, fr_, n, n, kl-1, j, il, iu);
      }

      for (int k=kl; k<=ku; ++k)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX3(rv, sf_p, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x3flux(n,k,j,i) += fl_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


      for (int n=0; n<NHYDRO; ++n)
      {
        pr->ReconstructFieldX3(rv, sf_m, fl_, fr_, n, n, kl-1, j, il, iu);
      }

      for (int k=kl; k<=ku; ++k)
      {

        for (int n=0; n<NHYDRO; ++n)
        {
          pr->ReconstructFieldX3(rv, sf_m, flb_, fr_, n, n, k, j, il, iu);
        }

        for (int n=0; n<NHYDRO; ++n)
        for (int i=il; i<=iu; ++i)
        {
          x3flux(n,k,j,i) += fr_(n,i);
        }

        // swap the arrays for the next step
        fl_.SwapAthenaArray(flb_);
      }


    }
  }

}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalculateFluxes_STS
//  \brief Calculate Hydrodynamic Diffusion Fluxes for STS

void Hydro::CalculateFluxes_STS() {
  AddDiffusionFluxes();
}

void Hydro::AddDiffusionFluxes() {
  Field *pf = pmy_block->pfield;
  // add diffusion fluxes
  if (hdif.hydro_diffusion_defined) {
    if (hdif.nu_iso > 0.0 || hdif.nu_aniso > 0.0)
      hdif.AddDiffusionFlux(hdif.visflx,flux);
    if (NON_BAROTROPIC_EOS) {
      if (hdif.kappa_iso > 0.0 || hdif.kappa_aniso > 0.0)
        hdif.AddDiffusionEnergyFlux(hdif.cndflx,flux);
    }
  }
  if (MAGNETIC_FIELDS_ENABLED && NON_BAROTROPIC_EOS) {
    if (pf->fdif.field_diffusion_defined)
      pf->fdif.AddPoyntingFlux(pf->fdif.pflux);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CombineFluxes(k,j,il,iu,efl,flux_HO,flux_LO,flux)
//  \brief Hybridization of Low and High order flux, weighted by entropy flux limiter

void Hydro::CombineFluxes(const int k,const int j,const int il, const int iu,const int dir,
                          AthenaArray<Real> const &efl,
                          AthenaArray<Real> const &f_HO,
                          AthenaArray<Real> const &f_LO,
                          AthenaArray<Real> &f)
{
  
#if 1
  for(int i =il; i<= iu ;++i){
    if ((dir == 1) && ((atm_mask(k,j,i-1) > 0.9) || (atm_mask(k,j,i) > 0.9))){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }
    if ((dir == 2) && ((atm_mask(k,j-1,i) > 0.9) || (atm_mask(k,j,i) > 0.9)) ){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }
    if ((dir == 3) && ((atm_mask(k-1,j,i) > 0.9) || (atm_mask(k,j,i) > 0.9)) ){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }

    if(pmy_block->pmy_mesh->efl_it_count <= buffer_it){
      for (int n=0; n<NHYDRO;++n){
        f(n,k,j,i)=f_LO(n,k,j,i);
      }
    }

    else{  
      for (int n=0; n<NHYDRO;++n){
        f(n,k,j,i)= efl(k,j,i)*f_HO(n,k,j,i) + (1.0-efl(k,j,i))*f_LO(n,k,j,i);
      }
      if (!std::isfinite(f(0,k,j,i)) ||
          !std::isfinite(f(1,k,j,i)) ||
          !std::isfinite(f(2,k,j,i)) ||
          !std::isfinite(f(3,k,j,i)) ||
          !std::isfinite(f(4,k,j,i)) ){
          for (int n =0; n<NHYDRO; ++n){
            f(n,k,j,i) = 0.0;
          }
        }
    }
  }
#endif

#if 0
  for(int i =il; i<= iu ;++i){
    if ((dir == 1) && ((atm_mask(k,j,i-1) > 0.9) || (atm_mask(k,j,i) > 0.9))){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }
    if ((dir == 2) && ((atm_mask(k,j-1,i) > 0.9) || (atm_mask(k,j,i) > 0.9)) ){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }
    if ((dir == 3) && ((atm_mask(k-1,j,i) > 0.9) || (atm_mask(k,j,i) > 0.9)) ){
      for (int n=0; n<NHYDRO;++n) f(n,k,j,i)=0.0;
      continue;
    }
    else{
      for (int n=0; n<NHYDRO;++n){
        f(n,k,j,i)=f_LO(n,k,j,i);
      }
    }
  }
#endif
  return;
}
   