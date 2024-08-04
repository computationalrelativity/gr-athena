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
#include "../gravity/gravity.hpp"
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

#ifdef DBG_COMBINED_HYDPA
  CalculateFluxesCombined(w,b,bcc,order);
  return;
#endif

  MeshBlock *pmb = pmy_block;
  Reconstruction * pr = pmb->precon;
  typedef Reconstruction::ReconstructionVariant ReconstructionVariant;
  ReconstructionVariant rv = pr->xorder_style;
  ReconstructionVariant r_rv = pr->xorder_style_fb;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
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
  il = is, iu = ie+1;
  jl = js, ju = je+(pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3);  // 2d or 3d
  kl = ks, ku = ke+(pmb->pmy_mesh->f3);                       // if 3d

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
    RiemannSolver(k, j, il, iu, IVX, wl_, wr_, x1flux, dxw_);
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
    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+(pmb->pmy_mesh->f3);  // if 3d

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
        RiemannSolver(k, j, il, iu, IVY, wl_, wr_,
                      x2flux, dxw_);
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
    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+1;

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
        RiemannSolver(k, j, il, iu, IVZ, wl_, wr_,
                      x3flux, dxw_);
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

  PassiveScalars *ps = pmb->pscalars;

  // For passive-scalar reconstruction
  AthenaArray<Real> mass_flux;
  AthenaArray<Real> &r = pmb->pscalars->r;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
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

  il = is, iu = ie+1;
  jl = js, ju = je+(pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3);  // 2d or 3d
  kl = ks, ku = ke+(pmb->pmy_mesh->f3);                       // if 3d

  for (int k=kl; k<=ku; ++k)
  for (int j=jl; j<=ju; ++j)
  {
    pr->ReconstructPrimitivesX1_(rv, w,
                                 wl_, wr_,
                                 k, j, il, iu);
    pr->ReconstructMagneticFieldX1_(rv, bcc,
                                    wl_, wr_,
                                    k, j, il, iu);
    pr->ReconstructPassiveScalarsX1_(rv, r,
                                     ps->rl_, ps->rr_,
                                     k, j, il, iu);

    if (pr->xorder_use_fb)
    {
      pr->ReconstructPrimitivesX1_(r_rv, w, r_wl_, r_wr_,
                                   k, j, il, iu);
      pr->ReconstructMagneticFieldX1_(r_rv, bcc, r_wl_, r_wr_,
                                      k, j, il, iu);
      pr->ReconstructPassiveScalarsX1_(r_rv, r,
                                       ps->r_rl_, ps->r_rr_,
                                       k, j, il, iu);

      FallbackInadmissiblePrimitiveX1_(wl_, wr_,
                                       r_wl_, r_wr_,
                                       il, iu);
      ps->FallbackInadmissibleScalarX1_(ps->rl_, ps->rr_,
                                        ps->r_rl_, ps->r_rr_,
                                        il, iu);

      ps->ApplySpeciesLimits(ps->rl_, il, iu);
      ps->ApplySpeciesLimits(ps->rr_, il, iu);

    }
    else
    {
      ps->ApplySpeciesLimits(ps->rl_, il, iu);
      ps->ApplySpeciesLimits(ps->rr_, il, iu);

#if USETM
      FloorPrimitiveX1_(wl_, wr_,
                        ps->rl_, ps->rr_,
                        k, j, il, iu);
#else
      FloorPrimitiveX1_(wl_, wr_,
                        k, j, il, iu);
#endif
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
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2)
  {
    AthenaArray<Real> &x2flux = flux[X2DIR];
    AthenaArray<Real> &s_x2flux = pmb->pscalars->s_flux[X2DIR];
    mass_flux.InitWithShallowSlice(flux[X2DIR], 4, IDN, 1);

    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+(pmb->pmy_mesh->f3);  // if 3d

    for (int k=kl; k<=ku; ++k)
    {
      pr->ReconstructPrimitivesX2_(rv, w,
                                   wl_, wr_,
                                   k, jl-1, il, iu);
      pr->ReconstructMagneticFieldX2_(rv, bcc,
                                      wl_, wr_,
                                      k, jl-1, il, iu);
      pr->ReconstructPassiveScalarsX2_(rv, r,
                                       ps->rl_, ps->rr_,
                                       k, jl-1, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX2_(r_rv, w,
                                     r_wl_, r_wr_,
                                     k, jl-1, il, iu);
        pr->ReconstructMagneticFieldX2_(r_rv, bcc,
                                        r_wl_, r_wr_,
                                        k, jl-1, il, iu);
        pr->ReconstructPassiveScalarsX2_(r_rv, r,
                                         ps->r_rl_, ps->r_rr_,
                                         k, jl-1, il, iu);

        FallbackInadmissiblePrimitiveX2_(wl_, wr_,
                                         r_wl_, r_wr_,
                                         il, iu);
        ps->FallbackInadmissibleScalarX2_(ps->rl_, ps->rr_,
                                          ps->r_rl_, ps->r_rr_,
                                          il, iu);

        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);

      }
      else
      {
        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);

#if USETM
        FloorPrimitiveX2_(wl_, wr_,
                          ps->rl_, ps->rr_,
                          k, jl-1, il, iu);
#else
        FloorPrimitiveX2_(wl_, wr_,
                          k, jl-1, il, iu);
#endif
      }


      for (int j=jl; j<=ju; ++j)
      {

        pr->ReconstructPrimitivesX2_(rv, w,
                                     wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX2_(rv, bcc,
                                        wlb_, wr_,
                                        k, j, il, iu);
        pr->ReconstructPassiveScalarsX2_(rv, r,
                                         ps->rlb_, ps->rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX2_(r_rv, w,
                                       r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX2_(r_rv, bcc,
                                          r_wlb_, r_wr_,
                                          k, j, il, iu);
          pr->ReconstructPassiveScalarsX2_(r_rv, r,
                                           ps->r_rlb_, ps->r_rr_,
                                           k, j, il, iu);

          FallbackInadmissiblePrimitiveX2_(wlb_, wr_,
                                           r_wlb_, r_wr_,
                                           il, iu);
          ps->FallbackInadmissibleScalarX2_(ps->rlb_, ps->rr_,
                                            ps->r_rlb_, ps->r_rr_,
                                            il, iu);

          ps->ApplySpeciesLimits(ps->rlb_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_,  il, iu);

        }
        else
        {
          ps->ApplySpeciesLimits(ps->rlb_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_,  il, iu);

#if USETM
          FloorPrimitiveX2_(wlb_, wr_,
                            ps->rlb_, ps->rr_,
                            k, j, il, iu);
#else
          FloorPrimitiveX2_(wlb_, wr_, k, j, il, iu);
#endif
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
        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
#if NSCALARS > 0
          ps->r_rl_.SwapAthenaArray(ps->r_rlb_);
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
    AthenaArray<Real> &s_x3flux = pmb->pscalars->s_flux[X3DIR];
    mass_flux.InitWithShallowSlice(flux[X3DIR], 4, IDN, 1);

    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+1;

    for (int j=jl; j<=ju; ++j)
    { // this loop ordering is intentional
      pr->ReconstructPrimitivesX3_(rv, w,
                                   wl_, wr_,
                                   kl-1, j, il, iu);
      pr->ReconstructMagneticFieldX3_(rv, bcc,
                                      wl_, wr_,
                                      kl-1, j, il, iu);
      pr->ReconstructPassiveScalarsX3_(rv, r,
                                       ps->rl_, ps->rr_,
                                       kl-1, j, il, iu);

      if (pr->xorder_use_fb)
      {
        pr->ReconstructPrimitivesX3_(r_rv, w,
                                     r_wl_, r_wr_,
                                     kl-1, j, il, iu);
        pr->ReconstructMagneticFieldX3_(r_rv, bcc,
                                        r_wl_, r_wr_,
                                        kl-1, j, il, iu);
        pr->ReconstructPassiveScalarsX3_(r_rv, r,
                                         ps->r_rl_, ps->r_rr_,
                                         kl-1, j, il, iu);

        FallbackInadmissiblePrimitiveX3_(wl_, wr_,
                                         r_wl_, r_wr_,
                                         il, iu);
        ps->FallbackInadmissibleScalarX3_(ps->rl_, ps->rr_,
                                          ps->r_rl_, ps->r_rr_,
                                          il, iu);

        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);

      }
      else
      {
        ps->ApplySpeciesLimits(ps->rl_, il, iu);
        ps->ApplySpeciesLimits(ps->rr_, il, iu);

#if USETM
        FloorPrimitiveX3_(wl_, wr_,
                          ps->rl_, ps->rr_,
                          kl-1, j, il, iu);
#else
        FloorPrimitiveX3_(wl_, wr_, kl-1, j, il, iu);
#endif
      }


      for (int k=kl; k<=ku; ++k)
      {
        pr->ReconstructPrimitivesX3_(rv, w,
                                     wlb_, wr_,
                                     k, j, il, iu);
        pr->ReconstructMagneticFieldX3_(rv, bcc,
                                        wlb_, wr_,
                                        k, j, il, iu);
        pr->ReconstructPassiveScalarsX3_(rv, r,
                                         ps->rlb_, ps->rr_,
                                         k, j, il, iu);

        if (pr->xorder_use_fb)
        {
          pr->ReconstructPrimitivesX3_(r_rv, w,
                                       r_wlb_, r_wr_,
                                       k, j, il, iu);
          pr->ReconstructMagneticFieldX3_(r_rv, bcc,
                                          r_wlb_, r_wr_,
                                          k, j, il, iu);
          pr->ReconstructPassiveScalarsX3_(r_rv, r,
                                           ps->r_rlb_, ps->r_rr_,
                                           k, j, il, iu);

          FallbackInadmissiblePrimitiveX3_(wlb_, wr_,
                                           r_wlb_, r_wr_,
                                           il, iu);
          ps->FallbackInadmissibleScalarX3_(ps->rlb_, ps->rr_,
                                            ps->r_rlb_, ps->r_rr_,
                                            il, iu);

          ps->ApplySpeciesLimits(ps->rlb_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_,  il, iu);

        }
        else
        {
          ps->ApplySpeciesLimits(ps->rlb_, il, iu);
          ps->ApplySpeciesLimits(ps->rr_,  il, iu);

#if USETM
          FloorPrimitiveX3_(wlb_, wr_,
                            ps->rlb_, ps->rr_,
                            k, j, il, iu);
#else
          FloorPrimitiveX3_(wlb_, wr_,
                            k, j, il, iu);
#endif
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
        if (pr->xorder_use_fb)
        {
          r_wl_.SwapAthenaArray(r_wlb_);
#if NSCALARS > 0
          ps->r_rl_.SwapAthenaArray(ps->r_rlb_);
#endif
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

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
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
    il = is, iu = ie+1;
    jl = js, ju = je+(pmb->pmy_mesh->f2 || pmb->pmy_mesh->f3);  // 2d or 3d
    kl = ks, ku = ke+(pmb->pmy_mesh->f3);                       // if 3d

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
    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+(pmb->pmy_mesh->f3);  // if 3d

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
    il = is, iu = ie+1;
    jl = js, ju = je+1;
    kl = ks, ku = ke+1;

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
