//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_refine.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <iterator>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../scalars/scalars.hpp"
#include "bvals.hpp"
#include "bvals_interfaces.hpp"
#include "cc/hydro/bvals_hydro.hpp"
#include "fc/bvals_fc.hpp"
#include "vc/bvals_vc.hpp"
#include "cx/bvals_cx.hpp"

#include "../z4c/z4c.hpp"
#include "../wave/wave.hpp"
#include "../m1/m1.hpp"

void BoundaryValues::ProlongateBoundariesHydroCons(const Real time,
                                                   const Real dt)
{
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;

  // BD: TODO - opt- if nn all same level not required?
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  if (pmb->NeighborBlocksSameLevel())
    return;
#endif // DBG_NO_REF_NN_SAME_LEVEL

  // Populate (through restriction) complementary coarse rep. on physical layer
  // This is needed for prolongation
  for (auto bvar : pbval->GetBvarsMatter())
  {
    bvar->RestrictInterior(time, dt);
  }

  ApplyPhysicalBoundariesOnCoarseLevel(
    time, dt,
    pbval->GetBvarsMatter(),
    pmb->cis, pmb->cie,
    pmb->cjs, pmb->cje,
    pmb->cks, pmb->cke,
    NGHOST);

  // Finally prolongate to fill ghosts ----------------------------------------
  for (auto bvar : pbval->GetBvarsMatter())
  {
    bvar->ProlongateBoundaries(time, dt);
  }

  if (MAGNETIC_FIELDS_ENABLED)
    CalculateCellCenteredFieldOnProlongedBoundaries();
}

void BoundaryValues::ProlongateBoundariesHydroPrim(const Real time,
                                                   const Real dt)
{
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;

  // BD: TODO - opt- if nn all same level not required?
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  if (pmb->NeighborBlocksSameLevel())
    return;
#endif // DBG_NO_REF_NN_SAME_LEVEL

  // Populate (through restriction) complementary coarse rep. on physical layer
  // This is needed for prolongation
  for (auto bvar : pbval->GetBvarsMatter())
  {
    bvar->RestrictInterior(time, dt);
  }

  // (unique to Hydro) also restrict primitive values in ghost zones when GR +
  // multilevel
  if (GENERAL_RELATIVITY)
  {
    pmb->SetBoundaryVariablesPrimitive();
    pmb->phydro->hbvar.RestrictInterior(time, dt);
    pmb->SetBoundaryVariablesConserved();
  }

  CalculateCellCenteredFieldOnCoarseLevel();
  PrimitiveToConservedOnCoarseLevelBoundaries();

  pmb->SetBoundaryVariablesPrimitive();

  ApplyPhysicalBoundariesOnCoarseLevel(
    time, dt,
    pbval->GetBvarsMatter(),
    pmb->cis, pmb->cie,
    pmb->cjs, pmb->cje,
    pmb->cks, pmb->cke,
    NGHOST);

  pmb->SetBoundaryVariablesConserved();

  // Finally prolongate to fill ghosts ----------------------------------------
  pmb->SetBoundaryVariablesPrimitive();

  for (auto bvar : pbval->GetBvarsMatter())
  {
    bvar->ProlongateBoundaries(time, dt);
  }

  pmb->SetBoundaryVariablesConserved();

  if (MAGNETIC_FIELDS_ENABLED)
    CalculateCellCenteredFieldOnProlongedBoundaries();

  PrimitiveToConservedOnProlongedBoundaries();
}

void BoundaryValues::ProlongateBoundariesZ4c(const Real time, const Real dt)
{
  // BD: TODO - opt- if nn all same level not required?
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;
  Z4c *pz4c = pz4c = pmb->pz4c;

  std::vector<BoundaryVariable *> ubvar_ { &pz4c->ubvar };

  // BD: TODO - should restrict here also to have updated fields...
  // pz4c->ubvar.RestrictInterior(time, dt);

  ApplyPhysicalBoundariesOnCoarseLevel(
    time, dt,
    ubvar_,  // or pbval->GetBvarsZ4c()
    pz4c->mbi.cil, pz4c->mbi.ciu,
    pz4c->mbi.cjl, pz4c->mbi.cju,
    pz4c->mbi.ckl, pz4c->mbi.cku,
    pz4c->mbi.cng);

  // for (auto bvar : pbval->GetBvarsZ4c())
  // {
  //   bvar->ProlongateBoundaries(time, dt);
  // }
  pz4c->ubvar.ProlongateBoundaries(time, dt);
}

void BoundaryValues::ProlongateBoundariesWave(const Real time, const Real dt)
{
  // BD: TODO - opt- if nn all same level not required?
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;
  Wave *pwave = pmb->pwave;

  std::vector<BoundaryVariable *> ubvar_ {
    WAVE_SW_CC_CX_VC(&pwave->ubvar_cc,
                     &pwave->ubvar_cx,
                     &pwave->ubvar_vc)
  };

  ApplyPhysicalBoundariesOnCoarseLevel(
    time, dt,
    ubvar_, // pbval->GetBvarsWave(),
    pwave->mbi.cil, pwave->mbi.ciu,
    pwave->mbi.cjl, pwave->mbi.cju,
    pwave->mbi.ckl, pwave->mbi.cku,
    pwave->mbi.cng);

  // for (auto bvar : pbval->GetBvarsWave())
  // {
  //   bvar->ProlongateBoundaries(time, dt);
  // }
  WAVE_SW_CC_CX_VC(pwave->ubvar_cc,
                   pwave->ubvar_cx,
                   pwave->ubvar_vc).ProlongateBoundaries(time, dt);
}

void BoundaryValues::ProlongateBoundariesAux(const Real time, const Real dt)
{
  // BD: This is currently only utilized for Z4c variables
  // BD: TODO - opt- if nn all same level not required?
  if (Z4C_ENABLED)
  {
    MeshBlock *pmb = pmy_block_;
    BoundaryValues *pbval = pmb->pbval;
    MeshRefinement *pmr = pmb->pmr;
    Z4c *pz4c = nullptr;

    pz4c = pmb->pz4c;

    ApplyPhysicalBoundariesOnCoarseLevel(
      time, dt,
      pbval->GetBvarsAux(),
      pz4c->mbi.cil, pz4c->mbi.ciu,
      pz4c->mbi.cjl, pz4c->mbi.cju,
      pz4c->mbi.ckl, pz4c->mbi.cku,
      pz4c->mbi.cng);

    for (auto bvar : pbval->GetBvarsAux())
    {
      bvar->ProlongateBoundaries(time, dt);
    }
  }
}

void BoundaryValues::ProlongateBoundariesM1(const Real time, const Real dt)
{
  MeshBlock *pmb = pmy_block_;
  BoundaryValues *pbval = pmb->pbval;

  // BD: TODO - opt- if nn all same level not required?
#if defined(DBG_NO_REF_NN_SAME_LEVEL)
  if (pmb->NeighborBlocksSameLevel())
    return;
#endif // DBG_NO_REF_NN_SAME_LEVEL

  // Populate (through restriction) complementary coarse rep. on physical layer
  // This is needed for prolongation
  for (auto bvar : pbval->GetBvarsM1())
  {
    bvar->RestrictInterior(time, dt);
  }

  ApplyPhysicalBoundariesOnCoarseLevel(
    time, dt,
    pbval->GetBvarsM1(),
    pmb->cis, pmb->cie,
    pmb->cjs, pmb->cje,
    pmb->cks, pmb->cke,
    NGHOST);

  // Finally prolongate to fill ghosts ----------------------------------------
  for (auto bvar : pbval->GetBvarsM1())
  {
    bvar->ProlongateBoundaries(time, dt);
  }
}

// Handler for application of physical boundaries for all variable types
void BoundaryValues::ApplyPhysicalBoundariesOnCoarseLevel(
  const Real time, const Real dt,
  std::vector<BoundaryVariable *> & bvars,
  const int var_cis, const int var_cie,
  const int var_cjs, const int var_cje,
  const int var_cks, const int var_cke,
  const int cng)
{
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  Coordinates *pco = pmr->pcoarsec;

  // Automatic application of BC on coarse representation
  for (auto bvar : bvars)
  {
    bvar->InterchangeFundamentalCoarse();
  }

  int bis = var_cis - cng, bie = var_cie + cng,
      bjs = var_cjs, bje = var_cje,
      bks = var_cks, bke = var_cke;

  // Extend the transverse limits that correspond to periodic boundaries as they are
  // updated: x1, then x2, then x3
  if (!apply_bndry_fn_[BoundaryFace::inner_x2] && pmb->block_size.nx2 > 1)
    bjs = var_cjs - cng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x2] && pmb->block_size.nx2 > 1)
    bje = var_cje + cng;
  if (!apply_bndry_fn_[BoundaryFace::inner_x3] && pmb->block_size.nx3 > 1)
    bks = var_cks - cng;
  if (!apply_bndry_fn_[BoundaryFace::outer_x3] && pmb->block_size.nx3 > 1)
    bke = var_cke + cng;

  // Apply boundary function on inner-x1 and update W,bcc (if not periodic)
  if (apply_bndry_fn_[BoundaryFace::inner_x1])
  {
    DispatchBoundaryFunctions(pmb, pco, time, dt,
                              var_cis, var_cie,
                              bjs, bje,
                              bks, bke,
                              cng,
                              BoundaryFace::inner_x1,
                              bvars);
  }

  // Apply boundary function on outer-x1 and update W,bcc (if not periodic)
  if (apply_bndry_fn_[BoundaryFace::outer_x1]) {
    DispatchBoundaryFunctions(pmb, pco, time, dt,
                              var_cis, var_cie,
                              bjs, bje,
                              bks, bke,
                              cng,
                              BoundaryFace::outer_x1,
                              bvars);
  }

  if (pmb->block_size.nx2 > 1) { // 2D or 3D
    // Apply boundary function on inner-x2 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::inner_x2])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                var_cjs, var_cje,
                                bks, bke,
                                cng,
                                BoundaryFace::inner_x2,
                                bvars);
    }

    // Apply boundary function on outer-x2 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::outer_x2])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                var_cjs, var_cje,
                                bks, bke,
                                cng,
                                BoundaryFace::outer_x2,
                                bvars);
    }
  }

  if (pmb->block_size.nx3 > 1)
  { // 3D
    bjs = var_cjs - cng;
    bje = var_cje + cng;

    // Apply boundary function on inner-x3 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::inner_x3])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                bjs, bje,
                                var_cks, var_cke,
                                cng,
                                BoundaryFace::inner_x3,
                                bvars);
    }

    // Apply boundary function on outer-x3 and update W,bcc (if not periodic)
    if (apply_bndry_fn_[BoundaryFace::outer_x3])
    {
      DispatchBoundaryFunctions(pmb, pco, time, dt,
                                bis, bie,
                                bjs, bje,
                                var_cks, var_cke,
                                cng,
                                BoundaryFace::outer_x3,
                                bvars);
    }
  }

  // Revert internal representation of variables
  for (auto bvar : bvars)
  {
    bvar->InterchangeFundamentalCoarse();
  }

}

void BoundaryValues::CalculateCellCenteredFieldOnProlongedBoundaries()
{
#if MAGNETIC_FIELDS_ENABLED
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  BoundaryValues *pbval = pmb->pbval;
  Field *pf = pmb->pfield;

  const int mylevel = pbval->loc.level;
  const int nneighbor = pbval->nneighbor;

  int fsi, fei, fsj, fej, fsk, fek;

  for (int n=0; n<nneighbor; ++n)
  {
    NeighborBlock& nb = pbval->neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    pf->fbvar.CalculateProlongationIndicesFine(nb,
                                               fsi, fei, fsj, fej, fsk, fek);

    pf->CalculateCellCenteredField(pf->b, pf->bcc, pmb->pcoord,
                                   fsi, fei, fsj, fej, fsk, fek);
  }
#endif
}

void BoundaryValues::PrimitiveToConservedOnProlongedBoundaries()
{
#if FLUID_ENABLED
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  BoundaryValues *pbval = pmb->pbval;

  EquationOfState *peos = pmb->peos;
  Hydro *ph = pmb->phydro;
  Field *pf = (MAGNETIC_FIELDS_ENABLED) ? pmb->pfield : nullptr;
  PassiveScalars *ps = (NSCALARS>0) ? pmb->pscalars : nullptr;

  const int mylevel = pbval->loc.level;
  const int nneighbor = pbval->nneighbor;

  int fsi, fei, fsj, fej, fsk, fek;

  for (int n=0; n<nneighbor; ++n)
  {
    NeighborBlock& nb = pbval->neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    ph->hbvar.CalculateProlongationIndicesFine(nb,
                                               fsi, fei, fsj, fej, fsk, fek);

    peos->PrimitiveToConserved(ph->w, ps->r, pf->bcc, ph->u, ps->s,
                               pmb->pcoord,
                               fsi, fei, fsj, fej, fsk, fek);
  }
#endif
}

void BoundaryValues::CalculateCellCenteredFieldOnCoarseLevel()
{
#if MAGNETIC_FIELDS_ENABLED
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  BoundaryValues *pbval = pmb->pbval;

  EquationOfState *peos = pmb->peos;
  Hydro *ph = pmb->phydro;
  Field *pf = (MAGNETIC_FIELDS_ENABLED) ? pmb->pfield : nullptr;
  PassiveScalars *ps = (NSCALARS>0) ? pmb->pscalars : nullptr;

  const int mylevel = pbval->loc.level;
  const int nneighbor = pbval->nneighbor;

  int si, ei, sj, ej, sk, ek;

  for (int n=0; n<nneighbor; ++n)
  {
    NeighborBlock& nb = pbval->neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    pf->fbvar.CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);

    // convert the ghost zone and ghost-ghost zones into primitive variables
    // this includes cell-centered field calculation
    int f1m = 0, f1p = 0, f2m = 0, f2p = 0, f3m = 0, f3p = 0;
    if (nb.ni.ox1 == 0) {
      if (nblevel[1][1][0] != -1) f1m = 1;
      if (nblevel[1][1][2] != -1) f1p = 1;
    } else {
      f1m = 1;
      f1p = 1;
    }
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox2 == 0) {
        if (nblevel[1][0][1] != -1) f2m = 1;
        if (nblevel[1][2][1] != -1) f2p = 1;
      } else {
        f2m = 1;
        f2p = 1;
      }
    }
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox3 == 0) {
        if (nblevel[0][1][1] != -1) f3m = 1;
        if (nblevel[2][1][1] != -1) f3p = 1;
      } else {
        f3m = 1;
        f3p = 1;
      }
    }

    pf->CalculateCellCenteredField(pf->coarse_b_,
                                   pf->coarse_bcc_,
                                   pmr->pcoarsec,
                                   si-f1m, ei+f1p,
                                   sj-f2m, ej+f2p,
                                   sk-f3m, ek+f3p);
  }
#endif
}

void BoundaryValues::PrimitiveToConservedOnCoarseLevelBoundaries()
{
#if FLUID_ENABLED
  MeshBlock *pmb = pmy_block_;
  MeshRefinement *pmr = pmb->pmr;
  BoundaryValues *pbval = pmb->pbval;

  EquationOfState *peos = pmb->peos;
  Hydro *ph = pmb->phydro;
  Field *pf = (MAGNETIC_FIELDS_ENABLED) ? pmb->pfield : nullptr;
  PassiveScalars *ps = (NSCALARS>0) ? pmb->pscalars : nullptr;

  const int mylevel = pbval->loc.level;
  const int nneighbor = pbval->nneighbor;

  int si, ei, sj, ej, sk, ek;

  for (int n=0; n<nneighbor; ++n)
  {
    NeighborBlock& nb = pbval->neighbor[n];
    if (nb.snb.level >= mylevel) continue;

    ph->hbvar.CalculateProlongationIndices(nb, si, ei, sj, ej, sk, ek);

    // convert the ghost zone and ghost-ghost zones into primitive variables
    // this includes cell-centered field calculation
    int f1m = 0, f1p = 0, f2m = 0, f2p = 0, f3m = 0, f3p = 0;
    if (nb.ni.ox1 == 0) {
      if (nblevel[1][1][0] != -1) f1m = 1;
      if (nblevel[1][1][2] != -1) f1p = 1;
    } else {
      f1m = 1;
      f1p = 1;
    }
    if (pmb->block_size.nx2 > 1) {
      if (nb.ni.ox2 == 0) {
        if (nblevel[1][0][1] != -1) f2m = 1;
        if (nblevel[1][2][1] != -1) f2p = 1;
      } else {
        f2m = 1;
        f2p = 1;
      }
    }
    if (pmb->block_size.nx3 > 1) {
      if (nb.ni.ox3 == 0) {
        if (nblevel[0][1][1] != -1) f3m = 1;
        if (nblevel[2][1][1] != -1) f3p = 1;
      } else {
        f3m = 1;
        f3p = 1;
      }
    }

    static const int coarseflag = 1;
    pmb->peos->ConservedToPrimitive(ph->coarse_cons_,
                                    ph->coarse_prim_,
                                    ph->coarse_prim_,
                                    ps->coarse_s_,
                                    ps->coarse_r_,
                                    pf->coarse_bcc_,
                                    pmr->pcoarsec,
                                    si-f1m, ei+f1p,
                                    sj-f2m, ej+f2p,
                                    sk-f3m, ek+f3p,
                                    coarseflag);
  }
#endif
}

//
// :D
//