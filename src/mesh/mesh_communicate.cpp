// C headers

// C++ headers
#include <iostream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_aliases.hpp"
#include "../bvals/bvals.hpp"
#include "../globals.hpp"
#include "mesh.hpp"

#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../m1/m1.hpp"
#include "../scalars/scalars.hpp"
#include "../wave/wave.hpp"
#include "../z4c/z4c.hpp"

void Mesh::FinalizeWave(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Wave *pw = nullptr;

  #pragma omp for private(pmb, pbval, pw)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pw = pmb->pwave;

    if (multilevel)
    {
      pbval->ProlongateBoundariesWave(time, 0.0);
    }

    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsWave(),
      pw->mbi.il, pw->mbi.iu,
      pw->mbi.jl, pw->mbi.ju,
      pw->mbi.kl, pw->mbi.ku,
      pw->mbi.ng);
  }
}

void Mesh::FinalizeZ4cADMPhysical(std::vector<MeshBlock*> & pmb_array,
                                  const bool enforce_alg)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pbval, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pz = pmb->pz4c;

    const bool skip_physical = false;

    // Enforce the algebraic constraints
    if (enforce_alg)
    {
      pz->AlgConstr(pz->storage.u,
                    pz->mbi.il, pz->mbi.iu,
                    pz->mbi.jl, pz->mbi.ju,
                    pz->mbi.kl, pz->mbi.ku,
                    skip_physical);
    }

    // Need ADM variables for con2prim
    pz->Z4cToADM(pz->storage.u,
                 pz->storage.adm,
                 pz->mbi.il, pz->mbi.iu,
                 pz->mbi.jl, pz->mbi.ju,
                 pz->mbi.kl, pz->mbi.ku,
                 skip_physical);
  }
}

void Mesh::FinalizeZ4cADMGhosts(std::vector<MeshBlock*> & pmb_array,
                                const bool enforce_alg)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pbval, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pz = pmb->pz4c;

    if (multilevel)
    {
      pbval->ProlongateBoundariesZ4c(time, 0.0);
    }

    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsZ4c(),
      pz->mbi.il, pz->mbi.iu,
      pz->mbi.jl, pz->mbi.ju,
      pz->mbi.kl, pz->mbi.ku,
      pz->mbi.ng);

    const bool skip_physical = true;

    // Enforce the algebraic constraints
    if (enforce_alg)
    {
      pz->AlgConstr(pz->storage.u,
        0, pz->mbi.nn1-1,
        0, pz->mbi.nn2-1,
        0, pz->mbi.nn3-1,
        skip_physical);
    }

    // Need ADM variables for con2prim
    pz->Z4cToADM(pz->storage.u,
      pz->storage.adm,
      0, pz->mbi.nn1-1,
      0, pz->mbi.nn2-1,
      0, pz->mbi.nn3-1,
      skip_physical);
  }
}

void Mesh::FinalizeZ4cADM(std::vector<MeshBlock*> & pmb_array,
                          const bool enforce_alg)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pbval, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pz = pmb->pz4c;

    if (multilevel)
    {
      pbval->ProlongateBoundariesZ4c(time, 0.0);
    }

    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsZ4c(),
      pz->mbi.il, pz->mbi.iu,
      pz->mbi.jl, pz->mbi.ju,
      pz->mbi.kl, pz->mbi.ku,
      pz->mbi.ng);

    // Enforce the algebraic constraints
    if (enforce_alg)
    {
      pz->AlgConstr(pz->storage.u);
    }

    // Need ADM variables for con2prim
    pz->Z4cToADM(pz->storage.u, pz->storage.adm);
  }
}

void Mesh::FinalizeZ4cADM_Matter(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb = nullptr;
  BoundaryValues *pbval = nullptr;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;
  Z4c *pz = nullptr;

  #pragma omp for private(pmb, pbval, pf, ph, ps, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pf = pmb->pfield;
    ph = pmb->phydro;
    ps = pmb->pscalars;
    pz = pmb->pz4c;

    pz->GetMatter(pz->storage.mat, pz->storage.adm, ph->w, ps->r, pf->bcc);

    // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@S:Sc\n", "@E:Sc\n");

  }
}

void Mesh::FinalizeM1(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  M1::M1 *pm1 = nullptr;

  #pragma omp for private(pmb, pbval, pm1)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pm1 = pmb->pm1;

    if (multilevel)
    {
      pbval->ProlongateBoundariesM1(time, 0.0);
    }

    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsM1(),
      pm1->mbi.il, pm1->mbi.iu,
      pm1->mbi.jl, pm1->mbi.ju,
      pm1->mbi.kl, pm1->mbi.ku,
      pm1->mbi.ng);

#if M1_ENABLED
    // Conserved variables are now available globally;
    // Ensure that geometric & hydro terms are available
    pm1->UpdateGeometry(pm1->geom, pm1->scratch);
    pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
    // Need the reference velocity & closure
    pm1->CalcFiducialVelocity();
    pm1->CalcClosure(pm1->storage.u);
#endif // M1_ENABLED
  }
}

void Mesh::FinalizeDiffusion(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;

  #pragma omp for private(pmb, pbval, pf, ph)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];

    pf = pmb->pfield;
    ph = pmb->phydro;

    if (FLUID_ENABLED && ph->hdif.hydro_diffusion_defined)
      ph->hdif.SetDiffusivity(ph->w, pf->bcc);

    if (MAGNETIC_FIELDS_ENABLED) {
      if (pf->fdif.field_diffusion_defined)
        pf->fdif.SetDiffusivity(ph->w, pf->bcc);
    }
  }
}

void Mesh::FinalizeHydro_pgen(std::vector<MeshBlock*> & pmb_array)
{
#if FLUID_ENABLED
  MeshBlock *pmb;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;

  #pragma omp for private(pmb, ph)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];
    ph = pmb->phydro;

    const int il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
    const int jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
    const int kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;

    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
  }
#endif // FLUID_ENABLED
}

void Mesh::FinalizeHydroPrimRP(std::vector<MeshBlock*> & pmb_array)
{
#if FLUID_ENABLED
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, ph, ps, pf)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pf = pmb->pfield;
    ph = pmb->phydro;
    ps = pmb->pscalars;

    if (multilevel)
    {
      pbval->ProlongateBoundariesHydro(time, 0.0);
    }

    // BoundaryVariable interface from conserved to primitive
    // formulations:
    pmb->SetBoundaryVariablesPrimitive();

    // N.B.
    // Results in two-fold application of BC to magnetic fields;
    // but that is harmless
    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsMatter(),
      pmb->is, pmb->ie,
      pmb->js, pmb->je,
      pmb->ks, pmb->ke,
      NGHOST);

    if (MAGNETIC_FIELDS_ENABLED)
    {
      const int il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
      const int jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
      const int kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;

      pf->CalculateCellCenteredField(pf->b, pf->bcc, pmb->pcoord,
                                     il, iu, jl, ju, kl, ku);
    }

    pbval->PrimitiveToConservedOnPhysicalBoundaries();
    pmb->SetBoundaryVariablesConserved();
  }
#endif // FLUID_ENABLED
}

void Mesh::FinalizeHydroConsRP(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, ph, ps,pf)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    pf = pmb->pfield;
    ph = pmb->phydro;
    ps = pmb->pscalars;

    if (multilevel)
    {
      pbval->ProlongateBoundariesHydro(time, 0.0);
    }

    // N.B.
    // Results in two-fold application of BC to magnetic fields;
    // but that is harmless
    pbval->ApplyPhysicalBoundaries(
      time, 0.0,
      pbval->GetBvarsMatter(),
      pmb->is, pmb->ie,
      pmb->js, pmb->je,
      pmb->ks, pmb->ke,
      NGHOST);

    if (MAGNETIC_FIELDS_ENABLED)
    {
      const int il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
      const int jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
      const int kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;

      pf->CalculateCellCenteredField(pf->b, pf->bcc, pmb->pcoord,
                                     il, iu, jl, ju, kl, ku);
    }
  }
}

void Mesh::PreparePrimitives(std::vector<MeshBlock*> & pmb_array,
                             const bool interior_only)
{
#if FLUID_ENABLED
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, pf, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    ps = pmb->pscalars;

    int il = pmb->is, iu = pmb->ie,
        jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;

    if (!interior_only)
    {
      il = 0, iu = (pmb->ncells1 > 1)? pmb->ncells1 - 1 : 0;
      jl = 0, ju = (pmb->ncells2 > 1)? pmb->ncells2 - 1 : 0;
      kl = 0, ku = (pmb->ncells3 > 1)? pmb->ncells3 - 1 : 0;
    }

    static const int coarseflag = 0;
    pmb->peos->ConservedToPrimitive(ph->u, ph->w1, ph->w,
                                    ps->s, ps->r,
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku,
                                    coarseflag);

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
  }

#endif // FLUID_ENABLED
}

void Mesh::PreparePrimitivesGhosts(std::vector<MeshBlock*> & pmb_array)
{
#if FLUID_ENABLED
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Field *pf = nullptr;
  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, pf, ph, ps)
  for (int i = 0; i < nmb; ++i)
  {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    ps = pmb->pscalars;

    int il = 0, iu = pmb->ncells1 - 1,
        jl = 0, ju = pmb->ncells2 - 1,
        kl = 0, ku = pmb->ncells3 - 1;

    static const int coarseflag = 0;
    static const bool skip_physical = true;
    pmb->peos->ConservedToPrimitive(ph->u, ph->w1, ph->w,
                                    ps->s, ps->r,
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku,
                                    coarseflag, skip_physical);

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
  }

#endif // FLUID_ENABLED
}

void Mesh::CommunicateConserved(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;
  Field *pf = nullptr;
  M1::M1 *pm1 = nullptr;
  PassiveScalars *ps = nullptr;
  Wave *pw = nullptr;
  Z4c *pz = nullptr;

  // prepare to receive conserved variables
  #pragma omp for private(pmb, pbval)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;
    pbval->StartReceiving(BoundaryCommSubset::mesh_init);
  }

  // send conserved variables
  #pragma omp for private(pmb, pbval, ph, pf, pm1, ps, pw, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    pm1 = pmb->pm1;
    ps = pmb->pscalars;
    pw = pmb->pwave;
    pz = pmb->pz4c;

    pmb->SetBoundaryVariablesConserved();

    if (FLUID_ENABLED)
    {
      ph->hbvar.SendBoundaryBuffers();
    }

    if (MAGNETIC_FIELDS_ENABLED)
      pf->fbvar.SendBoundaryBuffers();

    // and (conserved variable) passive scalar:
    if (NSCALARS > 0)
      ps->sbvar.SendBoundaryBuffers();

    if (WAVE_ENABLED) {
      if (WAVE_CC_ENABLED) {
        pw->ubvar_cc.SendBoundaryBuffers();
      } else if (WAVE_VC_ENABLED) {
        pw->ubvar_vc.SendBoundaryBuffers();
      } else if (WAVE_CX_ENABLED) {
        pw->ubvar_cx.SendBoundaryBuffers();
      }
    }

    if (Z4C_ENABLED)
      pz->ubvar.SendBoundaryBuffers();

    if (M1_ENABLED)
      pm1->ubvar.SendBoundaryBuffers();
  }

  // wait to receive conserved variables
  #pragma omp for private(pmb, pbval, ph, pf, pm1, ps, pw, pz)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    pm1 = pmb->pm1;
    ps = pmb->pscalars;
    pw = pmb->pwave;
    pz = pmb->pz4c;

    if (FLUID_ENABLED)
      ph->hbvar.ReceiveAndSetBoundariesWithWait();

    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->fbvar.ReceiveAndSetBoundariesWithWait();
    }

    if (NSCALARS > 0)
      ps->sbvar.ReceiveAndSetBoundariesWithWait();

    if (WAVE_ENABLED) {
      if (WAVE_CC_ENABLED) {
        pw->ubvar_cc.ReceiveAndSetBoundariesWithWait();
      } else if (WAVE_VC_ENABLED) {
        pw->ubvar_vc.ReceiveAndSetBoundariesWithWait();
      } else if (WAVE_CX_ENABLED) {
        pw->ubvar_cx.ReceiveAndSetBoundariesWithWait();
      }
    }

    if (Z4C_ENABLED)
      pmb->pz4c->ubvar.ReceiveAndSetBoundariesWithWait();

    if (M1_ENABLED)
      pmb->pm1->ubvar.ReceiveAndSetBoundariesWithWait();

    pbval->ClearBoundary(BoundaryCommSubset::mesh_init);
  }
}

void Mesh::CommunicateConservedMatter(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;
  Field *pf = nullptr;
  PassiveScalars *ps = nullptr;

  // prepare to receive conserved variables
  #pragma omp for private(pmb, pbval)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;
    pbval->StartReceiving(BoundaryCommSubset::matter);
  }

  // send conserved variables
  #pragma omp for private(pmb, pbval, ph, pf, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    ps = pmb->pscalars;

    pmb->SetBoundaryVariablesConserved();

#if FLUID_ENABLED
      ph->hbvar.SendBoundaryBuffers();
#endif // FLUID_ENABLED

    if (MAGNETIC_FIELDS_ENABLED)
      pf->fbvar.SendBoundaryBuffers();

    // and (conserved variable) passive scalar:
    if (NSCALARS > 0)
      ps->sbvar.SendBoundaryBuffers();
  }

  // wait to receive conserved variables
  #pragma omp for private(pmb, pbval, ph, pf, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    pf = pmb->pfield;
    ps = pmb->pscalars;

    if (FLUID_ENABLED)
      ph->hbvar.ReceiveAndSetBoundariesWithWait();

    if (MAGNETIC_FIELDS_ENABLED)
      pf->fbvar.ReceiveAndSetBoundariesWithWait();

    if (NSCALARS > 0)
      ps->sbvar.ReceiveAndSetBoundariesWithWait();

    pbval->ClearBoundary(BoundaryCommSubset::matter);
  }
}

void Mesh::CommunicatePrimitives(std::vector<MeshBlock*> & pmb_array)
{
#if FLUID_ENABLED
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;
  Field *pf = nullptr;
  PassiveScalars *ps = nullptr;

  // prepare to receive primitives
  #pragma omp for private(pmb, pbval)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;
    pbval->StartReceiving(BoundaryCommSubset::matter_primitives);
  }

  // send primitives
  #pragma omp for private(pmb, pbval, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    ps = pmb->pscalars;

    pmb->SetBoundaryVariablesPrimitive();

    ph->hbvar.SendBoundaryBuffers();

    if (NSCALARS > 0) {
      ps->sbvar.SendBoundaryBuffers();
    }
  }

  // wait to receive AMR/SMR GR primitives
  #pragma omp for private(pmb, pbval, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    ph = pmb->phydro;
    ps = pmb->pscalars;

    ph->hbvar.ReceiveAndSetBoundariesWithWait();

    if (NSCALARS > 0)
      ps->sbvar.ReceiveAndSetBoundariesWithWait();

    pbval->ClearBoundary(BoundaryCommSubset::matter_primitives);

    // Revert to conserved representation
    pmb->SetBoundaryVariablesConserved();
  }
#endif // FLUID_ENABLED
}

void Mesh::CommunicateAuxZ4c()
{
  // short-circuit if this is called without z4c
  if (!Z4C_ENABLED)
  {
    return;
  }

  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);


  // initialize a vector of MeshBlock pointers
  nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  if (static_cast<unsigned int>(nmb) != pmb_array.size()) pmb_array.resize(nmb);
  MeshBlock *pmbl = pblock;
  for (int i=0; i<nmb; ++i) {
    pmb_array[i] = pmbl;
    pmbl = pmbl->next;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    MeshBlock *pmb = nullptr;
    BoundaryValues *pbval = nullptr;
    Z4c *pz = nullptr;

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pbval->StartReceiving(BoundaryCommSubset::aux_z4c);
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;
      pz->abvar.SendBoundaryBuffers();
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;
      pz->abvar.ReceiveAndSetBoundariesWithWait();
      pbval->ClearBoundary(BoundaryCommSubset::aux_z4c);
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;

      if (multilevel)
      {
        // Handle aux. coarse MeshBlock boundaries
        pbval->ProlongateBoundariesAux(time, 0);
      }

      // Handle aux. fund. MeshBlock boundaries
      pbval->ApplyPhysicalBoundaries(
        time, 0.0,
        pbval->GetBvarsAux(),
        pz->mbi.il, pz->mbi.iu,
        pz->mbi.jl, pz->mbi.ju,
        pz->mbi.kl, pz->mbi.ku,
        pz->mbi.ng);
    }

  }
}

void Mesh::CommunicateAuxADM()
{
  // short-circuit if this is called without z4c or no com. is required
  if (!Z4C_ENABLED || !pblock->pz4c->opt.communicate_aux_adm)
  {
    return;
  }

  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  (void)nthreads;
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);


  // initialize a vector of MeshBlock pointers
  nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  if (static_cast<unsigned int>(nmb) != pmb_array.size()) pmb_array.resize(nmb);
  MeshBlock *pmbl = pblock;
  for (int i=0; i<nmb; ++i) {
    pmb_array[i] = pmbl;
    pmbl = pmbl->next;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    MeshBlock *pmb = nullptr;
    BoundaryValues *pbval = nullptr;
    Z4c *pz = nullptr;

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pbval->StartReceiving(BoundaryCommSubset::aux_adm);
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;
      pz->adm_abvar->SendBoundaryBuffers();
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;
      pz->adm_abvar->ReceiveAndSetBoundariesWithWait();
      pbval->ClearBoundary(BoundaryCommSubset::aux_adm);
    }

    #pragma omp for private(pmb,pbval,pz)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;
      pz = pmb->pz4c;

      if (multilevel)
      {
        // Handle aux. coarse MeshBlock boundaries
        pbval->ProlongateBoundariesAuxADM(time, 0);
      }

      // Handle aux. fund. MeshBlock boundaries
      pbval->ApplyPhysicalBoundaries(
        time, 0.0,
        pbval->GetBvarsAuxADM(),
        pz->mbi.il, pz->mbi.iu,
        pz->mbi.jl, pz->mbi.ju,
        pz->mbi.kl, pz->mbi.ku,
        pz->mbi.ng);
    }

  }
}

void Mesh::CommunicateIteratedZ4c(const int iterations)
{

#if defined(Z4C_CX_ENABLED)

  if (iterations > 0)
  {
    int inb = nbtotal;
    int nthreads = GetNumMeshThreads();
    (void)nthreads;
    int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
    std::vector<MeshBlock*> pmb_array(nmb);

    // initialize a vector of MeshBlock pointers
    nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
    if (static_cast<unsigned int>(nmb) != pmb_array.size()) pmb_array.resize(nmb);
    MeshBlock *pmbl = pblock;
    for (int i=0; i<nmb; ++i) {
      pmb_array[i] = pmbl;
      pmbl = pmbl->next;
    }

    for (int iter=0; iter<iterations; ++iter)
    {
      #pragma omp parallel num_threads(nthreads)
      {
        MeshBlock *pmb = nullptr;
        BoundaryValues *pbval = nullptr;
        Z4c *pz = nullptr;

        #pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pbval = pmb->pbval;
          pbval->StartReceiving(BoundaryCommSubset::iterated_z4c);
        }

        #pragma omp for private(pmb,pbval,pz)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pbval = pmb->pbval;
          pz = pmb->pz4c;
          pz->rbvar.SendBoundaryBuffersFullRestriction();
        }

        #pragma omp for private(pmb,pbval,pz)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pbval = pmb->pbval;
          pz = pmb->pz4c;
          pz->rbvar.ReceiveAndSetBoundariesWithWait();
          pbval->ClearBoundary(BoundaryCommSubset::iterated_z4c);
        }

        #pragma omp for private(pmb,pbval,pz)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pbval = pmb->pbval;
          pz = pmb->pz4c;

          // RBC uses storage.u & coarse_u_
          // Therefore can reuse the usual interface
          if (multilevel)
          {
            pbval->ProlongateBoundariesZ4c(time, 0);
          }

          pbval->ApplyPhysicalBoundaries(
            time, 0.0,
            pbval->GetBvarsZ4c(),
            pz->mbi.il, pz->mbi.iu,
            pz->mbi.jl, pz->mbi.ju,
            pz->mbi.kl, pz->mbi.ku,
            pz->mbi.ng);

        }
      }
    }
  }
#endif // Z4C_CX_ENABLED
}

// Communicate only matter fields
void Mesh::ScatterMatter(std::vector<MeshBlock*> & pmb_array)
{
  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  #pragma omp parallel num_threads(nthreads)
  {
    MeshBlock *pmb;
    BoundaryValues *pbval;

    CommunicateConservedMatter(pmb_array);

    // Treat R/P with prim_rp in the case of fluid + gr + z4c -----------------
    //
    // This requires:
    // - ConservedToPrimitive on interior (physical) grid points
    // - communication of primitives
#if FLUID_ENABLED && GENERAL_RELATIVITY && !defined(DBG_USE_CONS_BC)
    if (multilevel)
    {
      const bool interior_only = true;
      PreparePrimitives(pmb_array, interior_only);
      CommunicatePrimitives(pmb_array);
    }
#endif
    // ------------------------------------------------------------------------

    // Deal with matter prol. & BC --------------------------------------------
#if FLUID_ENABLED && !defined(DBG_USE_CONS_BC)
    FinalizeHydroPrimRP(pmb_array);
#elif FLUID_ENABLED && defined(DBG_USE_CONS_BC)
    FinalizeHydroConsRP(pmb_array);

    const bool interior_only = false;
    PreparePrimitives(pmb_array, interior_only);
#endif
    // ------------------------------------------------------------------------

#if FLUID_ENABLED && Z4C_ENABLED
    // Prepare ADM sources
    // Requires B-field in ghost-zones
    FinalizeZ4cADM_Matter(pmb_array);
#endif

  } // omp parallel


}

// Compute global minima of various quantities
void Mesh::GlobalExtrema()
{
  enum vars_min {
    IX_min_adm_alpha,
    IX_min_hydro_cons_D,
    N_vars_min
  };

  enum vars_max {
    IX_max_adm_alpha=N_vars_min,
    IX_max_hydro_cons_D,
    N_vars
  };

  AA res_V;
  res_V.NewAthenaArray(N_vars);
  res_V.Fill(std::numeric_limits<Real>::infinity());

  int nthreads = GetNumMeshThreads();
  (void)nthreads;

  int nmb = -1;
  std::vector<MeshBlock*> pmb_array;

  GetMeshBlocksMyRank(pmb_array);
  nmb = pmb_array.size();

  // minima-per-MeshBlock and then reduce
  AA res_V_mb;
  res_V_mb.NewAthenaArray(N_vars, nmb);
  res_V_mb.Fill(std::numeric_limits<Real>::infinity());

  // Per-MeshBlock parallism
  #pragma omp parallel for num_threads(nthreads)
  for (int nix = 0; nix < nmb; ++nix)
  {
    MeshBlock *pmb = pmb_array[nix];
    Hydro * ph = pmb->phydro;

    Z4c * pz4c = pmb->pz4c;

#if Z4C_ENABLED
    AA adm_alpha;
    adm_alpha.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_alpha, 1);

    AA ms_adm_sqrt_det_gamma;
    ms_adm_sqrt_det_gamma.InitWithShallowSlice(
      pz4c->storage.aux_extended, Z4c::I_AUX_EXTENDED_ms_sqrt_detgamma, 1
    );
#endif // Z4C_ENABLED

#if FLUID_ENABLED && Z4C_ENABLED
    AA hydro_cons_D;
    hydro_cons_D.InitWithShallowSlice(
      ph->u, IDN, 1
    );
#endif // FLUID_ENABLED

#if Z4C_ENABLED
    ILOOP2(k, j)
    for (int i=pz4c->mbi.il; i<=pz4c->mbi.iu; ++i)
    {
      res_V_mb(vars_min::IX_min_adm_alpha, nix) = std::min(
        res_V_mb(vars_min::IX_min_adm_alpha, nix),
        adm_alpha(k,j,i)
      );

      res_V_mb(vars_max::IX_max_adm_alpha, nix) = std::min(
        res_V_mb(vars_max::IX_max_adm_alpha, nix),
        -adm_alpha(k,j,i)
      );
    }
#endif // Z4C_ENABLED

#if FLUID_ENABLED && Z4C_ENABLED
    CC_ILOOP2(k, j)
    for (int i=pmb->is; i<=pmb->ie; ++i)
    {
      const Real oo_sqrt_detgamma = OO(
        ms_adm_sqrt_det_gamma(k,j,i)
      );
      const Real hydro_cons_D__ = hydro_cons_D(k,j,i) * oo_sqrt_detgamma;

      res_V_mb(vars_min::IX_min_hydro_cons_D, nix) = std::min(
        res_V_mb(vars_min::IX_min_hydro_cons_D, nix),
        hydro_cons_D__
      );

      res_V_mb(vars_max::IX_max_hydro_cons_D, nix) = std::min(
        res_V_mb(vars_max::IX_max_hydro_cons_D, nix),
        -hydro_cons_D__
      );
    }
#endif // FLUID_ENABLED && Z4C_ENABLED

  }

  // reduce over MeshBlock
  for (int vix = 0; vix < N_vars; ++vix)
  for (int nix = 0; nix < nmb; ++nix)
  {
    res_V(vix) = std::min(
      res_V(vix), res_V_mb(vix, nix)
    );
  }

#ifdef MPI_PARALLEL
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Allreduce(MPI_IN_PLACE, res_V.data(),
    N_vars,
    MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  // flip sign to get max for salient variables
  for (int vix=0; vix < N_vars-N_vars_min; ++vix)
  {
    res_V(vix+N_vars_min) = -res_V(vix+N_vars_min);
  }

  global_extrema.min_adm_alpha = res_V(vars_min::IX_min_adm_alpha);
  global_extrema.max_adm_alpha = res_V(vars_max::IX_max_adm_alpha);

  global_extrema.min_hydro_cons_D = res_V(vars_min::IX_min_hydro_cons_D);
  global_extrema.max_hydro_cons_D = res_V(vars_max::IX_max_hydro_cons_D);
}

//
// :D
//
