// C headers

// C++ headers
#include <iostream>

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

    pbval->ApplyPhysicalBoundaries(time, 0.0);
  }
}

void Mesh::FinalizeZ4cADM(std::vector<MeshBlock*> & pmb_array)
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

    // Switch based on Z4c sampling
    FCN_CC_CX_VC(
      pbval->ApplyPhysicalBoundaries,
      pbval->ApplyPhysicalCellCenteredXBoundaries,
      pbval->ApplyPhysicalVertexCenteredBoundaries
    )(time, 0);

    // Enforce the algebraic constraints
    pz->AlgConstr(pz->storage.u);
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

#if USETM
    pz->GetMatter(pz->storage.mat, pz->storage.adm, ph->w, ps->r, pf->bcc);
#else
    pz->GetMatter(pz->storage.mat, pz->storage.adm, ph->w, pf->bcc);
#endif // USETM

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

    pbval->ApplyPhysicalBoundariesM1(time, 0.0);
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

void Mesh::FinalizeHydroPrimRP(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;

    if (multilevel)
    {
      pbval->ProlongateBoundariesHydro(time, 0.0);
    }

    // Swap Hydro and passive scalar quantities in
    // BoundaryVariable interface from conserved to primitive
    // formulations:
    ph->hbvar.SwapHydroQuantity(ph->w, HydroBoundaryQuantity::prim);
    if (NSCALARS > 0)
      ps->sbvar.var_cc = &(ps->r);

    pbval->ApplyPhysicalBoundaries(time, 0.0);

    ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);
  }
}

void Mesh::FinalizeHydroConsRP(std::vector<MeshBlock*> & pmb_array)
{
  MeshBlock *pmb;
  BoundaryValues *pbval;

  const int nmb = pmb_array.size();

  Hydro *ph = nullptr;
  PassiveScalars *ps = nullptr;

  #pragma omp for private(pmb, pbval, ph, ps)
  for (int i = 0; i < nmb; ++i) {
    pmb = pmb_array[i];
    pbval = pmb->pbval;

    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;

    if (multilevel)
    {
      pbval->ProlongateBoundariesHydro(time, 0.0);
    }
    pbval->ApplyPhysicalBoundaries(time, 0.0);
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
      il = 0;
      iu = pmb->ncells1;

      jl = 0;
      ju = pmb->ncells2;

      kl = 0;
      ku = pmb->ncells3;
    }

    static const int coarse_flag = 0;
    pmb->peos->ConservedToPrimitive(ph->u, ph->w1, pf->b, ph->w,
#if USETM
                                    ps->s, ps->r,
#endif
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku, coarse_flag);
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

    if (FLUID_ENABLED) {
      ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);
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
      pf->fbvar.ReceiveAndSetBoundariesWithWait();

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

    if (FLUID_ENABLED) {
      ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);
      ph->hbvar.SendBoundaryBuffers();
    }

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

    ph->hbvar.SwapHydroQuantity(ph->w, HydroBoundaryQuantity::prim);
    ph->hbvar.SendBoundaryBuffers();

    if (NSCALARS > 0) {
      ps->sbvar.var_cc = &(ps->r);
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

    ph->hbvar.SwapHydroQuantity(ph->u, HydroBoundaryQuantity::cons);

    if (NSCALARS > 0)
      ps->sbvar.var_cc = &(ps->s);
  }

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
    MeshBlock *pmb;
    BoundaryValues *pbval;

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i]; pbval = pmb->pbval;
      pbval->StartReceiving(BoundaryCommSubset::aux_z4c);
    }

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i]; pbval = pmb->pbval;
      pmb->pz4c->abvar.SendBoundaryBuffers();
    }

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i]; pbval = pmb->pbval;
      pmb->pz4c->abvar.ReceiveAndSetBoundariesWithWait();
      pbval->ClearBoundary(BoundaryCommSubset::aux_z4c);
    }

    #pragma omp for private(pmb,pbval)
    for (int i=0; i<nmb; ++i)
    {
      pmb = pmb_array[i];
      pbval = pmb->pbval;

      if (multilevel)
      {
        // Handle aux. coarse MeshBlock boundaries
        pbval->ProlongateBoundariesAux(time, 0);
      }

      // Handle aux. fund. MeshBlock boundaries
      pbval->ApplyPhysicalBoundariesAux(time, 0);
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
        MeshBlock *pmb;
        BoundaryValues *pbval;

        #pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pbval->StartReceiving(BoundaryCommSubset::iterated_z4c);
        }

        #pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pmb->pz4c->rbvar.SendBoundaryBuffersFullRestriction();
        }

        #pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pmb->pz4c->rbvar.ReceiveAndSetBoundariesWithWait();
          pbval->ClearBoundary(BoundaryCommSubset::iterated_z4c);
        }

        #pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i)
        {
          pmb = pmb_array[i];
          pbval = pmb->pbval;

          // RBC uses storage.u & coarse_u_
          // Therefore can reuse the usual interface
          if (multilevel)
          {
            pbval->ProlongateBoundariesZ4c(time, 0);
          }

          pbval->ApplyPhysicalCellCenteredXBoundaries(time, 0);
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

//
// :D
//
