// C headers

// C++ headers
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../globals.hpp"
#include "mesh.hpp"

#include "../z4c/z4c.hpp"

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
            pbval->ProlongateBoundaries(time, 0);
          }

          pbval->ApplyPhysicalCellCenteredXBoundaries(time, 0);
        }
      }
    }
  }
#endif // Z4C_CX_ENABLED
}

//
// :D
//
