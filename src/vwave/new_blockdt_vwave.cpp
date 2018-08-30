//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//  \brief computes timestep using CFL condition on a MEshBlock

// C/C++ headers
#include <algorithm>  // min()
#include <cfloat>     // FLT_MAX
#include <cmath>      // fabs(), sqrt()

// Athena++ headers
#include "vect_wave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//! \fn Real Vwave::NewBlockTimeStep(void)
//  \brief calculate the minimum timestep within a MeshBlock

Real Vwave::NewBlockTimeStep(void)
{
  MeshBlock * pmb = pmy_block;
  int tid = 0;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

  Real min_dt = (FLT_MAX);
#pragma omp parallel default(shared) private(tid) num_threads(nthreads) reduction(min: min_dt)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif
    AthenaArray<Real> dt1, dt2, dt3;
    dt1.InitWithShallowSlice(dt1_, 2, tid, 1);
    dt2.InitWithShallowSlice(dt2_, 2, tid, 1);
    dt3.InitWithShallowSlice(dt3_, 2, tid, 1);

    for (int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for (int j = js; j <= je; ++j) {
        pmb->pcoord->CenterWidth1(k,j,is,ie,dt1);
        pmb->pcoord->CenterWidth2(k,j,is,ie,dt2);
        pmb->pcoord->CenterWidth3(k,j,is,ie,dt3);
        for (int i = is; i <= ie; ++i) {
          dt1(i) /= c;
          dt2(i) /= c;
          dt3(i) /= c;
        }
        for (int i = is; i <= ie; ++i) {
          Real & dt_1 = dt1(i);
          min_dt = std::min(min_dt, dt_1);
        }
        if (pmb->block_size.nx2 > 1) {
          for (int i = is; i <= ie; ++i) {
            Real & dt_2 = dt2(i);
            min_dt = std::min(min_dt, dt_2);
          }
        }
        if (pmb->block_size.nx3 > 1) {
          for (int i = is; i <= ie; ++i) {
            Real & dt_3 = dt3(i);
            min_dt = std::min(min_dt, dt_3);
          }
        }
      }
    }
  } // end of omp parallel region
  min_dt *= pmb->pmy_mesh->cfl_number;

  pmb->new_block_dt = min_dt;
  return min_dt;
}
