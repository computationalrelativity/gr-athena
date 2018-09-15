//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file add_rhs.cpp
//  \brief adds the vectorial wave equation RHS to the state vector

// Athena++ headers
#include "vwave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//! \fn void Vwave::AddRHSToVals
//  \brief Adds the RHS to the weighted average of conservative variables from
//  previous step(s) of the time integrator

void Vwave::AddRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
    IntegratorWeight wght, AthenaArray<Real> &u_out) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int tid=0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();
/*
#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid=omp_get_thread_num();
#endif
#pragma omp for schedule(static)
//...............................................................................


//...............................................................................    
/*    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
#pragma simd
        for(int i=is; i<=ie; ++i) {
          u_out(0,k,j,i) = wght.a*u1(0,k,j,i) + wght.b*u2(0,k,j,i) +
            wght.c*(pmb->pmy_mesh->dt)*rhs(0,k,j,i);
          u_out(1,k,j,i) = wght.a*u1(1,k,j,i) + wght.b*u2(1,k,j,i) +
            wght.c*(pmb->pmy_mesh->dt)*rhs(1,k,j,i);
        }
      }
    }
  } // end of omp parallel region
*/
  return;
}

