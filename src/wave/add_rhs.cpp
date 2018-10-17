//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file add_rhs.cpp
//  \brief adds the wave equation RHS to the state vector

// Athena++ headers
#include "wave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//! \fn void Wave::AddRHSToVals
//  \brief Adds the RHS to the weighted average of conservative variables from
//  previous step(s) of the time integrator

//void Wave::AddWaveRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
//    IntegratorWeight w, AthenaArray<Real> &u_out) {
//  MeshBlock *pmb=pmy_block;
//  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
//  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

//  int tid=0;
//  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();
//#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
//  {
//#ifdef OPENMP_PARALLEL
//    tid=omp_get_thread_num();
//#endif
//#pragma omp for schedule(static)
//    for(int k=ks; k<=ke; ++k) {
//      for(int j=js; j<=je; ++j) {
//#pragma omp simd
//        for(int i=is; i<=ie; ++i) {
//          // RK2:
//          u_out(0,k,j,i) = w.gamma_2*u1(0,k,j,i) + w.gamma_1*u2(0,k,j,i) +
//            w.beta*(pmb->pmy_mesh->dt)*rhs(0,k,j,i);
//          u_out(1,k,j,i) = w.gamma_2*u1(1,k,j,i) + w.gamma_1*u2(1,k,j,i) +
//            w.beta*(pmb->pmy_mesh->dt)*rhs(1,k,j,i);
//        }
//      }
//    }
//  } // end of omp parallel region

//  return;
//}
//................................................................................
//----------------------------------------------------------------------------------------
//! \fn  void Wave::AddWaveRHS
//  \brief Adds RHS to weighted average of variables from
//  previous step(s) of time integrator algorithm

void Wave::AddWaveRHS(const Real wght, AthenaArray<Real> &u_out) {

  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {

      // update variables
      for (int n=0; n<2; ++n) { //Gotta fix this. We need the analog of NHYDRO
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          u_out(n,k,j,i) += wght*(pmb->pmy_mesh->dt)*rhs(n,k,j,i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::WeightedAveW
//  \brief Compute weighted average of cell-averaged U in time integrator step

void Wave::WeightedAveW(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                         AthenaArray<Real> &u_in2, const Real wght[3]) {
  MeshBlock *pmb=pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2
  // if c=0, c=b=0, or c=b=a=0 (in that order) to avoid extra FMA operations

  // u_in2 may be an unallocated AthenaArray if using a 2S time integrator
  if (wght[2] != 0.0) {
    for (int n=0; n<2; ++n) { //Will fix this. We need the analog of NHYDRO instead of '2'
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            u_out(n,k,j,i) = wght[0]*u_out(n,k,j,i) + wght[1]*u_in1(n,k,j,i)
                + wght[2]*u_in2(n,k,j,i);
          }
        }
      }
    }
  } else { // do not dereference u_in2
    if (wght[1] != 0.0) {
      for (int n=0; n<2; ++n) { //Will fix this. We need the analog of NHYDRO instead of '2'
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              u_out(n,k,j,i) = wght[0]*u_out(n,k,j,i) + wght[1]*u_in1(n,k,j,i);
            }
          }
        }
      }
    } else { // do not dereference u_in1
      if (wght[0] != 0.0) {
        for (int n=0; n<2; ++n) { //Will fix this. We need the analog of NHYDRO instead of '2'
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                u_out(n,k,j,i) = wght[0]*u_out(n,k,j,i);
              }
            }
          }
        }
      } else { // directly initialize u_out to 0
        for (int n=0; n<2; ++n) { //Will fix this. We need the analog of NHYDRO instead of '2'
          for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma omp simd
              for (int i=is; i<=ie; ++i) {
                u_out(n,k,j,i) = 0.0;
              }
            }
          }
        }
      }
    }
  }
  return;
}

