//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_rhs.cpp
//  \brief Calculate wave equation RHS

// C/C++ headers
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error

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

static Real const diff_2_ord_2[] = {
  1., -2., 1.
};
static Real const diff_2_ord_4[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.
};
static Real const diff_2_ord_6[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.
};
static Real const diff_2_ord_8[] = {
  -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.
};

//! \fn void Wave::CalculateRHS
//  \brief Calculate RHS for the wave equation using finite-differencing

void Wave::WaveRHS(AthenaArray<Real> & u, int order)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real const * stencil;
  switch(order) {
    case 2:
      stencil = &diff_2_ord_2[0];
      break;
    case 4:
      stencil = &diff_2_ord_4[0];
      break;
    case 6:
      stencil = &diff_2_ord_6[0];
      break;
    case 8:
      stencil = &diff_2_ord_8[0];
      break;
    default:
      msg << "FD order not supported: " << order << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  int const stencil_offset = order/2;
  int const stencil_size = order + 1;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();
#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

//----------------------------------------------------------------------------------------
// i-direction

    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
#pragma simd
        for(int i = is; i <= ie; ++i) {
          rhs(0,k,j,i) = u(1,k,j,i);

          // assume Cartesian coordinates!
          Real rhs_loc = 0;
          for(int n = 0; n < stencil_size; ++n) {
            rhs_loc += stencil[n]*u(0, k, j, i + n - stencil_offset);
          }
          rhs_loc *= SQR(c)/SQR(pco->dx1v(i));
          rhs(1,k,j,i) = rhs_loc;
        }
      }
    }

//----------------------------------------------------------------------------------------
// j-direction

    if(pmb->block_size.nx2 > 1) {
      for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
        for(int j = js; j <= je; ++j) {
#pragma simd
          for(int i = is; i <= ie; ++i) {
            // assume Cartesian coordinates!
            Real rhs_loc = 0;
            for(int n = 0; n < stencil_size; ++n) {
              rhs_loc += stencil[n]*u(0, k, j + n - stencil_offset, i);
            }
            rhs_loc *= SQR(c)/SQR(pco->dx2v(j));
            rhs(1,k,j,i) += rhs_loc;
          }
        }
      }
    }

//----------------------------------------------------------------------------------------
// k-direction

    if(pmb->block_size.nx3 > 1) {
      for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
        for(int j = js; j <= je; ++j) {
#pragma simd
          for(int i = is; i <= ie; ++i) {
            // assume Cartesian coordinates!
            Real rhs_loc = 0;
            for(int n = 0; n < stencil_size; ++n) {
              rhs_loc += stencil[n]*u(0, k + n - stencil_offset, j, i);
            }
            rhs_loc *= SQR(c)/SQR(pco->dx3v(k));
            rhs(1,k,j,i) += rhs_loc;
          }
        }
      }
    }
  } // end of parallel region

  return;
}
