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
#include "../finite_differencing.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//! \fn void Wave::CalculateRHS
//  \brief Calculate RHS for the wave equation using finite-differencing

void Wave::WaveRHS(AthenaArray<Real> & u)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  // Note: we are assuming uniform mesh here!
  Real const dx[3] = {pco->dx1v(0), pco->dx2v(0), pco->dx3v(0)};
  Real const idx[3] = {1.0/dx[0], 1.0/dx[1], 1.0/dx[2]};

  AthenaArray<Real> wu, wpi;
  wu.InitWithShallowSlice(u,0,1);
  wpi.InitWithShallowSlice(u,1,1);

  for(int k = ks; k <= ke; ++k) {
    for(int j = js; j <= je; ++j) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs(0,k,j,i) = wpi(k,j,i);
        rhs(1,k,j,i) = 0.0;
      }
      for(int a = 0; a < 3; ++a) {
        FDKernelH<FDCenteredStencil<2, NGHOST>, Real> dwu(a, wu);
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          rhs(1,k,j,i) += dwu(k,j,i)*SQR(idx[a]);
        }
      }
    }
  }
}
