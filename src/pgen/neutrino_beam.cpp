//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shock_tube.cpp
//! \brief Problem generator for shock tube problems.
//!
//! Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
//! shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), freopen(), fprintf(), fclose()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../m1/m1.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput * pin) {

  pz4c->ADMMinkowski(pz4c->storage.adm);
  pm1->beam_dir[0] = pin->GetOrAddReal("problem", "beam_dir1", 1.0);
  pm1->beam_dir[1] = pin->GetOrAddReal("problem", "beam_dir2", 0.0);
  pm1->beam_dir[2] = pin->GetOrAddReal("problem", "beam_dir3", 0.0);
  pm1->beam_position = pin->GetOrAddReal("problem", "beam_position", 0.0);
  pm1->beam_width = pin->GetOrAddReal("problem", "beam_width", 1.0);
  pm1->SetupBeamTest(pm1->storage.u);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the shock tube tests
//========================================================================================

/* void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  pz4c->ADMMinkowski(pz4c->storage.adm);
  M1::Lab_vars & vec = pm1->lab;
  Real nx = beam_dir[0];
  Real ny = beam_dir[1];
  Real nz = beam_dir[2];
  Real n2 = SQR(nx) + SQR(ny) + SQR(nz);
  if (n2 > 0.0) {
    Real nn = sqrt(n2);
    nx /= nn;
    ny /= nn;
    nz /= nn;
  } else {
    nx = 1.0;
    nz = ny = 0.0;
  }

  for (int k=ks; k<=ke; ++k) {
    Real z = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = pcoord->x1v(i);
        Real proj = nx*x + ny*y + nz*z;
        Real offset2 = SQR(x-nx*x) + SQR(y-ny*y) + SQR(z-nz*z);
        if (proj < beam_position && offset2 < SQR(beam_width)) {
          for (int ig=0; ig<(pm1->ngroups)*(pm1->nspecies); ++ig) {
            vec.N(ig,k,j,i) = 1.0;
            vec.E(ig,k,j,i) = 1.0;
            vec.F_d(0,ig,k,j,i) = nx;
            vec.F_d(1,ig,k,j,i) = ny;
            vec.F_d(2,ig,k,j,i) = nz;
          }
        } else {
          for (int ig=0; ig<(pm1->ngroups)*(pm1->nspecies); ++ig) {
            vec.N(ig,k,j,i) = 0.0;
            vec.E(ig,k,j,i) = 0.0;
            vec.F_d(0,ig,k,j,i) = 0.0;
            vec.F_d(1,ig,k,j,i) = 0.0;
            vec.F_d(2,ig,k,j,i) = 0.0;
          }
        }
      }
    }
  }

  for (int ig=0; ig<(pm1->ngroups)*(pm1->nspecies); ig++) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pm1->storage.u(0,ig,k,j,i) = vec.N(ig,k,j,i);
          pm1->storage.u(1,ig,k,j,i) = vec.E(ig,k,j,i);
          pm1->storage.u(2,ig,k,j,i) = vec.F_d(1,ig,k,j,i);
          pm1->storage.u(3,ig,k,j,i) = vec.F_d(2,ig,k,j,i);
          pm1->storage.u(4,ig,k,j,i) = vec.F_d(3,ig,k,j,i);
        }
      }
    }
  }
  return;
}
*/