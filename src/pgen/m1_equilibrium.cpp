//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_shadow.cpp
//! \brief Problem generator for M1 shadow tests (thick regime) in flat spacetime
//========================================================================================

// C headers

// C++ headers
// #include <cmath>      // sqrt()
// #include <cstdio>     // fopen(), freopen(), fprintf(), fclose()
// #include <iostream>   // endl
// #include <sstream>    // stringstream
// #include <stdexcept>  // runtime_error
// #include <string>

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

Real threshold;

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
    threshold = pin->GetReal("problem","thr");
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Problem Generator for the beam test
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput * pin) {
  pz4c->ADMMinkowski(pz4c->storage.adm);
  pz4c->GaugeGeodesic(pz4c->storage.u);
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
  pm1->SetZeroLabVars(pm1->storage.u);
}


// refinement condition: 
int RefinementCondition(MeshBlock *pmb) {
  //TODO
  return 0;
}
