//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file awa_test.cpp
//  \brief Initial conditions for Apples with Apples Test

#include <cassert> // assert
#include <iostream>
#include <sstream>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
// #include "../athena_tensor.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
// #include "../mesh/mesh_refinement.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_amr.hpp"

//using namespace std;

static int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  pz4c->ADMOnePuncture(pin, pz4c->storage.adm);
  pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);

  std::cout << "One puncture initialized ";
  std::cout << "@ pmb.gid = " << gid << std::endl;

  // Constructing Z4c vars from ADM ones
  pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);

  return;
}


void MeshBlock::Z4cUserWorkInLoop() {

  return;
}

// 1: refines, -1: de-refines, 0: does nothing
static int RefinementCondition(MeshBlock *pmb)
{
  int ret = 0;
  ParameterInput *const pin = pmb->pmy_in;

  Z4c_AMR *amr = new Z4c_AMR(pmb);

  // finite difference error must fall less that a prescribed value.
  if (amr->ref_method == "FD_error")
  {
    ret = amr->FDErrorApprox(pmb);
  }
  else
  {
    std::stringstream msg;
    msg << "No such option for z4c/refinement" << std::endl;
    ATHENA_ERROR(msg);
  }

  delete amr;
    
  return ret;
}  

