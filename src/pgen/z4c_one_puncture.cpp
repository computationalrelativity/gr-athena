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

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
// #include "../athena_tensor.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
// #include "../mesh/mesh_refinement.hpp"
#include "../z4c/z4c.hpp"

//using namespace std;

static int RefinementCondition(MeshBlock *pmb);
static int FDErrorApprox(MeshBlock *pmb);

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  std::cout << __FUNCTION__ << std::endl;
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  std::cout << __FUNCTION__ << std::endl;
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
  std::cout << __FUNCTION__ << std::endl;
  int ret = 0;
  ParameterInput *const pin = pmb->pmy_in;
  
  // finite difference error must fall less that a prescribed value.
  if (pin->GetOrAddString("z4c","refinement","FD_error") == "FD_error")
  {
    ret = FDErrorApprox(pmb);
  }
  else
  {
    std::stringstream msg;
    msg << "No such option for z4c/refinement" << std::endl;
    ATHENA_ERROR(msg);
  }
  
  return ret;
}  

// using the FD error as an approximation of the error in the meshblock.
// if this error falls below a prescribed value, the meshblock should be refined.
static int FDErrorApprox(MeshBlock *pmb)
{
  //std::cout << __FUNCTION__ << std::endl;
  
  int ret = 0;
  double err = 0.;
  ParameterInput *const pin = pmb->pmy_in;
  double ref_tol  = pin->GetOrAddReal("z4c","refinement_tol",1e-5);
  double dref_tol = pin->GetOrAddReal("z4c","derefinement_tol",1e-6);
  
  // calc. err
  err = pmb->pz4c->amr_err_L2_ddchi_pow(pmb,3);
  
  // if it's bigger than the specified params then refine;
  if (err > ref_tol)
  {
    ret = 1.;
    printf("err > ref-tol:   %e > %e  ==> refine me!\n",err,ref_tol);
  }
  else if (err < dref_tol)
  {
    ret = -1;
    printf("err < deref-tol: %e < %e  ==> derefine me!\n",err,dref_tol);
  }
  else 
  {
    ret = 0;
    printf("dref-tol <= err <= ref-tol: %e <= %e <= %e ==> I'm good!\n",dref_tol,err,ref_tol);
  }
  
  fflush(stdout);
  
  return ret;
  
}
