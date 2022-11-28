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
  
  // finite difference error must fall less that a prescribed value.
  if (pin->GetOrAddString("z4c","refinement_method","FD_error") == "FD_error")
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
  int ret = 0;
  double err = 0.;
  ParameterInput *const pin = pmb->pmy_in;
  const double dmax= std::numeric_limits<double>::max();
  const double dmin=-std::numeric_limits<double>::max();
  const double ref_tol   = pin->GetOrAddReal("z4c","refinement_tol",1e-5);
  const double dref_tol  = pin->GetOrAddReal("z4c","derefinement_tol",1e-6);
  const double ref_x1min = pin->GetOrAddReal("z4c","refinement_x1min",dmin);
  const double ref_x1max = pin->GetOrAddReal("z4c","refinement_x1max",dmax);
  const double ref_x2min = pin->GetOrAddReal("z4c","refinement_x2min",dmin);
  const double ref_x2max = pin->GetOrAddReal("z4c","refinement_x2max",dmax);
  const double ref_x3min = pin->GetOrAddReal("z4c","refinement_x3min",dmin);
  const double ref_x3max = pin->GetOrAddReal("z4c","refinement_x3max",dmax);
  const int ref_deriv = pin->GetOrAddReal("z4c","refinement_deriv_order",7);
  const int ref_pow   = pin->GetOrAddReal("z4c","refinement_deriv_power",1);
  const bool verbose  = pin->GetOrAddBoolean("z4c", "refinement_verbose",false);
  char region[999] = {0};
  
  if (verbose)
    sprintf(region,"[%0.1f,%0.1f]x[%0.1f,%0.1f]x[%0.1f,%0.1f]",
            pmb->block_size.x1min,pmb->block_size.x1max,
            pmb->block_size.x2min,pmb->block_size.x2max,
            pmb->block_size.x3min,pmb->block_size.x3max);

  // calc. err
  err = pmb->pz4c->amr_err_L2_derive_chi_pow(pmb,ref_deriv,ref_pow);
  
  // check the region of interest for the refinement
  if (pmb->block_size.x1min < ref_x1min || pmb->block_size.x1max > ref_x1max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  else if (pmb->block_size.x2min < ref_x2max || pmb->block_size.x2max > ref_x2max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  else if (pmb->block_size.x3min < ref_x3max || pmb->block_size.x3max > ref_x3max)
  {
    if (verbose) 
      printf("out of bound %s.\n",region);
    ret = 0;
  }
  
  // compare with the error bounds
  else if (err > ref_tol)
  {
    if (verbose)
      printf("err > ref-tol:   %e > %e  ==> refine %s.\n",err,ref_tol,region);
    ret = 1.;
  }
  else if (err < dref_tol)
  {
    if (verbose)
      printf("err < deref-tol: %e < %e  ==> derefine %s.\n",err,dref_tol,region);
    ret = -1;
  }
  else 
  {
    if (verbose)
      printf("dref-tol <= err <= ref-tol: %e <= %e <= %e ==> nothing %s.\n",
              dref_tol,err,ref_tol,region);
    ret = 0;
  }
  
  fflush(stdout);
  
  return ret;
}
