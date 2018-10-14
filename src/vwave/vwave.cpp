//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vwave.hpp
//  \brief implementation of functions in the VWave class

// Athena++ headers
#include "vwave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
//#include "../athena_tensor.hpp"

// constructor, initializes data structures and parameters

Vwave::Vwave(MeshBlock *pmb, ParameterInput *pin)
{
  pmy_block = pmb;

  // Allocate memory for the solution and its time derivative
  int ncells1 = pmy_block->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if(pmy_block->block_size.nx2 > 1) ncells2 = pmy_block->block_size.nx2 + 2*(NGHOST);
  if(pmy_block->block_size.nx3 > 1) ncells3 = pmy_block->block_size.nx3 + 2*(NGHOST);

  u.  NewAthenaArray(2, ncells3, ncells2, ncells1);
  u1. NewAthenaArray(2, ncells3, ncells2, ncells1);
  rhs.NewAthenaArray(2, ncells3, ncells2, ncells1);

  int nthreads = pmy_block->pmy_mesh->GetNumMeshThreads();
  dt1_.NewAthenaArray(nthreads,ncells1);
  dt2_.NewAthenaArray(nthreads,ncells1);
  dt3_.NewAthenaArray(nthreads,ncells1);

  // Allocate memory for aux vars
  eta.NewAthenaTensor(2, ncells1); //TODO:check
  ieta.NewAthenaTensor(2, ncells1); 
  ddg.NewAthenaTensor(4, ncells1);
  R.NewAthenaTensor(2, ncells1);
}

// destructor

Vwave::~Vwave()
{
  u.DeleteAthenaArray();
  u1.DeleteAthenaArray();
  rhs.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  eta.DeleteAthenaTensor();
  ieta.DeleteAthenaTensor();
  ddg.DeleteAthenaTensor();
  R.DeleteAthenaTensor();
}

//----------------------------------------------------------------------------------------
// \!fn Real Vwave::SpatiaDet(Real gxx, ... , Real gzz)
// \brief returns determinant of 3-metric

Real Vwave::SpatiaDet(Real const gxx, Real const gxy, Real const gxz,
		      Real const gyy, Real const gyz, Real const gzz)
{
  return std::(- SQ(gxz)*gyy + 2*gxy*gxz*gyz - gxx*SQ(gyz) - SQ(gxy)*gzz + gxx*gyy*gzz);
}

//----------------------------------------------------------------------------------------
// \!fn void Vwave::SpatialInv(Real const det,
//		       Real const gxx, Real const gxy, Real const gxz,
//		       Real const gyy, Real const gyz, Real const gzz,
//		       Real * uxx, Real * uxy, Real * uxz,
//		       Real * uyy, Real * uyz, Real * uzz)
// \brief returns inverse of 3-metric

void Vwave::SpatialInv(Real const det,
		       Real const gxx, Real const gxy, Real const gxz,
		       Real const gyy, Real const gyz, Real const gzz,
		       Real * uxx, Real * uxy, Real * uxz,
		       Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQ(gyz) + gyy*gzz)/det;
  *uxy = (gxz*gyz  - gxy*gzz)/det;
  *uyy = (-SQ(gxz) + gxx*gzz)/det;
  *uxz = (-gxz*gyy + gxy*gyz)/det;
  *uyz = (gxy*gxz  - gxx*gyz)/det;
  *uzz = (-SQ(gxy) + gxx*gyy)/det;
  return;
}
