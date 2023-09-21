//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <cassert> // assert
#include <cmath> // abs, exp, sin, fmod
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/mesh_refinement.hpp"
#include "../dummy/dummy.hpp"

using namespace std;

int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initialize the problem.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // ...

  return;
}

void MeshBlock::WaveUserWorkInLoop() {
  // ...
  return;
}

//----------------------------------------------------------------------------------------
//! \fn
//  \brief refinement condition: simple time-dependent test

int RefinementCondition(MeshBlock *pmb){

  int root_lev = pmb->pmy_mesh->GetRootLevel();
  int level = pmb->loc.level - root_lev;

  bool need_ref = false;
  bool satisfied_ref = false;

  for (int six=0; six < pmb->pdummy->opt.sphere_zone_number; ++six)
  {
    Real xyz_wz[3] = {0., 0., 0.};

    xyz_wz[0] = pmb->pdummy->opt.sphere_zone_center1(six);
    xyz_wz[1] = pmb->pdummy->opt.sphere_zone_center2(six);
    xyz_wz[2] = pmb->pdummy->opt.sphere_zone_center3(six);

    int const lev_wz = pmb->pdummy->opt.sphere_zone_levels(six);
    Real const R_wz = pmb->pdummy->opt.sphere_zone_radii(six);

    if (lev_wz > 0)
    { // ensure currently iterated sphere actually has non-trivial level
      if (pmb->SphereIntersects(xyz_wz[0], xyz_wz[1], xyz_wz[2], R_wz))
      {
        need_ref = need_ref or (level < lev_wz);
        satisfied_ref = satisfied_ref or (level == lev_wz);
      }
    }
  }

  if (need_ref)
  {
    return 1;
  } else if (satisfied_ref)
  {
    return 0;
  }
  // force de-refine if no condition satisfied
  return -1;
}