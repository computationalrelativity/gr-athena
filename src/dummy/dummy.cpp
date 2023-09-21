//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dummy.cpp
//  \brief implementation of functions in the dummy class

// C++ headers
#include <iostream>
#include <string>
// #include <algorithm>  // min()
// #include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

#include "dummy.hpp"

// constructor, initializes data structures and parameters

Dummy::Dummy(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb)
{
  Mesh *pm = pmb->pmy_mesh;
  Coordinates * pco = pmb->pcoord;

  // dimensions required for data allocation
  mbi.nn1 = pmb->nverts1;
  mbi.nn2 = pmb->nverts2;
  mbi.nn3 = pmb->nverts3;

  int nn1 = mbi.nn1, nn2 = mbi.nn2, nn3 = mbi.nn3;

  // convenience for per-block iteration (private Dummy scope)
  mbi.il = pmb->is;
  mbi.iu = pmb->ive;

  mbi.jl = pmb->js;
  mbi.ju = pmb->jve;

  mbi.kl = pmb->ks;
  mbi.ku = pmb->kve;

  // point to appropriate grid
  mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, nn1);
  mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, nn2);
  mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, nn3);

  // sphere-zone refinement test [disabled by default]
  opt.sphere_zone_number = pin->GetOrAddInteger("dummy", "sphere_zone_number", 0);
  if (opt.sphere_zone_number > 0) {
    opt.sphere_zone_levels.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_radii.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_puncture.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center1.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center2.NewAthenaArray(opt.sphere_zone_number);
    opt.sphere_zone_center3.NewAthenaArray(opt.sphere_zone_number);

    for (int i=0; i<opt.sphere_zone_number; ++i) {
      opt.sphere_zone_levels(i) = pin->GetOrAddInteger("dummy",
        "sphere_zone_level_" + std::to_string(i), 0);
      opt.sphere_zone_radii(i) = pin->GetOrAddReal("dummy",
        "sphere_zone_radius_" + std::to_string(i), 0.);
      opt.sphere_zone_puncture(i) = pin->GetOrAddInteger("dummy",
        "sphere_zone_puncture_" + std::to_string(i), -1);
      // populate centers
      opt.sphere_zone_center1(i) = pin->GetOrAddReal("dummy",
        "sphere_zone_center1_" + std::to_string(i), 0.);
      opt.sphere_zone_center2(i) = pin->GetOrAddReal("dummy",
        "sphere_zone_center2_" + std::to_string(i), 0.);
      opt.sphere_zone_center3(i) = pin->GetOrAddReal("dummy",
        "sphere_zone_center3_" + std::to_string(i), 0.);
    }
  }

}


// destructor
Dummy::~Dummy()
{

  if (opt.sphere_zone_number > 0) {
    opt.sphere_zone_levels.DeleteAthenaArray();
    opt.sphere_zone_radii.DeleteAthenaArray();
    opt.sphere_zone_puncture.DeleteAthenaArray();
    opt.sphere_zone_center1.DeleteAthenaArray();
    opt.sphere_zone_center2.DeleteAthenaArray();
    opt.sphere_zone_center3.DeleteAthenaArray();
  }

}

