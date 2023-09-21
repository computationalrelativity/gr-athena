#ifndef DUMMY_HPP
#define DUMMY_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave.hpp
//  \brief definitions for the Dummy class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

class MeshBlock;
class ParameterInput;

//! \class Dummy
//  \brief Dummy data and functions

class Dummy {
public:
  Dummy(MeshBlock *pmb, ParameterInput *pin);
  ~Dummy();

  struct MB_info {
    int il, iu, jl, ju, kl, ku;        // local block iter.
    int nn1, nn2, nn3;                 // number of nodes (simplify switching)

    AthenaArray<Real> x1, x2, x3;      // for CC / VC grid switch
    AthenaArray<Real> cx1, cx2, cx3;   // for CC / VC grid switch (coarse)

    // provide coarse analogues of the above
    // int cnn1, cnn2, cnn3;
    // int cil, ciu, cjl, cju, ckl, cku;
  };

  MB_info mbi;

  struct {
    // Sphere-zone refinement
    int sphere_zone_number;
    AthenaArray<int> sphere_zone_levels;
    AthenaArray<Real> sphere_zone_radii;
    AthenaArray<int> sphere_zone_puncture;
    AthenaArray<Real> sphere_zone_center1;
    AthenaArray<Real> sphere_zone_center2;
    AthenaArray<Real> sphere_zone_center3;
  } opt;


  // data
  MeshBlock *pmy_block;         // ptr to MeshBlock containing this Dummy

};
#endif // DUMMY_HPP
