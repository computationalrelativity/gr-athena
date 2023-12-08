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

Real threshold;

int RefinementCondition(MeshBlock *pmb);
void BeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BeamOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BeamInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BeamOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BeamInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void BeamOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x1, BeamInnerX1);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x1, BeamOuterX1);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x2, BeamInnerX2);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x2, BeamOuterX2);
  // EnrollUserBoundaryFunction(BoundaryFace::inner_x3, BeamInnerX3);
  // EnrollUserBoundaryFunction(BoundaryFace::outer_x3, BeamOuterX3);
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

  pm1->m1_test = pin->GetOrAddString("M1", "test", "beam");

  if (pm1->m1_test == "beam") {
    pz4c->ADMMinkowski(pz4c->storage.adm);
    pz4c->GaugeGeodesic(pz4c->storage.u);
    pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
    
    pm1->beam_dir[0] = pin->GetOrAddReal("problem", "beam_dir1", 1.0);
    pm1->beam_dir[1] = pin->GetOrAddReal("problem", "beam_dir2", 0.0);
    pm1->beam_dir[2] = pin->GetOrAddReal("problem", "beam_dir3", 0.0);
    pm1->beam_position[0] = pin->GetOrAddReal("problem", "beam_position_x", 0.0);
    pm1->beam_position[1] = pin->GetOrAddReal("problem", "beam_position_y", 0.0);
    pm1->beam_position[2] = pin->GetOrAddReal("problem", "beam_position_z", 0.0);
    pm1->beam_width = pin->GetOrAddReal("problem", "beam_width", 1.0);
    pm1->SetupBeamTest(pm1->storage.u);
  } else if (pm1->m1_test == "kerr_beam") {
    //pz4c->ADMOnePuncture(pin, pz4c->storage.adm);
    //pz4c->GaugePreCollapsedLapse(pz4c->storage.adm, pz4c->storage.u);

    pm1->beam_dir[0] = pin->GetOrAddReal("problem", "beam_dir1", 1.0);
    pm1->beam_dir[1] = pin->GetOrAddReal("problem", "beam_dir2", 0.0);
    pm1->beam_dir[2] = pin->GetOrAddReal("problem", "beam_dir3", 0.0);
    pm1->beam_width = pin->GetOrAddReal("problem", "beam_width", 1.0);
    pm1->beam_position[0] = pin->GetOrAddReal("problem", "beam_position_x", 0.0);
    pm1->beam_position[1] = pin->GetOrAddReal("problem", "beam_position_y", 0.0);
    pm1->beam_position[2] = pin->GetOrAddReal("problem", "beam_position_z", 0.0);
    pm1->kerr_mask_radius = pin->GetOrAddReal("problem", "kerr_mask_radius", 2.0);
    pm1->SetupKerrSchildMask(pm1->storage.u);
    pz4c->SetKerrSchild(pz4c->storage.adm, pz4c->storage.u);
    pz4c->ADMToZ4c(pz4c->storage.adm, pz4c->storage.u);
    pm1->SetupKerrBeamTest(pm1->storage.u);
  }
}




// refinement condition: check the maximum pressure gradient
int RefinementCondition(MeshBlock *pmb) {
  //TODO
  return 0;
}

void BeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamInnerX1(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}

void BeamOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamOuterX1(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}

void BeamInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamInnerX2(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}
void BeamOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamOuterX2(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}

void BeamInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamInnerX3(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}

void BeamOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u, FaceField &b,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  pmb->pm1->BeamOuterX3(pmb, pco, u, time, dt, il, iu, jl, ju, kl, ku, ngh);
}
