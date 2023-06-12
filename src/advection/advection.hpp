#ifndef ADVECTION_HPP
#define ADVECTION_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file advection.hpp
//  \brief definitions for the Advection class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../finite_differencing.hpp"
#include "../bvals/cc/bvals_cc.hpp"

class MeshBlock;
class ParameterInput;

//! \class Advection
//  \brief Advection data and functions

class Advection {
public:
  Advection(MeshBlock *pmb, ParameterInput *pin);
  ~Advection();

  // data
  MeshBlock *pmy_block;         // ptr to MeshBlock containing this Advection
  AthenaArray<Real> u;          // solution of the advection equation
  AthenaArray<Real> u1, u2;     // auxiliary arrays at intermediate steps
  AthenaArray<Real> rhs;        // advection equation rhs

  AthenaArray<Real> exact;      // exact solution of of the advection equation
  AthenaArray<Real> error;      // error with respect to the exact solution

  // propagation velocity components
  Real cx1;
  Real cx2;
  Real cx3;

  // control whether radiative condition is applied for outflow  or
  // extrapolate_outflow BC;
  // 0: not applied
  // 1,2,3: applied in respective dimensions
  int use_Sommerfeld = 0;

  // boundary and grid data
  CellCenteredBoundaryVariable ubvar;
  AthenaArray<Real> empty_flux[3];

  // storage for SMR/AMR
  // TODO(KGF): remove trailing underscore or revert to private:
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};

  // functions
  Real NewBlockTimeStep(void);  // compute new timestep on a MeshBlock
  void AdvectionRHS(AthenaArray<Real> &u);
  void AdvectionBoundaryRHS(AthenaArray<Real> &u);
  void AddAdvectionRHS(const Real wght, AthenaArray<Real> &u_out);

private:
  AthenaArray<Real> dt1_,dt2_,dt3_;    // scratch arrays used in NewTimeStep

private:
  FiniteDifference::Uniform * pfd;

};
#endif // ADVECTION_HPP
