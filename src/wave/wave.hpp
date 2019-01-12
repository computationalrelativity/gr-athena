#ifndef WAVE_HPP
#define WAVE_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave.hpp
//  \brief definitions for the Wave class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../finite_differencing.hpp"
#include "../task_list/wave_task_list.hpp"

class MeshBlock;
class ParameterInput;

//! \class Wave
//  \brief Wave data and functions

class Wave {
public:
  Wave(MeshBlock *pmb, ParameterInput *pin);
  ~Wave();

  // data
  MeshBlock * pmy_block;        // ptr to MeshBlock containing this Wave
  AthenaArray<Real> u;          // solution of the wave equation
  AthenaArray<Real> u1, u2;     // auxiliary arrays at intermediate steps
  AthenaArray<Real> rhs;        // wave equation rhs
  AthenaArray<Real> exact;      // exact solution of of the wave equation
  AthenaArray<Real> error;      // error with respect to the exact solution

  Real c;                       // light speed

  // functions
  Real NewBlockTimeStep(void);  // compute new timestep on a MeshBlock
  void WaveRHS(AthenaArray<Real> & u);
  void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
    AthenaArray<Real> &u_in2, const Real wght[3]);
  void AddWaveRHS(const Real wght, AthenaArray<Real> &u_out);
  void ComputeExactSol();
  Real WaveProfile(Real x, Real y, Real z);
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
private:
  struct {
    typedef FDCenteredStencil<2, NFDCEN> stencil;
    int stride[3];
    Real idx[3];
    Real Dxx(int dir, Real & u) {
      Real * pu = &u;
      Real out = 0.0;
      for(int n = 0; n < stencil::width; ++n) {
        out += stencil::coeff[n] * pu[(n - stencil::offset)*stride[dir]];
      }
      return out*SQR(idx[dir]);
    }
  } FD;
};
#endif // WAVE_HPP
