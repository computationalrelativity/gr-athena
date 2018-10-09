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
#include "../task_list/task_list.hpp"

class MeshBlock;
class ParameterInput;

//! \class Wave
//  \brief Wave data and functions

class Wave {
  public:
    Wave(MeshBlock *pmb, ParameterInput *pin);
    ~Wave();

    // data
    MeshBlock * pmy_block;             // ptr to MeshBlock containing this Wave
    AthenaArray<Real> u;               // solution of the wave equation
    AthenaArray<Real> u1;              // solution of the wave equation at intermediate step
    AthenaArray<Real> rhs;             // wave equation rhs
    AthenaArray<Real> exact;           // exact solution of of the wave equation

    Real c;                            // light speed

    // functions
    Real NewBlockTimeStep(void);       // compute new timestep on a MeshBlock
    void WaveRHS(AthenaArray<Real> & u, int order);
    void AddWaveRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
        IntegratorWeight step_wghts, AthenaArray<Real> &u_out);
    void ComputeExactSol();

  private:
    AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
};
#endif // WAVE_HPP
