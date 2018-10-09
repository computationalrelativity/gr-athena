#ifndef VWAVE_HPP
#define VWAVE_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vwave.hpp
//  \brief definitions for the Vwave class

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../task_list/task_list.hpp"

class MeshBlock;
class ParameterInput;

//! \class Vwave
//  \brief Vwave data and functions

class Vwave {
  public:
    Vwave(MeshBlock *pmb, ParameterInput *pin);
    ~Vwave();

    // data
    MeshBlock * pmy_block; // pointer to MeshBlock containing this Vwave
    AthenaArray<Real> u;   // solution of the vectorial wave equation
    AthenaArray<Real> u1;  // solution of the vectorial wave equation at intermediate step
    AthenaArray<Real> rhs; // vectorial wave equation rhs


//    AthenaTensor<Real, SYM2, 3, 2> g; // Metric tensor
//    AthenaTensor<Real, SYM2, 3, 2> K; // Extrinsic curvature tensor
//    AthenaTensor<Real, SYM2, 3, 2> R; // Ricci tensor
    Real c;                // light speed

    // functions
    Real NewBlockTimeStep(void);       // compute new timestep on a MeshBlock
    void VwaveRHS(AthenaArray<Real> & u, int order);
    void AddVwaveRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
        IntegratorWeight w, AthenaArray<Real> &u_out);
  private:
    AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
};
#endif // VWAVE_HPP
