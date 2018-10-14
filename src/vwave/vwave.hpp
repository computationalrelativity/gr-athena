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

// Metric indexes defined in athena.hpp : I00, ..., I33, NMETRIC
#define gab_IDX (0) // Metric components are the first vars ...
#define Kab_IDX (NMETRIC) // ... then curvature components
#define NVARS (2*NMETRIC) 
#define NDIM (4) 

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
    
  Real c;                            // light speed
  
  // functions
  Real NewBlockTimeStep(void);       // compute new timestep on a MeshBlock
  void VwaveRHS(AthenaArray<Real> & u, int order);
  void AddVwaveRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
			 IntegratorWeight w, AthenaArray<Real> &u_out);

  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
			Real const gyy, Real const gyz, Real const gzz);
  void SpatialInv(Real const det,
			 Real const gxx, Real const gxy, Real const gxz,
			 Real const gyy, Real const gyz, Real const gzz,
			 Real * uxx, Real * uxy, Real * uxz,
			 Real * uyy, Real * uyz, Real * uzz);

private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
  
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g; // Metric tensor
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> K; // Curvature
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_g; 
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_K; 

  // Auxiliary 1d vars
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> eta;   // flat Metric tensor     //TODO: check def
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> ieta;  // inverse Metric tensor 
  AthenaTensor<Real, TensorSymm::SYM22, 3, 4> ddg;   // metric drvts           //FIXME: symmetries
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> R;     // Ricci 
};
#endif // VWAVE_HPP
