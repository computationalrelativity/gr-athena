#ifndef Z4c_HPP
#define Z4c_HPP
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

// Indexes for variables in AthenaArray
#define NDIM (3) // Manifold dimension
#define NCa  (3) // No components in tensors with 1 index
#define NCab (6) // No components in tensors with 2 indexes

// Indexes of evolved variables
enum{
  chi_IDX, // 0
  gxx_IDX, gxy_IDX, gxz_IDX, gyy_IDX, gyz_IDX, gzz_IDX, // 1...6
  Khat_IDX, // 7
  Axx_IDX, Axy_IDX, Axz_IDX, Ayy_IDX, Ayz_IDX, Azz_IDX, // 8...13
  Gamx_IDX, Gamx_IDX, Gamz_IDX, // 14...16       
  Theta_IDX, // 17
  alpha_IDX, // 18
  betax_IDX, betay_IDX, betaz_IDX, // 19...21
  NVARS
}

// Indexes of ADM variables + auxiliary fields
enum{
  ADM_gxx_IDX, ADM_gxy_IDX, ADM_gxz_IDX, ADM_gyy_IDX, ADM_gyz_IDX, ADM_gzz_IDX, // 1...6
  ADM_Kxx_IDX, ADM_Kxy_IDX, ADM_Kxz_IDX, ADM_Kyy_IDX, ADM_Kyz_IDX, ADM_Kzz_IDX, // 7...12
  ADM_Psi4_IDX, // 13
  ADM_Ham_IDX, // 14
  ADM_Momx_IDX, ADM_Momy_IDX, ADM_Momz_IDX, // 15...17
  ADMVARS
}

//! \class Z4c
//  \brief Z4c data and functions

class Z4c {
public:
  Vwave(MeshBlock *pmb, ParameterInput *pin);
  ~Vwave();
  
  // data
  MeshBlock * pmy_block; // pointer to MeshBlock containing this Vwave
  AthenaArray<Real> u;   // solution of Z4c evolution system
  AthenaArray<Real> u1;  // solution at intermediate step
  AthenaArray<Real> rhs; // Z4c rhs
  AthenaArray<Real> adm; // ADM variables
  
  // ptrs for tensors vars 
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;   // Conf. factor
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g;     // Conf. 3-Metric 
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> A;     // Conf. Traceless Extr. Curvature
  AthenaTensor<Real, TensorSymm::SYM2, 3, 0> Khat;  // Trace Extr. Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> Theta; // Theta var in Z4c
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> Gam;   // Gamma functions (BSSN)
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha; // Lapse
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta;  // Shift

  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_chi;   // Conf. factor
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_g;     // Conf. 3-Metric 
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_A;     // Conf. Traceless Extr. Curvature
  AthenaTensor<Real, TensorSymm::SYM2, 3, 0> rhs_Khat;  // Trace Extr. Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_Theta; // Theta var in Z4c
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> rhs_Gam;   // Gamma functions (BSSN)
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_alpha; // Lapse
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> rhs_beta;  // Shift

  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> ADM_g; // 3-Metric 
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> ADM_K; // Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> Psi4;  // Conformal factor
  
  Real c;             // Light speed
  Real chi_psi_power; // chi = psi^N, N = chi_psi_power
  Real chi_div_floor; // Puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
  Real z4c_kappa_damp1, z4c_kappa_damp2; // Constrain damping parameters
  
  // functions
  Real NewBlockTimeStep(void);       // compute new timestep on a MeshBlock
  void AddVwaveRHSToVals(AthenaArray<Real> & u1, AthenaArray<Real> & u2,
			 IntegratorWeight w, AthenaArray<Real> &u_out);
  
  void Z4cRHS(AthenaArray<Real> & u, int order);
  void AlgConstr(AthenaArray<Real> & u);
  void ADMToZ4c(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  void Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  void ADMConstraints(AthenaArray<Real> & u);
  
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
			Real const gyy, Real const gyz, Real const gzz);
  void SpatialInv(Real const det,
		  Real const gxx, Real const gxy, Real const gxz,
		  Real const gyy, Real const gyz, Real const gzz,
		  Real * uxx, Real * uxy, Real * uxz,
		  Real * uyy, Real * uyz, Real * uzz);
  Real Trace(Real const detginv,
	     Real const gxx, Real const gxy, Real const gxz,
	     Real const gyy, Real const gyz, Real const gzz,
	     Real const Axx, Real const Axy, Real const Axz,
	     Real const Ayy, Real const Ayz, Real const Azz);

  void ADMFlat(AthenaArray<Real> & u_adm);
  void GaugeFlat(AthenaArray<Real> & u);
  
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
  
  // Auxiliary 1D vars  //TODO
  AthenaTensor<Real, TensorSymm::NONE, 3, 2> ginv;  // flat Metric tensor   
  AthenaTensor<Real, TensorSymm::NONE, 3, 2> detg;  // inverse Metric tensor
  AthenaTensor<Real, TensorSymm::NONE, 3, 2> epsg;  // inverse Metric tensor 

  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> dg;  // metric 1st drvts
  AthenaTensor<Real, TensorSymm::SYM22, 3, 4> ddg;  // metric 2nd drvts
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> R;     // Ricci 

};
#endif // Z4c_HPP
