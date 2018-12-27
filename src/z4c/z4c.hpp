#ifndef Z4c_HPP
#define Z4c_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c.hpp
//  \brief definitions for the Z4c class
//
// Convention: tensor names are followed by tensor type suffixes:
//    _u --> contravariant component
//    _d --> covariant component
// For example g_dd is a tensor, or tensor-like object, with two covariant indices.
//
// Indices used to access tensor components are prefixed with l or u for
// "lower" and "upper" index. For example Gamma_udd(ua, lb, lc) is \Gamma^a_b_c

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

//! \class Z4c
//  \brief Z4c data and functions
class Z4c {

public:
  // Indexes of evolved variables
  enum {
    I_Z4c_chi = 0,
    I_Z4c_g = I_Z4c_gxx = 1, I_Z4c_gxy = 2, I_Z4c_gxz = 3, I_Z4c_gyy = 4, I_Z4c_gyz = 5, I_Z4c_gzz = 6,
    I_Z4c_Khat = 7,
    I_Z4c_A = I_Z4c_Axx = 8, I_Z4c_Axy = 9, I_Z4c_Axz = 10, I_Z4c_Ayy = 11, I_Z4c_Ayz = 12, I_Z4c_Azz = 13,
    I_Z4c_Gam = I_Z4c_Gamx = 14, I_Z4c_Gamy = 15, I_Z4c_Gamz = 16,
    I_Z4c_Theta = 17,
    I_Z4c_alpha = 18,
    I_Z4c_beta = I_Z4c_betax = 19, I_Z4c_betay = 20, I_Z4c_betaz = 21,
    N_Z4c = 22
  }
  // Indexes of ADM variables + auxiliary fields
  enum {
    I_ADM_g = I_ADM_gxx = 0, I_ADM_gxy = 1, I_ADM_gxz = 2, I_ADM_gyy = 3, I_ADM_gyz = 4, I_ADM_gzz = 5,
    I_ADM_K = I_ADM_Kxx = 6, I_ADM_Kxy = 7, I_ADM_Kxz = 8, I_ADM_Kyy = 9, I_ADM_Kyz = 10, I_ADM_Kzz = 11,
    I_ADM_Psi4 = 12,
    I_ADM_Ham = 13,
    I_ADM_Mom = I_ADM_Momx = 14, I_ADM_Momz = 15, I_ADM_Momz = 16,
    N_ADM = 17
  }

public:
  Z4c(MeshBlock *pmb, ParameterInput *pin);
  ~Z4c();

  // data
  MeshBlock * pmy_block;     // pointer to MeshBlock containing this Z4c
  AthenaArray<Real> u;       // solution of Z4c evolution system
  AthenaArray<Real> u1, u2;  // solution at intermediate steps
  AthenaArray<Real> rhs;     // Z4c rhs
  AthenaArray<Real> adm;     // ADM variables

  // ptrs for tensors vars
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> chi;         // Conf. factor
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g_dd;        // Conf. 3-Metric
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> A_dd;        // Conf. Traceless Extr. Curvature
  AthenaTensor<Real, TensorSymm::SYM2, 3, 0> Khat;        // Trace Extr. Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> Theta;       // Theta var in Z4c
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> Gam_u;       // Gamma functions (BSSN)
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> alpha;       // Lapse
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> beta_u;      // Shift

  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_chi;     // Conf. factor
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_g_dd;    // Conf. 3-Metric
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> rhs_A_dd;    // Conf. Traceless Extr. Curvature
  AthenaTensor<Real, TensorSymm::SYM2, 3, 0> rhs_Khat;    // Trace Extr. Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_Theta;   // Theta var in Z4c
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> rhs_Gam_u;   // Gamma functions (BSSN)
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> rhs_alpha;   // Lapse
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> rhs_beta_u;  // Shift

  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> ADM_g_dd;    // 3-Metric
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> ADM_K_dd;    // Curvature
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> Psi4;        // Conformal factor
  AthenaTensor<Real, TensorSymm::NONE, 3, 0> H;           // Hamiltonian constraint
  AthenaTensor<Real, TensorSymm::NONE, 3, 1> Mom_d;       // Momentum constraint

  Real c;             // Light speed
  Real chi_psi_power; // chi = psi^N, N = chi_psi_power
  Real chi_div_floor; // Puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
  Real z4c_kappa_damp1, z4c_kappa_damp2; // Constrain damping parameters

public:
  // scheduled functions
  //
  // compute new timestep on a MeshBlock
  Real NewBlockTimeStep(void);
  // compute the RHS given the Z4c variables
  void Z4cRHS(AthenaArray<Real> & u);
  // compute linear combination of states
  void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                   AthenaArray<Real> &u_in2, const Real wght[3]);
  // add RHS to state
  void AddWaveRHS(const Real wght, AthenaArray<Real> &u_out);
  // enforce algebraic constraints on the solution
  void AlgConstr(AthenaArray<Real> & u);
  // compute Z4c variables from ADM variables
  void ADMToZ4c(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // compute ADM variables from Z4c variables
  void Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // compute ADM constraints
  void ADMConstraints(AthenaArray<Real> & u);

  // utility fnuctions
  //
  // set Z4c aliases given a state
  void SetZ4cAliases(AthenaArray<Real> & u);
  // compute spatial determinant of a 3x3  matrix
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
      Real const gyy, Real const gyz, Real const gzz);
  // compute inverse of a 3x3 matrix
  void SpatialInv(Real const det,
      Real const gxx, Real const gxy, Real const gxz,
      Real const gyy, Real const gyz, Real const gzz,
      Real * uxx, Real * uxy, Real * uxz,
      Real * uyy, Real * uyz, Real * uzz);
  // compute trace of a rank 2 covariant spatial tensor
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

private:
  // compute homogeneous derivative of a grid function
  void diffh(int dir, Real idx,
             AthenaArray<Real> & fun, AthenaArray<Real> & diff);
  // compute mixed derivative of a grid function
  void diffm(int dir1, int dir2, Real idx1, Real idx2,
             AthenaArray<Real> & fun, AthenaArray<Real> & diff);
  // compute upwind first derivative of a grid function
  void diffu(int dir, Real idx,
             AthenaArray<Real> & vec,
             AthenaArray<Real> & fun, AthenaArray<Real> & diff);


};
#endif // Z4c_HPP
