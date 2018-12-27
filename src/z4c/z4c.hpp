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

  MeshBlock * pmy_block;     // pointer to MeshBlock containing this Z4c

  // public data storage
  struct {
    AthenaArray<Real> u;     // solution of Z4c evolution system
    AthenaArray<Real> u1;    // solution at intermediate steps
    AthenaArray<Real> u2;    // solution at intermediate steps
    AthenaArray<Real> rhs;   // Z4c rhs
    AthenaArray<Real> adm;   // ADM variables
  } storage;

  // aliases for variables and RHS
  struct Z4c_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi;       // conf. factor
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Khat;      // trace extr. curvature
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Theta;     // Theta var in Z4c
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;     // lapse
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Gam_u;     // Gamma functions (BSSN)
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;    // shift
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;      // conf. 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> A_dd;      // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;

  // aliases for the ADM variables
  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Psi4;      // conformal factor
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Mom_d;     // momentum constraint
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;      // 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;      // curvature
  };
  ADM_vars adm;

  // user settings and options
  struct {
    Real chi_psi_power; // chi = psi^N, N = chi_psi_power
    Real chi_div_floor; // Puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
    Real z4c_kappa_damp1, z4c_kappa_damp2; // Constrain damping parameters
  } opts;
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
  void AddZ4cRHS(const Real wght, AthenaArray<Real> &u_out);
  // compute Z4c variables from ADM variables
  void ADMToZ4c(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // compute ADM variables from Z4c variables
  void Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // enforce algebraic constraints on the solution
  void AlgConstr(AthenaArray<Real> & u);
  // compute ADM constraints
  void ADMConstraints(AthenaArray<Real> & u);

  // utility fnuctions
  //
  // set Z4c aliases given a state
  void SetZ4cAliases(AthenaArray<Real> & u, Z4c_vars & z4c);
  // set ADM aliases given u_adm
  void SetADMAliases(AthenaArray<Real> & u, ADM_vars & adm)
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
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep

  // auxiliary tensors
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> detg;        // det(g)
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_uu;        // inverse of conf. metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> R_dd;        // Ricci tensor
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Kt_dd;       // extrinsic curvature conformally rescaled

  // auxiliary derivatives
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dalpha_d;    // lapse 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dchi_d;      // chi 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 1> dKhat_d;     // Khat 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dTheta_d;    // Theta 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddalpha_dd;  // lapse 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dbeta_du;    // shift 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddchi_dd;    // chi 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dGam_du;     // Gamma 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dg_ddd;      // metric 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dA_ddd;      // A 1st drvts
  AthenaTensor<Real, TensorSymm::ISYM2, NDIM, 3> ddbeta_ddu; // shift 2nd drvts
  AthenaTensor<Real, TensorSymm::SYM22, NDIM, 4> ddg_dddd;   // metric 2nd drvts

  // auxialiry Lie derivatives along the shift vector
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Lchi;        // Lie derivative of chi
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> LKhat;       // Lie derivative of Khat
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> LTheta;      // Lie derivative of Theta
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Lalpha;      // Lie derivative of the lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> LGam_u;      // Lie derivative of Gamma
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Lbeta_u;     // Lie derivative of the shift
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Lg_dd;       // Lie derivative of conf. 3-metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> LA_dd;       // Lie derivative of A

private:
  // compute homogeneous derivative of a grid function
  void diffh(int dir, Real idx,
             AthenaArray<Real> & fun, AthenaArray<Real> & diff);
  // compute mixed derivative of a grid function
  void diffm(int dir1, int dir2, Real idx1, Real idx2,
             AthenaArray<Real> & fun, AthenaArray<Real> & diff);
  // compute upwind first derivative of a grid function
  void advect(int dir, Real idx,
              AthenaArray<Real> & vec,
              AthenaArray<Real> & fun, AthenaArray<Real> & diff);
};

#endif // Z4c_HPP
