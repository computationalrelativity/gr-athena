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

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../finite_differencing.hpp"
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
    I_Z4c_gxx = 1, I_Z4c_gxy = 2, I_Z4c_gxz = 3, I_Z4c_gyy = 4, I_Z4c_gyz = 5, I_Z4c_gzz = 6,
    I_Z4c_Khat = 7,
    I_Z4c_Axx = 8, I_Z4c_Axy = 9, I_Z4c_Axz = 10, I_Z4c_Ayy = 11, I_Z4c_Ayz = 12, I_Z4c_Azz = 13,
    I_Z4c_Gamx = 14, I_Z4c_Gamy = 15, I_Z4c_Gamz = 16,
    I_Z4c_Theta = 17,
    I_Z4c_alpha = 18,
    I_Z4c_betax = 19, I_Z4c_betay = 20, I_Z4c_betaz = 21,
    N_Z4c = 22
  };
  // Indexes of ADM variables + auxiliary fields
  enum {
    I_ADM_gxx = 0, I_ADM_gxy = 1, I_ADM_gxz = 2, I_ADM_gyy = 3, I_ADM_gyz = 4, I_ADM_gzz = 5,
    I_ADM_Kxx = 6, I_ADM_Kxy = 7, I_ADM_Kxz = 8, I_ADM_Kyy = 9, I_ADM_Kyz = 10, I_ADM_Kzz = 11,
    I_ADM_Psi4 = 12,
    I_ADM_Ham = 13,
    I_ADM_Momx = 14, I_ADM_Momy = 15, I_ADM_Momz = 16,
    I_ADM_rho = 17,
    I_ADM_Sx = 18, I_ADM_Sy = 19, I_ADM_Sz = 20,
    I_ADM_Sxx = 21, I_ADM_Sxy = 22, I_ADM_Sxz = 23, I_ADM_Syy = 24, I_ADM_Syz = 25, I_ADM_S_zz = 26,
    I_ADM_psi4 = 27,
    N_ADM = 28
  };
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
  Z4c_vars rhs;

  // aliases for the ADM variables
  struct ADM_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> psi4;      // conformal factor
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Mom_d;     // momentum constraint
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;      // 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;      // curvature
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rho;       // matter energy density
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> S_d;       // matter momentum density
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> S_dd;      // matter stress tensor
  };
  ADM_vars adm;

  // user settings and options
  struct {
    Real chi_psi_power; // chi = psi^N, N = chi_psi_power
    Real chi_div_floor; // Puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
    Real eps_floor;     // A small number O(10^-12)
    Real z4c_kappa_damp1, z4c_kappa_damp2; // Constrain damping parameters
  } opt;
public:
  // scheduled functions
  //
  // compute new timestep on a MeshBlock
  Real NewBlockTimeStep(void);
  // compute the RHS given the Z4c variables
  void Z4cRHS(AthenaArray<Real> & u, AthenaArray<Real> & rhs);
  // compute linear combination of states
  void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                   AthenaArray<Real> &u_in2, const Real wght[3]);
  // add RHS to state
  void AddZ4cRHS(AthenaArray<Real> & rhs, Real const wght, AthenaArray<Real> &u_out);
  // compute Z4c variables from ADM variables
  void ADMToZ4c(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // compute ADM variables from Z4c variables
  void Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);
  // enforce algebraic constraints on the solution
  void AlgConstr(AthenaArray<Real> & u);
  // compute ADM constraints
  void ADMConstraints(AthenaArray<Real> & u_adm);

  // utility fnuctions
  //
  // set ADM aliases given u_adm
  void SetADMAliases(AthenaArray<Real> & u_adm, ADM_vars & adm);
  // set Z4c aliases given a state
  void SetZ4cAliases(AthenaArray<Real> & u, Z4c_vars & z4c);
  // compute spatial determinant of a 3x3  matrix
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
      Real const gyy, Real const gyz, Real const gzz);
  Real SpatialDet(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & g,
                  int const k, int const j, int const i) {
    return SpatialDet(g(0,0,k,j,i), g(0,1,k,j,i), g(0,2,k,j,i),
                      g(1,1,k,j,i), g(1,2,k,j,i), g(2,2,k,j,i));
  }
  // compute inverse of a 3x3 matrix
  void SpatialInv(Real const detginv,
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

  // additional global functions
  //
  // setup a Minkowski spacetime
  void ADMMinkowski(AthenaArray<Real> & u_adm);
  // set the ADM matter variables to zero
  void ADMVacuum(AthenaArray<Real> & u_adm);
  // set the gauge condition to geodesic slicing
  void GaugeGeodesic(AthenaArray<Real> & u);
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep

  // auxiliary tensors
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> detg;        // det(g)
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> oopsi4;      // 1/psi4
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> A;           // trace of A
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> R;           // Ricci scalar
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> K;           // trace of extrinsic curvature
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> KK;          // K^a_b K^b_a
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Mom_u;       // momentum constraint
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Gamma_u;     // Gamma computed from the metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_uu;        // inverse of conf. metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> R_dd;        // Ricci tensor
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Kt_dd;       // conformal extrinsic curvature
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_ud;        // extrinsic curvature
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> Gamma_ddd;   // Christoffel symbols of 1st kind
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> Gamma_udd;   // Christoffel symbols of 2nd kind
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> DK_ddd;      // differential of K
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> DK_udd;      // differential of K

  // auxiliary derivatives
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dalpha_d;    // lapse 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dchi_d;      // chi 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dK_d;        // K 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dKhat_d;     // Khat 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dTheta_d;    // Theta 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddalpha_dd;  // lapse 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dbeta_du;    // shift 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddchi_dd;    // chi 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dGam_du;     // Gamma 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dg_ddd;      // metric 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dg_duu;      // inverse metric 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dK_ddd;      // K 1st drvts
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
  struct {
    typedef FDCenteredStencil<1, NGHOST> s1;
    typedef FDLeftBiasedStencil<1, NGHOST> sl;
    typedef FDRightBiasedStencil<1, NGHOST> sr;
    typedef FDCenteredStencil<2, NGHOST> s2;

    int stride[3];
    Real idx[3];

    Real Dx(int dir, Real & u) {
      Real * pu = &u;
      Real out = 0.0;
      for(int n = 0; n < s1::width; ++n) {
        out += s1::coeff[n] * pu[(n - s1::offset)*stride[dir]];
      }
      return out*SQR(idx[dir]);
    }
    Real Lx(int dir, Real & vx, Real & u) {
      Real * pu = &u;
      Real dl(0.);
      for(int n = 0; n < sl::width; ++n) {
        dl += sl::coeff[n] * pu[(n - sl::offset)*stride[dir]];
      }
      Real dr(0.);
      for(int n = 0; n < sr::width; ++n) {
        dr += sr::coeff[n] * pu[(n - sr::offset)*stride[dir]];
      }
      return ((vx > 0) ? (vx*dl) : (vx*dr))*idx[dir];
    }
    Real Dxx(int dir, Real & u) {
      Real * pu = &u;
      Real out = 0.0;
      for(int n = 0; n < s2::width; ++n) {
        out += s2::coeff[n] * pu[(n - s2::offset)*stride[dir]];
      }
      return out*SQR(idx[dir]);
    }
    Real Dxy(int dir1, int dir2, Real & u) {
      Real * pu = &u;
      Real out = 0.0;
      for(int n1 = 0; n1 < s1::width; ++n1)
      for(int n2 = 0; n2 < s1::width; ++n2) {
        out += s1::coeff[n1] * s1::coeff[n2] *
          pu[(n1 - s1::offset)*stride[dir1] + (n2 - s1::offset)*stride[dir2]];
      }
      return out*idx[dir1]*idx[dir2];
    }
  } FD;
};

#endif // Z4c_HPP
