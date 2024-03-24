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

// C++ headers
// #include <string>

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/finite_differencing.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/interp_intergrid.hpp" //SB FIXME imported from matter_tracker_extrema

#include "../bvals/cc/bvals_cc.hpp"
#include "../bvals/cx/bvals_cx.hpp"
#include "../bvals/vc/bvals_vc.hpp"

#include "z4c_macro.hpp"

#ifdef TWO_PUNCTURES
// twopuncturesc: Stand-alone library ripped from Cactus
#include "TwoPunctures.h"
#endif

#ifdef DBG_SYMMETRIZE_FD
#include "../utils/floating_point.hpp"
#endif // DBG_SYMMETRIZE_FD

class MeshBlock;
class ParameterInput;
class Z4c_AMR;
class AHF;

//! \class Z4c
//  \brief Z4c data and functions
class Z4c {
friend class Z4c_AMR;
friend class AHF;
public:
  // Indexes of evolved variables
  enum {
    I_Z4c_chi,
    I_Z4c_gxx, I_Z4c_gxy, I_Z4c_gxz, I_Z4c_gyy, I_Z4c_gyz, I_Z4c_gzz,
    I_Z4c_Khat,
    I_Z4c_Axx, I_Z4c_Axy, I_Z4c_Axz, I_Z4c_Ayy, I_Z4c_Ayz, I_Z4c_Azz,
    I_Z4c_Gamx, I_Z4c_Gamy, I_Z4c_Gamz,
    I_Z4c_Theta,
    I_Z4c_alpha,
    I_Z4c_betax, I_Z4c_betay, I_Z4c_betaz,
    N_Z4c
  };
  // Names of Z4c variables
  static char const * const Z4c_names[N_Z4c];
  // Indexes of ADM variables
  enum {
    I_ADM_gxx, I_ADM_gxy, I_ADM_gxz, I_ADM_gyy, I_ADM_gyz, I_ADM_gzz,
    I_ADM_Kxx, I_ADM_Kxy, I_ADM_Kxz, I_ADM_Kyy, I_ADM_Kyz, I_ADM_Kzz,
    I_ADM_psi4,
    N_ADM
  };
  // Names of ADM variables
  static char const * const ADM_names[N_ADM];
  // Indexes of Constraint variables
  enum {
    I_CON_C,
    I_CON_H,
    I_CON_M,
    I_CON_Z,
    I_CON_Mx, I_CON_My, I_CON_Mz,
    N_CON,
  };
  // Names of costraint variables
  static char const * const Constraint_names[N_CON];
  // Indexes of matter fields
  enum {
    I_MAT_rho,
    I_MAT_Sx, I_MAT_Sy, I_MAT_Sz,
    I_MAT_Sxx, I_MAT_Sxy, I_MAT_Sxz, I_MAT_Syy, I_MAT_Syz, I_MAT_S_zz,
    N_MAT
  };
  // Names of matter variables
  static char const * const Matter_names[N_MAT];
  // Indexes of Weyl scalars
  enum {
    I_WEY_rpsi4, I_WEY_ipsi4,
    N_WEY
  };
  // Names of Weyl scalars
  static char const * const Weyl_names[N_WEY];

public:
  Z4c(MeshBlock *pmb, ParameterInput *pin);
  ~Z4c();

  Z4c *pz4c;                 // for macro propagation
  Mesh *pmy_mesh;            // pointer to Mesh containing MeshBlock
  MeshBlock *pmy_block;      // pointer to MeshBlock containing this Z4c
  Coordinates *pmy_coord;    // coordinates of current block
  Z4c_AMR *pz4c_amr;         // pointer to Z4c_AMR for the refinement condition

  MB_info mbi;

  // public data storage
  struct {
    AthenaArray<Real> u;     // solution of Z4c evolution system
    AthenaArray<Real> u1;    // solution at intermediate steps
    AthenaArray<Real> u2;    // solution at intermediate steps
    AthenaArray<Real> rhs;   // Z4c rhs
    AthenaArray<Real> adm;   // ADM variables
    AthenaArray<Real> con;   // constraints
    AthenaArray<Real> mat;   // matter variables
    AthenaArray<Real> weyl;  // weyl scalars
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
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> psi4;      // conformal factor
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_dd;      // 3-metric
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;      // curvature
  };
  ADM_vars adm;

  // aliases for the constraints
  struct Constraint_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> C;         // Z constraint monitor
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> H;         // hamiltonian constraint
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> M;         // norm squared of M_d
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Z;         // Z constraint violation
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  struct Matter_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rho;       // matter energy density
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> S_d;       // matter momentum density
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> S_dd;      // matter stress tensor
  };
  Matter_vars mat;

  // aliases for the Weyl scalars
  struct Weyl_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rpsi4;       // Real part of Psi_4
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> ipsi4;       // Imaginary part of Psi_4
  };
  Weyl_vars weyl;

  // BD: this should be refactored
  // user settings and options
  struct {
    Real chi_psi_power;   // chi = psi^N, N = chi_psi_power
    Real chi_div_floor;   // puncture's floor value for chi, use max(chi, chi_div_floor) in non-differentiated chi
    Real diss;            // amount of numerical dissipation
    Real eps_floor;       // a small number O(10^-12)
    // Constraint damping parameters
    Real damp_kappa1;
    Real damp_kappa2;
    // Gauge conditions for the lapse
    Real lapse_oplog;
    Real lapse_harmonicf;
    Real lapse_harmonic;
    Real lapse_advect;
    Real lapse_K;
    // Gauge condition for the shift
    Real shift_Gamma;
    Real shift_alpha2Gamma;
    Real shift_H;
    Real shift_advect;
    Real shift_eta;
    // Spatially dependent shift damping
#if defined(Z4C_ETA_CONF)
    Real shift_eta_a;
    Real shift_eta_b;
    Real shift_eta_R_0;
#elif defined(Z4C_ETA_TRACK_TP)
    Real shift_eta_w;
    Real shift_eta_delta;
    Real shift_eta_P;
    Real shift_eta_TP_ix;
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

    // Matter parameters
    int cowling; // if 1 then cowling approximation used, rhs of z4c equations -> 0
    int rhstheta0; // if 1 then rhs of Theta equation -> 0
    int fixedgauge; // if 1 then gauge is fixed, rhs of alpha, beta^i equations -> 0
    int fix_admsource; // if 1 then gauge is fixed, rhs of alpha, beta^i equations -> 0
    int Tmunuinterp; // interpolate stress energy
    int epsinterp; // interpolate stress energy
    int bssn; // reduce to bssn

    // AwA parameters
    Real AwA_amplitude; // amplitude parameter
    Real AwA_d_x; // d_x (width) parameter
    Real AwA_d_y; // d_y (width) parameter
    Real AwA_Gaussian_w; // 1d Gaussian parameter
    Real AwA_polarised_Gowdy_t0; // seed time for pG test

    // Sphere-zone refinement
    int sphere_zone_number;
    AthenaArray<int> sphere_zone_levels;
    AthenaArray<Real> sphere_zone_radii;
    AthenaArray<int> sphere_zone_puncture;
    AthenaArray<Real> sphere_zone_center1;
    AthenaArray<Real> sphere_zone_center2;
    AthenaArray<Real> sphere_zone_center3;

    // for twopuncturesc
#ifdef TWO_PUNCTURES
    bool impose_bitant_id;
#endif

    // Compute constraints up to a maximum radius
    Real r_max_con;
  } opt;

  // boundary and grid data (associated to state-vector)
  FCN_CC_CX_VC(
    CellCenteredBoundaryVariable   ubvar,
    CellCenteredXBoundaryVariable  ubvar,
    VertexCenteredBoundaryVariable ubvar
  );

  // auxiliary data (split task-list)
  FCN_CC_CX_VC(
    CellCenteredBoundaryVariable   abvar,
    CellCenteredXBoundaryVariable  abvar,
    VertexCenteredBoundaryVariable abvar
  );

#if defined(Z4C_CX_ENABLED)
  CellCenteredXBoundaryVariable  rbvar;
#endif

  AthenaArray<Real> empty_flux[3];

  // storage for SMR/AMR
  // BD: this should perhaps be combined with the above stuct.
  AthenaArray<Real> coarse_u_;
  AthenaArray<Real> coarse_a_;  // for auxiliary data (split task-list)

  int refinement_idx{-1};

  // metric derivatives used by AHF
  // it is allocated there as needed
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> aux_g_ddd;

public:
  // scheduled functions
  //
  // compute new timestep on a MeshBlock
  Real NewBlockTimeStep(void);
  // compute the RHS given the Z4c and matter variables
  void Z4cRHS(AthenaArray<Real> & u, AthenaArray<Real> & mat, AthenaArray<Real> & rhs);
  void Z4cRHS_(AthenaArray<Real> & u, AthenaArray<Real> & mat, AthenaArray<Real> & rhs);

  // compute the boundary RHS given the Z4c and matter variables
  void Z4cBoundaryRHS(AthenaArray<Real> & u, AthenaArray<Real> & mat, AthenaArray<Real> & rhs);
  // compute linear combination of states
  void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                   AthenaArray<Real> &u_in2, const Real wght[3]);
  // add RHS to state
  void AddZ4cRHS(AthenaArray<Real> & rhs, Real const wght, AthenaArray<Real> &u_out);
  // compute Z4c variables from ADM variables
  void ADMToZ4c(AthenaArray<Real> & u_adm, AthenaArray<Real> & u);
  // compute ADM variables from Z4c variables
  void Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm);

  // Conformal factor conversions
  // Floor applied: std::max(chi, opt.chi_div_floor)
  //
  // psi4 == g^(1/3)
  // g    == psi4^3
  // chi  == g^(1/12 * p)
  // g    == chi^(12/p)
  //
  // chi == psi4^(p/4), psi4 == chi^(4/p)
  inline Real psi4Regularized(const Real & psi4_bare)
  {
    // A floor on chi is a ceil on psi4
    const Real T_chi = opt.chi_div_floor;
    const Real p = opt.chi_psi_power;

    const Real T_psi4 = std::pow(T_chi, 4. / p);
    const Real psi4_guarded = (std::isfinite(psi4_bare)) ?
                               std::min(T_psi4, psi4_bare) : T_psi4;
    return psi4_guarded;
  }

  inline Real chiRegularized(const Real & chi_bare)
  {
    const Real T_chi = opt.chi_div_floor;
    return std::max(chi_bare, opt.chi_div_floor);
  }


  // enforce algebraic constraints on the solution
  void AlgConstr(AthenaArray<Real> & u);
  // compute ADM constraints
  void ADMConstraints(AthenaArray<Real> & u_con, AthenaArray<Real> & u_adm,
                      AthenaArray<Real> & u_mat, AthenaArray<Real> & u_z4c);
  // calculate weyl scalars
  void Z4cWeyl(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat,
               AthenaArray<Real> & u_weyl);
  // Update matter variables from hydro
  void GetMatter(AthenaArray<Real> & u_mat,
                 AthenaArray<Real> & u_adm,
                 AthenaArray<Real> & w,
#if USETM
                 AthenaArray<Real> & r,
#endif
                 AthenaArray<Real> & bb_cc);

  // utility functions
  //
  // set ADM aliases given u_adm
  static void SetADMAliases(AthenaArray<Real> & u_adm, ADM_vars & adm);
  // set constraint aliases for a given u_con
  static void SetConstraintAliases(AthenaArray<Real> & u_con, Constraint_vars & con);
  // set matter aliases given a state
  static void SetMatterAliases(AthenaArray<Real> & u_mat, Matter_vars & mat);
  // set Z4c aliases given a state
  static void SetZ4cAliases(AthenaArray<Real> & u, Z4c_vars & z4c);
  // set weyl aliases
  static void SetWeylAliases(AthenaArray<Real> & u_weyl, Weyl_vars & weyl);

  // additional global functions

  // setup a Minkowski spacetime
  void ADMMinkowski(AthenaArray<Real> & u_adm);
  // set the gauge condition to geodesic slicing
  void GaugeGeodesic(AthenaArray<Real> & u);
  // set the matter variables to zero
  void MatterVacuum(AthenaArray<Real> & u_adm);

  // initial data for the AwA tests
  void ADMRobustStability(AthenaArray<Real> & u_adm);
  void GaugeRobStab(AthenaArray<Real> & u);
  void ADMLinearWave1(AthenaArray<Real> & u_adm);
  void ADMLinearWave1Gaussian(AthenaArray<Real> & u_adm);
  void ADMLinearWave2(AthenaArray<Real> & u_adm);
  void ADMGaugeWave1(AthenaArray<Real> & u_adm);
  void ADMGaugeWave1_shifted(AthenaArray<Real> & u_adm);
  void ADMGaugeWave2(AthenaArray<Real> & u_adm);
  void GaugeGaugeWave1(AthenaArray<Real> & u);
  void GaugeGaugeWave1_shifted(AthenaArray<Real> & u);
  void GaugeGaugeWave2(AthenaArray<Real> & u);
  void GaugeSimpleGaugeWave(AthenaArray<Real> & u);

#ifdef GSL
  void ADMPolarisedGowdy(AthenaArray<Real> & u_adm);
  void GaugePolarisedGowdy(AthenaArray<Real> & u);
#endif

  // initial data for a single BH
  void ADMOnePuncture(ParameterInput *pin, AthenaArray<Real> & u_adm);
  void GaugePreCollapsedLapse(AthenaArray<Real> & u_adm, AthenaArray<Real> & u);

  // initial data for binary BHs
#ifdef TWO_PUNCTURES
  void ADMTwoPunctures(ParameterInput *pin, AthenaArray<Real> & u_adm, ini_data * data);
#endif

  // functions for debugging and monitoring
  bool is_finite_adm();
  bool is_finite_con();
  bool is_finite_mat();
  bool is_finite_z4c();

  void assert_is_finite_adm();
  void assert_is_finite_con();
  void assert_is_finite_mat();
  void assert_is_finite_z4c();

private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep

  // auxiliary tensors
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> r;           // radial coordinate
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> detg;        // det(g)
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi_guarded; // bounded version of chi
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> oopsi4;      // 1/psi4
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> A;           // trace of A
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> AA;          // trace of AA
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> R;           // Ricci scalar
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Ht;          // tilde H
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> K;           // trace of extrinsic curvature
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> KK;          // K^a_b K^b_a
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Ddalpha;     // Trace of Ddalpha_dd
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> S;           // Trace of S_ik
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> M_u;         // momentum constraint
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Gamma_u;     // Gamma computed from the metric
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> DA_u;        // Covariant derivative of A
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> s_u;         // x^i/r where r is the coord. radius
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_uu;        // inverse of conf. metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> A_uu;        // inverse of A
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> AA_dd;       // g^cd A_ac A_db
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> R_dd;        // Ricci tensor
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Rphi_dd;     // Ricci tensor, conformal contribution
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Kt_dd;       // conformal extrinsic curvature
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_ud;        // extrinsic curvature
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Ddalpha_dd;  // 2nd differential of the lapse
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Ddphi_dd;    // 2nd differential of phi
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> Gamma_ddd;   // Christoffel symbols of 1st kind
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> Gamma_udd;   // Christoffel symbols of 2nd kind
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> DK_ddd;      // differential of K
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> DK_udd;      // differential of K

  // Spatially dependent shift damping
#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> eta_damp;
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  // auxiliary derivatives
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dbeta;       // d_a beta^a
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dalpha_d;    // lapse 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> ddbeta_d;    // 2nd "divergence" of beta
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dchi_d;      // chi 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dphi_d;      // phi 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dK_d;        // K 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dKhat_d;     // Khat 1st drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dTheta_d;    // Theta 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddalpha_dd;  // lapse 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dbeta_du;    // shift 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> ddchi_dd;    // chi 2nd drvts
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dGam_du;     // Gamma 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dg_ddd;      // metric 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dK_ddd;      // K 1st drvts
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dA_ddd;      // A 1st drvts
  AthenaTensor<Real, TensorSymm::ISYM2, NDIM, 3> ddbeta_ddu; // shift 2nd drvts
  AthenaTensor<Real, TensorSymm::SYM22, NDIM, 4> ddg_dddd;   // metric 2nd drvts

  // auxiliary Lie derivatives along the shift vector
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Lchi;        // Lie derivative of chi
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> LKhat;       // Lie derivative of Khat
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> LTheta;      // Lie derivative of Theta
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Lalpha;      // Lie derivative of the lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> LGam_u;      // Lie derivative of Gamma
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> Lbeta_u;     // Lie derivative of the shift
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> Lg_dd;       // Lie derivative of conf. 3-metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> LA_dd;       // Lie derivative of A

  //WGC wext - TODO fix tensor index symmetries
  //auxiliary wave extraction tensors
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> uvec;        // radial vector in tetrad
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> vvec;        // theta vector in tetrad
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> wvec;        // phi vector in tetrad
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dotp1;       // dot product in Gram-Schmidt orthonormalisation
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> dotp2;       // second dot product in G-S orthonormalisation
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 4> Riem3_dddd;  // 3D Riemann tensor
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 4> Riemm4_dddd; // 4D Riemann tensor
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 3> Riemm4_ddd;  // 4D Riemann * n^a
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> Riemm4_dd;   // 4D Riemann *n^a*n^c

  // Aux vars handling cx/vc matter interpolation
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> w_rho;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> w_p;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> w_utilde_u;
#if USETM
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> w_r;
#endif

#if MAGNETIC_FIELDS_ENABLED
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> bb;
#endif

private:
  void Z4cSommerfeld_(AthenaArray<Real> & u, AthenaArray<Real> & rhs,
      int const is, int const ie, int const js, int const je, int const ks, int const ke);

private:
  FiniteDifference::Uniform *fd;

};

#endif // Z4c_HPP
