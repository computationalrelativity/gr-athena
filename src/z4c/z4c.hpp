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
#include "../athena_aliases.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/finite_differencing.hpp"

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

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

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
    I_ADM_alpha,
    I_ADM_betax, I_ADM_betay, I_ADM_betaz,
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
  // Indexes of auxiliary variables for 3D ADM metric drvts
  enum {
    I_AUX_dalpha_x, I_AUX_dalpha_y, I_AUX_dalpha_z,
    I_AUX_dbetax_x, I_AUX_dbetay_x, I_AUX_dbetaz_x,
    I_AUX_dbetax_y, I_AUX_dbetay_y, I_AUX_dbetaz_y,
    I_AUX_dbetax_z, I_AUX_dbetay_z, I_AUX_dbetaz_z,
    I_AUX_dgxx_x, I_AUX_dgxy_x, I_AUX_dgxz_x, I_AUX_dgyy_x, I_AUX_dgyz_x, I_AUX_dgzz_x,
    I_AUX_dgxx_y, I_AUX_dgxy_y, I_AUX_dgxz_y, I_AUX_dgyy_y, I_AUX_dgyz_y, I_AUX_dgzz_y,
    I_AUX_dgxx_z, I_AUX_dgxy_z, I_AUX_dgxz_z, I_AUX_dgyy_z, I_AUX_dgyz_z, I_AUX_dgzz_z,
    N_AUX
  };
  // Names of auxiliary variables
  static char const * const Aux_names[N_AUX];

  enum {
    I_AUX_EXTENDED_cc_sqrt_detgamma,
    N_AUX_EXTENDED
  };
  // Names of auxiliary variables
  static char const * const Aux_Extended_names[N_AUX_EXTENDED];

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
    AA u;     // solution of Z4c evolution system
    AA u1;    // solution at intermediate steps
    AA u2;    // solution at intermediate steps
    AA rhs;   // Z4c rhs
    AA adm;   // ADM variables
    AA con;   // constraints
    AA mat;   // matter variables
    AA weyl;  // weyl scalars
    AA aux;   // aux quantities such as derivatives
    AA aux_extended;  // further aux quantities (non-communicated)
  } storage;

  // aliases for variables and RHS
  struct Z4c_vars {
    AT_N_sca chi;       // conf. factor
    AT_N_sca Khat;      // trace extr. curvature
    AT_N_sca Theta;     // Theta var in Z4c
    AT_N_sca alpha;     // lapse
    AT_N_vec Gam_u;     // Gamma functions (BSSN)
    AT_N_vec beta_u;    // shift
    AT_N_sym g_dd;      // conf. 3-metric
    AT_N_sym A_dd;      // conf. traceless extr. curvature
  };
  Z4c_vars z4c;
  Z4c_vars rhs;

  // aliases for the ADM variables
  struct ADM_vars {
    AT_N_sca alpha;     // lapse
    AT_N_vec beta_u;    // shift
    AT_N_sca psi4;      // conformal factor
    AT_N_sym g_dd;      // 3-metric
    AT_N_sym K_dd;      // curvature
  };
  ADM_vars adm;

  // aliases for the constraints
  struct Constraint_vars {
    AT_N_sca C;         // Z constraint monitor
    AT_N_sca H;         // hamiltonian constraint
    AT_N_sca M;         // norm squared of M_d
    AT_N_sca Z;         // Z constraint violation
    AT_N_vec M_d;       // momentum constraint
  };
  Constraint_vars con;

  // aliases for the matter variables
  struct Matter_vars {
    AT_N_sca rho;       // matter energy density
    AT_N_vec S_d;       // matter momentum density
    AT_N_sym S_dd;      // matter stress tensor
  };
  Matter_vars mat;

  // aliases for the Weyl scalars
  struct Weyl_vars {
    AT_N_sca rpsi4;       // Real part of Psi_4
    AT_N_sca ipsi4;       // Imaginary part of Psi_4
  };
  Weyl_vars weyl;

  // aliases for auxiliary variables for metric derivatives
  struct Aux_vars {
    AT_N_vec dalpha_d; // lapse 1st derivatives
    AT_N_T2 dbeta_du;  // shift 1st derivatives
    AT_N_VS2 dg_ddd;   // ADM 3-metric 1st derivatives
  };
  Aux_vars aux;

  // metric derivatives used by AHF
  // it is allocated there as needed
  // this is alternative to the aux. storage,
  // used when 'store_metric_drvts' if off
  AT_N_VS2 aux_g_ddd;

  // aliases for auxiliary variables for metric derivatives
  struct Aux_extended_vars {
    AT_N_sca cc_sqrt_detgamma;  // adm gamma on cc
  };
  Aux_extended_vars aux_extended;

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
    AA sphere_zone_radii;
    AthenaArray<int> sphere_zone_puncture;
    AA sphere_zone_center1;
    AA sphere_zone_center2;
    AA sphere_zone_center3;

    // for twopuncturesc
#ifdef TWO_PUNCTURES
    bool impose_bitant_id;
#endif

    // Compute constraints up to a maximum radius
    Real r_max_con;

    // Compute & store 3D ADM metric derivatives for post-step analyses
    bool store_metric_drvts;
    // control whether ^ is communicated
    bool communicate_aux_adm;

    // Compute aux_extended variables?
    bool extended_aux_adm;

    // For debug
    bool use_tp_trackers_extrema;
  } opt;

  AA empty_flux[3];

  // storage for SMR/AMR
  AA coarse_u_;
  AA coarse_a_;  // for auxiliary data (split task-list)

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


  AA coarse_adm_;  // for auxiliary data (split task-list)

  // auxiliary data (split task-list)
  FCN_CC_CX_VC(
    CellCenteredBoundaryVariable   * adm_abvar,
    CellCenteredXBoundaryVariable  * adm_abvar,
    VertexCenteredBoundaryVariable * adm_abvar
  );

  int refinement_idx{-1};

  // BT style integrators -----------------------------------------------------
  std::vector<AA> bt_k;

public:
  // scheduled functions
  //
  // compute new timestep on a MeshBlock
  Real NewBlockTimeStep(void);
  // compute the RHS given the Z4c and matter variables
  void Z4cRHS(AA & u, AA & mat, AA & rhs);
  void Z4cRHS_(AA & u, AA & mat, AA & rhs);

  // compute the boundary RHS given the Z4c and matter variables
  void Z4cBoundaryRHS(AA & u, AA & mat, AA & rhs);
  // compute linear combination of states
  void WeightedAve(AA &u_out, AA &u_in1,
                   AA &u_in2, const Real wght[3]);
  // add RHS to state
  void AddZ4cRHS(AA & rhs, Real const wght, AA &u_out);
  // compute Z4c variables from ADM variables
  void ADMToZ4c(AA & u_adm, AA & u);
  // compute ADM variables from Z4c variables
  void Z4cToADM(AA & u, AA & u_adm);
  // compute and store ADM metric derivatives in 3D auxiliary storage
  void ADMDerivatives(AthenaArray<Real> &u, AthenaArray<Real> &u_adm,
                      AthenaArray<Real> &u_aux);

  void PrepareAuxExtended(AA &u_aux_extended, AA &u_adm);

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
  void AlgConstr(AA & u);

  // compute ADM constraints
  void ADMConstraints(AA & u_con, AA & u_adm, AA & u_mat, AA & u_z4c);

  // calculate weyl scalars
  void Z4cWeyl(AA & u_adm, AA & u_mat, AA & u_weyl);

  // Update matter variables from hydro
  void GetMatter(AA & u_mat,
                 AA & u_adm,
                 AA & w,
                 AA & r,
                 AA & bb_cc);

  // utility functions
  //
  // set ADM aliases given u_adm
  static void SetADMAliases(AA & u_adm, ADM_vars & adm);
  // set constraint aliases for a given u_con
  static void SetConstraintAliases(AA & u_con, Constraint_vars & con);
  // set matter aliases given a state
  static void SetMatterAliases(AA & u_mat, Matter_vars & mat);
  // set Z4c aliases given a state
  static void SetZ4cAliases(AA & u, Z4c_vars & z4c);
  // set weyl aliases
  static void SetWeylAliases(AA & u_weyl, Weyl_vars & weyl);
  // set auxiliary variable aliases
  static void SetAuxAliases(AA & u, Aux_vars & aux);
  // set auxiliary (extended) variable aliases
  static void SetAuxExtendedAliases(AA & u_adm,
                                    Aux_extended_vars & aux_extended);

  // additional global functions

  // setup a Minkowski spacetime
  void ADMMinkowski(AA & u_adm);
  // set the gauge condition to geodesic slicing
  void GaugeGeodesic(AA & u);
  // set the matter variables to zero
  void MatterVacuum(AA & u_adm);

  // initial data for the AwA tests
  void ADMRobustStability(AA & u_adm);
  void GaugeRobStab(AA & u);
  void ADMLinearWave1(AA & u_adm);
  void ADMLinearWave1Gaussian(AA & u_adm);
  void ADMLinearWave2(AA & u_adm);
  void ADMGaugeWave1(AA & u_adm);
  void ADMGaugeWave1_shifted(AA & u_adm);
  void ADMGaugeWave2(AA & u_adm);
  void GaugeGaugeWave1(AA & u);
  void GaugeGaugeWave1_shifted(AA & u);
  void GaugeGaugeWave2(AA & u);
  void GaugeSimpleGaugeWave(AA & u);

#ifdef GSL
  void ADMPolarisedGowdy(AA & u_adm);
  void GaugePolarisedGowdy(AA & u);
#endif

  void Z4cGaugeToADM(AA & u_adm, AA & u);

  // initial data for a single BH
  void ADMOnePuncture(ParameterInput *pin, AA & u_adm);
  void GaugePreCollapsedLapse(AA & u_adm, AA & u);

  // initial data for binary BHs
#ifdef TWO_PUNCTURES
  void ADMTwoPunctures(ParameterInput *pin, AA & u_adm, ini_data * data);
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
  AA dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep

  // auxiliary tensors
  AT_N_sca r;           // radial coordinate
  AT_N_sca detg;        // det(g)
  AT_N_sca chi_guarded; // bounded version of chi
  AT_N_sca oopsi4;      // 1/psi4
  AT_N_sca A;           // trace of A
  AT_N_sca trAA;        // trace of AA
  AT_N_sca R;           // Ricci scalar
  AT_N_sca Ht;          // tilde H
  AT_N_sca K;           // trace of extrinsic curvature
  AT_N_sca KK;          // K^a_b K^b_a
  AT_N_sca Ddalpha;     // Trace of Ddalpha_dd
  AT_N_sca S;           // Trace of S_ik

  AT_N_vec M_u;         // momentum constraint
  AT_N_vec Gamma_u;     // Gamma computed from the metric
  AT_N_vec DA_u;        // Covariant derivative of A
  AT_N_vec s_u;         // x^i/r where r is the coord. radius

  AT_N_sym g_uu;        // inverse of conf. metric
  AT_N_sym A_uu;        // inverse of A
  AT_N_sym AA_dd;       // g^cd A_ac A_db
  AT_N_sym R_dd;        // Ricci tensor
  AT_N_sym Rphi_dd;     // Ricci tensor, conformal contribution
  AT_N_sym Kt_dd;       // conformal extrinsic curvature
  AT_N_T2 K_ud;         // extrinsic curvature
  AT_N_sym Ddalpha_dd;  // 2nd differential of the lapse
  AT_N_sym Ddphi_dd;    // 2nd differential of phi

  AT_N_VS2 Gamma_ddd;   // Christoffel symbols of 1st kind
  AT_N_VS2 Gamma_udd;   // Christoffel symbols of 2nd kind
  AT_N_VS2 DK_ddd;      // differential of K
  AT_N_VS2 DK_udd;      // differential of K

  // Spatially dependent shift damping
#if defined(Z4C_ETA_CONF) || defined(Z4C_ETA_TRACK_TP)
  AT_N_sca eta_damp;
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  // auxiliary derivatives
  AT_N_sca dbeta;       // d_a beta^a

  AT_N_vec dalpha_d;    // lapse 1st drvts
  AT_N_vec ddbeta_d;    // 2nd "divergence" of beta
  AT_N_vec dchi_d;      // chi 1st drvts
  AT_N_vec dphi_d;      // phi 1st drvts
  AT_N_vec dK_d;        // K 1st drvts
  AT_N_vec dKhat_d;     // Khat 1st drvts
  AT_N_vec dTheta_d;    // Theta 1st drvts

  AT_N_sym ddalpha_dd;  // lapse 2nd drvts
  AT_N_T2  dbeta_du;    // shift 1st drvts
  AT_N_sym ddchi_dd;    // chi 2nd drvts
  AT_N_T2  dGam_du;     // Gamma 1st drvts

  AT_N_VS2 dg_ddd;      // metric 1st drvts
  AT_N_VS2 dK_ddd;      // K 1st drvts
  AT_N_VS2 dA_ddd;      // A 1st drvts

  AT_N_S2V ddbeta_ddu; // shift 2nd drvts

  AT_N_S2S2 ddg_dddd;   // metric 2nd drvts

  // auxiliary Lie derivatives along the shift vector
  AT_N_sca Lchi;        // Lie derivative of chi
  AT_N_sca LKhat;       // Lie derivative of Khat
  AT_N_sca LTheta;      // Lie derivative of Theta
  AT_N_sca Lalpha;      // Lie derivative of the lapse
  AT_N_vec LGam_u;      // Lie derivative of Gamma
  AT_N_vec Lbeta_u;     // Lie derivative of the shift
  AT_N_sym Lg_dd;       // Lie derivative of conf. 3-metric
  AT_N_sym LA_dd;       // Lie derivative of A

  //WGC wext - TODO fix tensor index symmetries
  //auxiliary wave extraction tensors
  AT_N_vec uvec;        // radial vector in tetrad
  AT_N_vec vvec;        // theta vector in tetrad
  AT_N_vec wvec;        // phi vector in tetrad
  AT_N_sca dotp1;       // dot product in Gram-Schmidt orthonormalisation
  AT_N_sca dotp2;       // second dot product in G-S orthonormalisation
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 4> Riem3_dddd;  // 3D Riemann tensor
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 4> Riemm4_dddd; // 4D Riemann tensor
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 3> Riemm4_ddd;  // 4D Riemann * n^a
  AT_N_T2  Riemm4_dd;   // 4D Riemann *n^a*n^c

  // Aux vars handling cx/vc matter interpolation
  AT_N_sca w_rho;
  AT_N_sca w_p;
  AT_N_vec w_utilde_u;
#if USETM
  AT_S_vec w_r;
#endif

#if MAGNETIC_FIELDS_ENABLED
  AT_N_vec bb;
#endif

private:
  void Z4cSommerfeld_(
    AA & u, AA & rhs,
    const int is, const int ie,
    const int js, const int je,
    const int ks, const int ke
  );

private:
  FiniteDifference::Uniform *fd;

};

#endif // Z4c_HPP
