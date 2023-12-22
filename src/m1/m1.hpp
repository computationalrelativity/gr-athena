#ifndef M1_HPP
#define M1_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file M1.hpp
//  \brief definitions for the M1 class
//
// Convention: tensor names are followed by tensor type suffixes:
//    _u --> contravariant component
//    _d --> covariant component
// For example g_dd is a tensor, or tensor-like object, with two covariant indices.

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../utils/lagrange_interp.hpp"
#include "../utils/interp_intergrid.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../bvals/vc/bvals_vc.hpp"

#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp" // (k,j,i) loops
#endif

#include "../utils/tensor.hpp" // TensorPointwise

#include "fake_opacities.hpp"
#include "photon_opacities.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_errno.h>

// Source update method
#define M1_SRC_METHOD_EXPL (0)
#define M1_SRC_METHOD_IMPL (1)
#define M1_SRC_BOOST (2)

// enum {
//   M1_SRC_METHOD_IMPL, // 0: implicit (default)
//   M1_SRC_BOOST,       // 1: boost to fluid frame (approximate!)
//   M1_SRC_METHOD_EXPL, // 2: explicit
// };

#ifndef M1_SRC_METHOD
#define M1_SRC_METHOD (M1_SRC_METHOD_IMPL)
#endif

#ifndef M1_NGHOST
#define M1_NGHOST (2) //TODO: check we have enough, NGHOSTS >= M1_NGHOSTS
#endif

#ifndef M1_NSPECIES
#define M1_NSPECIES (1)
#endif

#ifndef M1_NGROUPS
#define M1_NGROUPS (1)
#endif

#ifndef M1_EPSILON
#define M1_EPSILON (1e-10)
#endif

#ifndef M1_WARN_FOR_SRC_FIX
#define M1_WARN_FOR_SRC_FIX (1)
#endif

#ifndef M1_USE_EIGENVALUES_THIN
#define M1_USE_EIGENVALUES_THIN (0)
#endif

#define M1_DEBUG (0)
#define M1_DEBUG_PR(var)\
  if (M1_DEBUG) { std::cout << "M1_DEBUG: " << var << std::endl; }
#define M1_CALCFIDUCIALVELOCITY_OFF (0)
#define M1_CALCCLOSURE_OFF (0)
#define M1_CALCOPACITY_OFF (0)
#define M1_GRSOURCES_OFF (1)
#define M1_FLUXX_SET_ZERO (0)
#define M1_FLUXY_SET_ZERO (0)
#define M1_FLUXZ_SET_ZERO (0)

// CGS density conv. fact
#define CGS_GCC (1.619100425158886e-18) 

using namespace utils::tensor;

class MeshBlock;
class ParameterInput;

typedef Real (*closure_t)(Real const);
Real eddington(Real const xi);
Real kershaw(Real const xi);
Real minerbo(Real const xi);
Real thin(Real const xi);


//FIXME (temporary stuff)
typedef Real (*average_baryon_mass_t)();
typedef int (*NeutrinoRates_t)(Real const rho, Real const temperature, Real const Y_e,
			     Real const nudens_00, Real const nudens_10, Real const chi_loc0,
			     Real const nudens_01, Real const nudens_11, Real const chi_loc1,
			     Real const nudens_02, Real const nudens_12, Real const chi_loc2,
			     Real * eta_0_loc0, Real * eta_0_loc1, Real * eta_0_loc2,
			     Real * eta_1_loc0, Real * eta_1_loc1, Real * eta_1_loc2,
			     Real * abs_0_loc0, Real * abs_0_loc1, Real * abs_0_loc2,
			     Real * abs_1_loc0, Real * abs_1_loc1, Real * abs_1_loc2,
			     Real * scat_0_loc0, Real * scat_0_loc1, Real * scat_0_loc2,
			     Real * scat_1_loc0, Real * scat_1_loc1, Real * scat_1_loc2);
typedef int (*WeakEquilibrium_t)(Real const rho, Real const temperature, Real const Y_e,
				 Real const nudens_00, Real const nudens_01, Real const nudens_02,
				 Real const nudens_10, Real const nudens_11, Real const nudens_12,
				 Real * temperature_trap, Real * Y_e_trap,
				 Real * nudens_0_trap0, Real * nudens_0_trap1, Real * nudens_0_trap2,
			       Real * nudens_1_trap0, Real * nudens_1_trap1, Real * nudens_1_trap2);
typedef int (*NeutrinoDensity_t)(Real const rho, Real const temperature, Real const Y_e,
				 Real * nudens_0_thin0, Real * nudens_0_thin1, Real * nudens_0_thin2,
				 Real * nudens_1_thin0, Real * nudens_1_thin1, Real * nudens_1_thin2);


// Indexes of spacetime manifold vars in TensorPointwise 
#define MDIM (4)

// Indexes for hypersurface manifold vars in AthenaArray
#define NDIM (3) // Manifold dimension

// FIXME: do not use static member functions
//! \class M1
//  \brief M1 data and functions
class M1 {

public:

  // Indexes of Lab frame variables
  enum {
    I_Lab_E,
    I_Lab_Fx, I_Lab_Fy, I_Lab_Fz,
    I_Lab_N,
    N_Lab
  };
  // Names of Lab frame variables
  static char const * const Lab_names[N_Lab];

  // Indexes of fluid frame radiation variables + P_{ij}, etc.
  enum {
    I_Rad_nnu,
    I_Rad_J,
    I_Rad_Ht,
    I_Rad_Hx, I_Rad_Hy, I_Rad_Hz,
    I_Rad_Pxx, I_Rad_Pxy, I_Rad_Pxz, I_Rad_Pyy, I_Rad_Pyz, I_Rad_Pzz,
    I_Rad_chi,
    I_Rad_ynu,
    I_Rad_znu,
    N_Rad
  };
  // Names of M1 fluid frame and other radiation variables
  static char const * const Rad_names[N_Rad];

  // Indexes of radiation-matter variables
  enum {
    I_RadMat_abs_0, I_RadMat_abs_1,
    I_RadMat_eta_0, I_RadMat_eta_1,
    I_RadMat_scat_1,
    I_RadMat_nueave,
    N_RadMat
  };
  // Names of radiation-matter variables
  static char const * const RadMat_names[N_RadMat];

  // Indexes of diagnostic variables
  enum {
    I_Diagno_radflux_0,
    I_Diagno_radflux_1,
    I_Diagno_ynu,
    I_Diagno_znu,
    N_Diagno
  };
  // Names of matter variables
  static char const * const Diagno_names[N_Diagno];

  // Indexes of internal variables (no group dimension)
  enum {  
    I_Intern_fidu_vx, I_Intern_fidu_vy, I_Intern_fidu_vz,
    I_Intern_fidu_Wlorentz,
    I_Intern_netabs,
    I_Intern_netheat,
    I_Intern_mask,
    N_Intern
  };
  // Names of internal variables
  static char const * const Intern_names[N_Intern];
  
  // Source update results
  enum {
    M1_SRC_UPDATE_OK,
    M1_SRC_UPDATE_THIN,
    M1_SRC_UPDATE_EQUIL,
    M1_SRC_UPDATE_SCAT,
    M1_SRC_UPDATE_EDDINGTON,
    M1_SRC_UPDATE_FAIL,
    M1_SRC_UPDATE_RESULTS,
  };
  // Messages
  static char const * const source_update_msg[M1_SRC_UPDATE_RESULTS];

public:
  M1(MeshBlock *pmb, ParameterInput *pin);
  ~M1();

  MeshBlock * pmy_block;     // pointer to MeshBlock containing this M1
  FakeOpacities * fake_opac;     // pointer to fake opacities
  PhotonOpacities * photon_opac;     // pointer to photon opacities
  //NeutrinoOpacities * neutrino_opac;     // pointer to neutrino opacities //TODO
  
  // public data storage
  struct {
    AthenaArray<Real> u;       // solution of M1 evolution system
    AthenaArray<Real> u1;      // solution at intermediate steps
    AthenaArray<Real> flux[3]; // flux in the 3 directions
    AthenaArray<Real> u_rhs;   // M1 rhs
    AthenaArray<Real> u_rad;   // fluid frame variables + P_{ij} Lab
    AthenaArray<Real> radmat;  // radiation-matter fields
    AthenaArray<Real> diagno;  // analysis buffers
    AthenaArray<Real> intern;  // "internals": fiducial velocity, netabs, .. these do not have group dimension!
  } storage;

  // aliases for Lab variables and RHS
  struct Lab_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> E; 
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> F_d;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> N;
  };
  Lab_vars lab;
  Lab_vars rhs;

  // aliases for the fluid variables + P_ij Lab, etc.
  struct Rad_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> nnu;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> J;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Ht;    
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> H;
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> P_dd; // Lab frame (normalized by E)
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> ynu;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> znu;
  };
  Rad_vars rad;

  // aliases for the radiation-matter variables
  struct RadMat_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> abs_0;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> abs_1;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> eta_0;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> eta_1;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> scat_1;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> nueave;
  };
  RadMat_vars rmat;
  
  // aliases for the diagnostic variables
  struct Diagno_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> radflux_0;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> radflux_1;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> ynu;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> znu;
  };
  Diagno_vars rdia;

  // aliases for the fiducial vel. variables (no group dependency)
  struct Fidu_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> vel_u;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Wlorentz;
  };
  Fidu_vars fidu;
  
  // aliases for the net heat and abs (no group dependency)
  struct Net_vars {
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> abs;
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> heat;
  };
  Net_vars net;

  // Excision mask
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> m1_mask;
  
  // Parameters
  int nspecies;
  int ngroups;
  std::string closure;
  std::string opacities;
  std::string fiducial_velocity;      // Prescription for fiducial velocity; zero if not {"fluid","mixed"}
  Real fiducial_vel_rho_fluid;        // Density above which the fluid velocity is used (CGS) (if "mixed")
  Real opacity_equil_depth;           // Enforce Kirchhoff's law if the cell optical depth is larger than this
  Real opacity_corr_fac_max;          // Maximum correction factor for optically thin regime
  Real opacity_tau_trap;              // Include the effect of neutrino trapping above this optical depth (NB <0 never assume trapping)
  Real opacity_tau_delta;             // Range of optical depths over which trapping is introduced
  
  Real rad_E_floor;                   // Radiation energy density floor
  Real rad_N_floor;                   // Radiation number density floor
  Real closure_epsilon;               // recision with which to find the closure
  int closure_maxiter;                // Maximum number of iterations in the closure root finding
  Real source_limiter;                // Limit the source terms to avoid nonphysical states
  bool backreact;                     // Backreact on the fluid
  Real rad_eps;                       // Impose F_a F^a < (1 - rad_E_eps) E2
  bool set_to_equilibrium;            // Initialize everything to thermodynamic equilibrium
  bool reset_to_equilibrium;          // Set everything to equilibrium at recovery
  Real equilibrium_rho_min;           // Set to equilibrium only if the density is larger than this (CGS)
  Real source_therm_limit;            // Assume neutrinos to be thermalized above this optical depth
  Real source_thick_limit;            // Use the optically thick limit if the equilibration time is less than the timestep over this factor
  Real source_scat_limit;             // Use the scattering limit if the isotropization time is less than the timestep over this factor 
  Real source_epsabs;                 // Target absolute precision for the nonlinear solver
  Real source_epsrel;                 // Target relative precision for the nonlinear solver
  int source_maxiter;                 // Maximum number of iterations in the nonlinear solver
  Real mindiss;                       // Minimum numberical dissipation (use with caution)
  Real minmod_theta;                  // Theta parameter used for the minmod limiter, in (0,2)
  
  // Problem-specific parameters
  std::string m1_test; // Simple tests:
  // "beam"        :: "Evolve a single beam propagating through the grid"
  // "diffusion"   :: "Diffusion test"
  // "equilibrium" :: "Thermal equilibrium test"
  // "kerrschild"  :: "Radiation beam in Kerr geometry"
  // "none"        :: "No test is performed (production mode)"
  // "shadow"      :: "Sphere shadow test"
  // "sphere"      :: "Homogeneous sphere test"
  // ---
  Real beam_dir[3];        // Direction of propagation for the beam (internally normalized)
  Real beam_position[3];   // Offset with respect to the plane with normal beam_test_dir passing through the origin
  Real beam_width;         // Width of the beam
  Real equil_nudens_0[3];  // Comoving neutrino number densities // Hardocoded to 3 species!
  Real equil_nudens_1[3];  // Comoving neutrino energy densities // Hardocoded to 3 species!

  std::string diff_profile;           // Diffusion profile { "step", "gaussian" }
  Real medium_velocity;               // Modulus of fiducial velocity for tests
  
  Real kerr_beam_position;            // Position at which the ray is injected
  Real kerr_beam_width;               // Width of the beam used for the kerr test
  Real kerr_mask_radius;              // Excision radius for the Kerr Schild test
  
  // Intergrid interpolation 
  // boundary and grid data
  CellCenteredBoundaryVariable ubvar;

  // storage for SMR/AMR
  // BD: this should perhaps be combined with the above stuct.
  AthenaArray<Real> coarse_u_;
  int refinement_idx{-1};

  // for seamless CC/VC switching
  struct MB_info {
    int il, iu, jl, ju, kl, ku;        // local block iter.
    int nn1, nn2, nn3;                 // number of nodes (simplify switching)
    AthenaArray<Real> x1, x2, x3;      // for CC / VC grid switch
    AthenaArray<Real> cx1, cx2, cx3;   // for CC / VC grid switch (coarse)
  };

  MB_info mbi;
  
public:
  void CalcFiducialVelocity();
  void SetEquilibrium(AthenaArray<Real> & u);
  void AddToADMMatter(AthenaArray<Real> & u);
  void CalcOpacity(Real const dt,AthenaArray<Real> & u);
  void CalcOpacityNeutrinos(Real const dt,AthenaArray<Real> & u);
  void CalcOpacityPhotons(Real const dt,AthenaArray<Real> & u);
  void CalcOpacityFake(Real const dt,AthenaArray<Real> & u);
  void CalcClosure(AthenaArray<Real> & u);
  void CalcFluxes(AthenaArray<Real> & u);
  void AddFluxDivergence(AthenaArray<Real> & u_rhs);
  void GRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs);
  void CalcUpdate(const Real dt, AthenaArray<Real> & u_p, AthenaArray<Real> & u_c,
		  AthenaArray<Real> & u_rhs);
  void CalcUpdate_advection(const Real dt, AthenaArray<Real> & u_p, AthenaArray<Real> & u_c,
			    AthenaArray<Real> & u_rhs);

  average_baryon_mass_t AverageBaryonMass;
  
  // compute new timestep on a MeshBlock
  Real NewBlockTimeStep(void);

  // set aliases 
  void SetZeroLabVars(AthenaArray<Real> & u);
  void SetZeroFiduVars(AthenaArray<Real> & u);
  void SetLabVarsAliases(AthenaArray<Real> & u, Lab_vars & lab);
  void SetRadVarsAliases(AthenaArray<Real> & r, Rad_vars & rad);
  void SetRadMatVarsAliases(AthenaArray<Real> & radmat, RadMat_vars & rmat);
  void SetDiagnoVarsAliases(AthenaArray<Real> & diagno, Diagno_vars & rdia);
  void SetFiduVarsAliases(AthenaArray<Real> & intern, Fidu_vars & fid);
  void SetNetVarsAliases(AthenaArray<Real> & intern, Net_vars & net);
  
  // initial data for the tests
  void SetupZeroVars(AthenaArray<Real> & u);
  void SetupBeamTest(AthenaArray<Real> & u);
  void SetupAdvectionJumpTest(AthenaArray<Real> & u);
  void SetupDiffusionTest(AthenaArray<Real> & u);
  void SetupEquilibriumTest(AthenaArray<Real> & u);
  void SetupKerrSchildMask(AthenaArray<Real> & u);
  void BeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void KerrBeamInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int ngh);
  void OutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void OutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void OutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void ReflectInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void ReflectInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void ReflectInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &u,
                Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
  void SetupTestHydro();

  // wrappers/interfaces with GSL for source update and closure
  void prepare_closure(gsl_vector const * q, void * params);
  void prepare_sources(gsl_vector const * q, void * params);
  double zFunction(double xi, void * params);
   
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep
  // scratch space used to compute fluxes
  //AthenaArray<Real> dxw_;
  AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
  AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
  AthenaArray<Real> cell_volume_;
  AthenaArray<Real> dflx_; //  (N_Lab, ngroups*nspecies, ncells1)

  NeutrinoRates_t NeutrinoRates;
  WeakEquilibrium_t WeakEquilibrium;
  NeutrinoDensity_t NeutrinoDensity;

  // m1_source_update.cpp  
  int source_update_pt(MeshBlock * pmb,
		       int const i,
		       int const j,
		       int const k,
		       int const ig,
		       closure_t closure_fun,
		       gsl_root_fsolver * gsl_solver_1d,
		       gsl_multiroot_fdfsolver * gsl_solver_nd,
		       Real const cdt,
		       Real const alp,
		       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
		       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & gamma_ud,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
		       Real const W,
		       Real const Eold,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & Fold_d,
		       Real const Estar,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & Fstar_d,
		       Real * chi,
		       Real const eta,
		       Real const kabs,
		       Real const kscat,
		       Real * Enew,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & Fnew_d);  
  
  // m1_closure.cpp
  void calc_closure_pt(MeshBlock * pmb,
		       int const i, int const j, int const k,
		       int const ig,
		       closure_t closure_fun,
		       gsl_root_fsolver * fsolver,
		       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
		       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
		       Real const w_lorentz,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
		       Real const E,
		       TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		       Real * chi,
		       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd);

  void apply_closure(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
		     TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
		     Real const w_lorentz,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
		     Real const E,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		     Real const chi,
		     TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd);
  
  // m1_utils.cpp
  void calc_proj(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
                 TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
                 TensorPointwise<Real, Symmetries::NONE, MDIM, 2> & proj_ud);
  void calc_Pthin(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		  Real const E,
		  TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd);
  void calc_Pthick(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
		   TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_uu,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
		   Real const W,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_d,
		   Real const E,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		   TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd);
  void assemble_fnu(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
                    Real const J,
                    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_u,
                    TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & fnu_u);
  Real compute_Gamma(Real const W,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & v_u,
		     Real const J, Real const E,
		     TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		     Real rad_E_floor, Real rad_eps);
  void assemble_rT(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
		   Real const J,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_d,
		   TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & K_dd,
		   TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & rT_dd);

  Real calc_J_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u);
  void calc_H_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_u,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & H_d);
  void calc_K_from_rT(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & rT_dd,
                      TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & proj_ud,
                      TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & K_dd);
  void calc_rad_sources(Real const eta,
			Real const kabs,
			Real const kscat,
			TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & u_d,
			Real const J,
			TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const H_d,
			TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & S_d);

  Real calc_rE_source(Real const alp,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & S_d);
  void calc_rF_source(Real const alp,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const gamma_ud,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & S_d,
		      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & tS_d);
  
  Real calc_E_flux(Real const alp,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
		   Real const E,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_u,
		   int const dir);
  Real calc_F_flux(Real const alp,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		   TensorPointwise<Real, Symmetries::NONE, MDIM, 2> const & P_ud,
		   int const dir,
		   int const comp);

  void apply_floor(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const g_uu,
                   Real * E,
                   TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & F_d);
  void uvel(Real alp,
            Real betax,Real betay,Real betaz,
            Real w_lorentz,
            Real velx,Real vely,Real velz,
            Real * u0, Real * u1, Real * u2, Real * u3);
  void uvel(TensorPointwise<Real, Symmetries::NONE, NDIM, 0> const & alpha,
	            TensorPointwise<Real, Symmetries::NONE, NDIM, 1> const & beta_u,
	            Real const w_lorentz,
	            TensorPointwise<Real, Symmetries::NONE, NDIM, 1> const & vel_u,
	            TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & u_u);

  void pack_F_d(Real const betax, Real const betay, Real const betaz,
		Real const Fx, Real const Fy, Real const Fz,
		TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & F_d);
  void unpack_F_d(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & F_d,
		  Real * Fx, Real * Fy, Real * Fz);
  void pack_F_d(Real const Fx, Real const Fy, Real const Fz,
		TensorPointwise<Real, Symmetries::NONE, MDIM-1, 1> & F_d);
  void unpack_F_d(TensorPointwise<Real, Symmetries::NONE, MDIM-1, 1> const & F_d,
		  Real * Fx, Real * Fy, Real * Fz);
  void pack_H_d(Real const Ht, Real const Hx, Real const Hy, Real const Hz,
                TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & H_d);
  void unpack_H_d(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & H_d,
                  Real * Ht, Real * Hx, Real * Hy, Real * Hz);
  void pack_P_dd(Real const betax, Real const betay, Real const betaz,
                 Real const Pxx, Real const Pxy, Real const Pxz,
                 Real const Pyy, Real const Pyz, Real const Pzz,
                 TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & P_dd);
  void unpack_P_dd(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & P_dd,
                   Real * Pxx, Real * Pxy, Real * Pxz,
                   Real * Pyy, Real * Pyz, Real * Pzz);
  void pack_P_dd(Real const Pxx, Real const Pxy, Real const Pxz,
                 Real const Pyy, Real const Pyz, Real const Pzz,
                 TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> & P_dd);
  void unpack_P_dd(TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> const & P_dd,
                   Real * Pxx, Real * Pxy, Real * Pxz,
                   Real * Pyy, Real * Pyz, Real * Pzz);
  void pack_P_ddd(Real const Pxxx, Real const Pxxy, Real const Pxxz,
                  Real const Pxyy, Real const Pxyz, Real const Pxzz,
                  Real const Pyxx, Real const Pyxy, Real const Pyxz,
                  Real const Pyyy, Real const Pyyz, Real const Pyzz,
                  Real const Pzxx, Real const Pzxy, Real const Pzxz,
                  Real const Pzyy, Real const Pzyz, Real const Pzzz,
                  TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 3> & P_ddd);
  void pack_v_u(Real const velx, Real const vely, Real const velz,
                TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & v_u);
  
  Real GetWLorentz_from_utilde(Real const utx, Real const uty, Real const utz,
                               Real const gxx, Real const gxy, Real const gxz,
                               Real const gyy, Real const gyz, Real const gzz,
                               Real * utlx, Real * utly, Real * utlz,
                               Real *ut2);
  void Get4Metric_VC2CCinterp(MeshBlock * pmb,
                              const int k, const int j, const int i,
                              AthenaArray<Real> & u,
                              AthenaArray<Real> & u_adm,
                              TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_dd,
                              TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & beta_u,
                              TensorPointwise<Real, Symmetries::NONE, MDIM, 0> & alpha);
  void Get4Metric_Inv(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
                      TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
                      TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
                      TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_uu);
  void Get4Metric_Inv_Inv3(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
                           TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
                           TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
                           TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_uu,
                           TensorPointwise<Real, Symmetries::SYM2, MDIM-1, 2> & gam_uu);
  void Get4Metric_ExtrCurv_VC2CCinterp(MeshBlock * pmb,
                                       const int k, const int j, const int i,
                                       AthenaArray<Real> & u_adm,
                                       TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & K_dd);
  void Get4Metric_Normal(TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & beta_u,
                         TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & n_u);
  void Get4Metric_NormalForm(TensorPointwise<Real, Symmetries::NONE, MDIM, 0> const & alpha,
                             TensorPointwise<Real, Symmetries::NONE, MDIM, 1> & n_d);
  void Get4Metric_SpaceProj(TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_u,
                            TensorPointwise<Real, Symmetries::NONE, MDIM, 1> const & n_d,
                            TensorPointwise<Real, Symmetries::NONE, MDIM, 2> & gamma_ud);
  Real SpatialDet(Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz);
  Real SpatialDet(TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> const & g_dd);
  Real SpatialDet(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd);
  void SpatialInv(Real const detginv,
                  Real const gxx, Real const gxy, Real const gxz,
                  Real const gyy, Real const gyz, Real const gzz,
                  Real * uxx, Real * uxy, Real * uxz,
                  Real * uyy, Real * uyz, Real * uzz);
  void SpatialInv(TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> const & g_dd,
                  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> & g_uu);
  void SpatialInv(TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> const & g_dd,
                  TensorPointwise<Real, Symmetries::SYM2, MDIM, 2> & g_uu);
  
};

#endif // M1_HPP
