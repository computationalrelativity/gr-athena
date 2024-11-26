#ifndef M1_HPP
#define M1_HPP
// Ref(s):
// [1]: Radice, David, et al. "A new moment-based general-relativistic
//      neutrino-radiation transport code: Methods and first applications to
//      neutron star mergers." Monthly Notices of the Royal Astronomical
//      Society 512.1 (2022): 1499-1521.
//
// [2]: Izquierdo, Manuel R., et al. "Global high-order numerical schemes for
//      the time evolution of the general relativistic radiation
//      magneto-hydrodynamics equations." Classical and Quantum Gravity 40.14
//      (2023): 145014.

// c++
#include <iostream>

// Athena++ classes headers
#include "../athena_aliases.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "m1_containers.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>

// Forward declarations
namespace M1::Opacities {
class Opacities;
}


// ============================================================================
namespace M1 {
// ============================================================================

// M1 settings ----------------------------------------------------------------

// Class ======================================================================

class M1
{

// internal solver data =======================================================
public:
  gsl_root_fsolver   * gsl_brent_solver;
  gsl_root_fdfsolver * gsl_newton_solver;

// methods ====================================================================
public:
  M1(MeshBlock *pmb, ParameterInput *pin);
  ~M1();

  void CalcFiducialVelocity();
  void CalcClosure(AA & u);
  void CalcFiducialFrame(AA & u);
  void CalcOpacity(Real const dt, AA & u);
  // BD: TODO - Deprecated.. retain for testing..
  void CalcUpdate(Real const dt,
                  AA & u_pre,
                  AA & u_cur,
		              AA & u_inh,
                  AA & sources);

  void CalcUpdateNew(Real const dt,
                     AA & u_pre,
                     AA & u_cur,
		                 AA & u_inh,
                     AA & sources);

  void CalcFluxes(AA & u);
  void AddFluxDivergence(AA & u_inh);
  void AddSourceGR(AA & u, AA & u_inh);
  void AddSourceMatter(AA & u, AA & u_inh, AA & u_src);

  Real NewBlockTimeStep();

  void CoupleSourcesADM(AT_C_sca &A_rho, AT_N_vec &A_S_d, AT_N_sym & A_S_dd);
  void CoupleSourcesHydro(const Real weight, AA &cons);
  void CoupleSourcesYe(   const Real weight, const Real mb, AA &ps);

  // `master_m1` style
  void CoupleSourcesHydro(AA &cons);
  void CoupleSourcesYe(const Real mb, AA &ps);

  void PerformAnalysis();

// data =======================================================================
public:
  // Mesh->MeshBlock->M1
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;

  // Athena++ imposes BC as a monolith. This requires an awkward work-around:
  bool enable_user_bc { false };

  // M1-indicial information
  MB_info mbi;

  // Species and groups (from parameter file)
  const int N_GRPS;
  const int N_SPCS;
  const int N_GS;

  struct
  {
    AA u;         // solution of M1 evolution system
    AA u1;        // solution at intermediate steps
    AA flux[3];   // flux in the 3 directions
    AA u_rhs;     // M1 rhs
    AA u_lab_aux; // lab frame auxiliaries
    AA u_rad;     // fluid frame variables
    AA u_sources;
    AA radmat;    // radiation-matter fields
    AA diagno;    // analysis buffers
    // "internals": fiducial velocity, netabs, ..
    // N.B. these do not have group dimension!
    AA intern;
  } storage;

  // Variables to deal with refinement
  AthenaArray<Real> coarse_u_;
  CellCenteredBoundaryVariable ubvar;
  int refinement_idx{-1};

// configuration ==============================================================
public:
  // BD: TODO - after refactor remove extraneous named..
  enum class opt_integration_strategy { full_explicit,
                                        explicit_approximate_semi_implicit,
                                        semi_implicit_PicardFrozenP,
                                        semi_implicit_PicardMinerboP,
                                        semi_implicit_PicardMinerboPC,
                                        semi_implicit_Hybrids,
                                        semi_implicit_HybridsJ,
                                        semi_implicit_HybridsJFrozenP,
                                        semi_implicit_HybridsJMinerbo,
                                        auto_esi_Hybrids,
                                        auto_esi_HybridsJMinerbo,
                                        auto_esi_PicardMinerboP};
  enum class opt_fiducial_velocity { fluid, mixed, zero, none };

  enum class opt_flux_variety { HybridizeMinModA,
                                HybridizeMinModB,
                                HybridizeMinMod,
                                LO };

  enum class opt_characteristics_variety { approximate,
                                           exact_thin,
                                           exact_thick,
                                           exact_Minerbo };
  enum class opt_closure_variety { thin,
                                   thick,
                                   Minerbo,
                                   MinerboP,
                                   MinerboB,
                                   MinerboN };

  enum class opt_closure_method {
    gsl_Brent, Newton, Picard
  };

  struct
  {
    bool use_split_step;

    // Control flux calculation
    opt_characteristics_variety characteristics_variety;

    // Switching between style of flux calculation / limiter
    opt_flux_variety flux_variety;

    // Prescription for fiducial velocity; zero if not {"fluid","mixed"}
    opt_fiducial_velocity fiducial_velocity;
    Real fiducial_velocity_rho_fluid;

    // Various tolerances / ad-hoc fiddle parameters
    Real fl_E;
    Real fl_J;
    Real eps_E;
    Real eps_J;
    bool enforce_causality;
    Real eps_ec_fac;
    Real min_flux_A;

    // Control the couplings
    bool couple_sources_ADM;
    bool couple_sources_hydro;
    bool couple_sources_Y_e;

    // debugging:
    bool value_inject;
  } opt;

  struct
  {
    opt_closure_variety variety;
    opt_closure_method method;

    Real eps_tol;
    Real eps_Z_o_E;
    Real fac_Z_o_E;
    Real w_opt_ini;
    Real fac_err_amp;

    int iter_max;
    int iter_max_rst;

    bool fallback_thin;
    bool use_Ostrowski;
    bool use_Neighbor;

    bool verbose;
  } opt_closure;

  struct
  {
    // BD: TODO - remove the following var. after refactor
    opt_integration_strategy strategy;

    struct {
      opt_integration_strategy non_stiff;
      opt_integration_strategy stiff;
      opt_integration_strategy scattering;
      opt_integration_strategy equilibrium;
    } solvers;

    bool solver_reduce_to_common;

    Real eps_tol;
    Real w_opt_ini;
    Real fac_err_amp;

    int iter_max;
    int iter_max_rst;

    bool use_Neighbor;

    // source settings
    Real src_lim;
    Real src_lim_Ye_min;
    Real src_lim_Ye_max;
    Real src_lim_thick;
    Real src_lim_scattering;

    // equilibrium parameters
    Real eql_rho_min;

    bool verbose;
  } opt_solver;

// variable alias / storage ===================================================
public:
  // Conventions for fields:
  // sc: (s)calar-(f)ield
  // sp: (sp)atial
  // st: (s)pace-time
  // Appended "_" indicates scratch (in i)

  struct vars_Flux
  {
    GroupSpeciesFluxContainer<AT_C_sca> sc_E;
    GroupSpeciesFluxContainer<AT_N_vec> sp_F_d;
    GroupSpeciesFluxContainer<AT_C_sca> sc_nG;
  };
  vars_Flux fluxes;

  // Eulerian (Lab) variables and RHS
  struct vars_Lab {
    // N.B.
    // These quantities should be viewed as \sqrt(\gamma) densitized
    GroupSpeciesContainer<AT_C_sca> sc_E;
    GroupSpeciesContainer<AT_N_vec> sp_F_d;
    GroupSpeciesContainer<AT_C_sca> sc_nG;
  };
  vars_Lab lab, rhs;

  // Eulerian (Lab) variables: not directly evolved
  struct vars_LabAux {
    // N.B.
    // These quantities should be viewed as \sqrt(\gamma) densitized
    GroupSpeciesContainer<AT_N_sym> sp_P_dd;
    // GroupSpeciesContainer<AT_C_sca> sc_n;

    // Closure weight - not densitized
    GroupSpeciesContainer<AT_C_sca> sc_chi;
    GroupSpeciesContainer<AT_C_sca> sc_xi;
  };
  vars_LabAux lab_aux;

  // Lagrangian (Rad) fiducial frame
  struct vars_Rad
  {
    // N.B.
    // These quantities should be viewed as \sqrt(\gamma) densitized
    GroupSpeciesContainer<AT_C_sca> sc_n;
    GroupSpeciesContainer<AT_C_sca> sc_J;
    GroupSpeciesContainer<AT_C_sca> sc_H_t;
    GroupSpeciesContainer<AT_N_vec> sp_H_d;
  };
  vars_Rad rad;

  // radiation-matter variables
  struct vars_RadMat
  {
    GroupSpeciesContainer<AT_C_sca> sc_eta_0;
    GroupSpeciesContainer<AT_C_sca> sc_kap_a_0;

    GroupSpeciesContainer<AT_C_sca> sc_eta;
    GroupSpeciesContainer<AT_C_sca> sc_kap_a;
    GroupSpeciesContainer<AT_C_sca> sc_kap_s;

    GroupSpeciesContainer<AT_C_sca> sc_avg_nrg;

    // AT_C_sca abs_0;
    // AT_C_sca abs_1;
    // AT_C_sca eta_0;
    // AT_C_sca eta_1;
    // AT_C_sca scat_1;
    // AT_C_sca nueave;
  };
  vars_RadMat radmat;

  struct vars_Source
  {
    GroupSpeciesContainer<AT_C_sca> sc_nG;
    GroupSpeciesContainer<AT_C_sca> sc_E;
    GroupSpeciesContainer<AT_N_vec> sp_F_d;

    // For source limiting
    AT_C_sca theta;
  };
  vars_Source sources;

  // diagnostic variables
  struct vars_Diag
  {
    GroupSpeciesContainer<AT_C_sca> sc_radflux_0;
    GroupSpeciesContainer<AT_C_sca> sc_radflux_1;
    GroupSpeciesContainer<AT_C_sca> sc_y;        // neutrino fractions
    GroupSpeciesContainer<AT_C_sca> sc_z;        // neutrino energies
  };
  vars_Diag rdiag;

  // fiducial vel. variables (no group dependency)
  struct vars_Fidu {
    AT_N_vec sp_v_u;
    AT_N_vec sp_v_d;

    AT_D_vec st_v_u;

    AT_C_sca sc_W;
  };
  vars_Fidu fidu;

  // net heat and abs (no group dependency)
  struct vars_Net {
    AT_C_sca abs;
    AT_C_sca heat;
  };
  vars_Net net;

  // geometric quantities (storage)
  struct vars_Geom {
    // (sc)alar fields
    AT_C_sca sc_alpha;

    // (sp)atial quantities
    AT_N_vec sp_beta_u;
    AT_N_sym sp_g_dd;
    AT_N_sym sp_K_dd;

    // derived quantities
    AT_N_vec sp_beta_d;
    AT_C_sca sc_sqrt_det_g;
    AT_N_sym sp_g_uu;

    AT_N_D1sca sp_dalpha_d;
    AT_N_D1vec sp_dbeta_du;
    AT_N_D1sym sp_dg_ddd;
  };
  vars_Geom geom;

  // hydrodynamical quantities (storage)
  struct vars_Hydro {
    // (sc)alar fields
    AT_C_sca sc_w_rho;
    AT_C_sca sc_w_Ye;
    AT_C_sca sc_w_p;
    AT_C_sca sc_W;

    // (sp)atial quantities
    AT_N_vec sp_w_util_u;
  };
  vars_Hydro hydro;

  // various persistent scratch quantities not fitting elsewhere
  struct vars_Scratch {
    AT_D_sym st_T_rad_;
    AT_D_vec st_S_u_;

    // (sp)atial quantities (scratch: assembled as required)
    AT_N_vec sp_F_u_;
    AT_N_vec sp_f_u_;
    AT_N_vec sp_H_u_;
    AT_N_sym sp_P_uu_;

    // (s)pace-(t)ime quantities (scratch: assembled as required)
    AT_D_vec st_F_d_;
    AT_C_sca sc_norm_st_F_;
    AT_C_sca sc_G_;

    AT_D_sym st_P_dd_;

    AT_C_sca sc_norm_sp_H_;
    AT_C_sca sc_norm_st_H_;
    AT_D_vec st_H_d_;
    AT_D_vec st_H_u_;
    AT_D_vec st_f_u_;

    AT_D_vec st_n_u_;
    AT_D_vec st_n_d_;

    AT_D_vec st_beta_u_;
    AT_D_sym st_g_dd_;
    AT_D_sym st_g_uu_;

    AT_D_bil st_Phyp_ud_;  // projector (based on hypersurf. normal)
    AT_D_bil st_Pfid_ud_;  // projector (based on fiducial vel.)

    AT_D_vec st_w_u_u_;

    // For flux calculation (allocated for N-dim grid / not pencil)
    AT_C_sca F_sca;
    AT_N_vec F_vec;

    AT_C_sca lambda;

    // Assembly of flux-divergence
    AA dflx_;

    // Generic quantities of specific valence
    AT_C_sca sc_A_;
    AT_C_sca sc_B_;
    AT_C_sca sc_C_;
    AT_N_vec sp_vec_A_;
    AT_N_vec sp_vec_B_;
    AT_D_vec st_vec_;
    AT_N_sym sp_sym_A_;
    AT_N_sym sp_sym_B_;
    AT_N_sym sp_sym_C_;

    // Generic dense (dim = N) quantities of specific valence
    AT_C_sca sc_A;
    AT_C_sca sc_B;
  };
  vars_Scratch scratch;

  AT_C_sca m1_mask;  // Excision mask

private:
  // called during ctor for scratch (see descriptions in vars_Scratch)
  void InitializeScratch(vars_Scratch & scratch,
                         vars_Lab     & lab,
                         vars_Rad     & rad,
                         vars_Geom    & geom,
                         vars_Hydro   & hydro,
                         vars_Fidu    & fidu)
  {
    // general scratch --------------------------------------------------------
    scratch.st_T_rad_.NewAthenaTensor(mbi.nn1);
    scratch.st_S_u_.NewAthenaTensor(mbi.nn1);

    scratch.sc_A_.NewAthenaTensor(mbi.nn1);
    scratch.sc_B_.NewAthenaTensor(mbi.nn1);
    scratch.sc_C_.NewAthenaTensor(mbi.nn1);
    scratch.sp_vec_A_.NewAthenaTensor(mbi.nn1);
    scratch.sp_vec_B_.NewAthenaTensor(mbi.nn1);
    scratch.st_vec_.NewAthenaTensor(mbi.nn1);
    scratch.sp_sym_A_.NewAthenaTensor(mbi.nn1);
    scratch.sp_sym_B_.NewAthenaTensor(mbi.nn1);
    scratch.sp_sym_C_.NewAthenaTensor(mbi.nn1);

    // Lab (Eulerian) frame ---------------------------------------------------
    scratch.sp_F_u_.NewAthenaTensor(mbi.nn1);
    scratch.sp_f_u_.NewAthenaTensor(mbi.nn1);

    scratch.st_F_d_.NewAthenaTensor(mbi.nn1);
    scratch.sc_norm_st_F_.NewAthenaTensor(mbi.nn1);
    scratch.sc_G_.NewAthenaTensor(mbi.nn1);

    // Lab (Eulerian) frame auxiliaries ---------------------------------------
    scratch.sp_P_uu_.NewAthenaTensor(mbi.nn1);
    scratch.st_P_dd_.NewAthenaTensor(mbi.nn1);

    // flux-related -----------------------------------------------------------
    scratch.F_sca.NewAthenaTensor( mbi.nn3,mbi.nn2,mbi.nn1);
    scratch.F_vec.NewAthenaTensor( mbi.nn3,mbi.nn2,mbi.nn1);
    scratch.lambda.NewAthenaTensor(mbi.nn3,mbi.nn2,mbi.nn1);

    scratch.dflx_.NewAthenaArray(N_GRPS, N_SPCS, ixn_Lab::N, mbi.nn1);

    // Rad (fiducial) frame ---------------------------------------------------
    scratch.sc_norm_sp_H_.NewAthenaTensor(mbi.nn1);
    scratch.sc_norm_st_H_.NewAthenaTensor(mbi.nn1);
    scratch.sp_H_u_.NewAthenaTensor(mbi.nn1);
    scratch.st_H_d_.NewAthenaTensor(mbi.nn1);
    scratch.st_H_u_.NewAthenaTensor(mbi.nn1);
    scratch.st_f_u_.NewAthenaTensor(mbi.nn1);

    // geometric --------------------------------------------------------------
    scratch.st_n_u_.NewAthenaTensor(mbi.nn1);
    scratch.st_n_d_.NewAthenaTensor(mbi.nn1);

    scratch.st_beta_u_.NewAthenaTensor(mbi.nn1);
    scratch.st_g_dd_.NewAthenaTensor(mbi.nn1);
    scratch.st_g_uu_.NewAthenaTensor(mbi.nn1);

    scratch.st_Phyp_ud_.NewAthenaTensor(mbi.nn1);

    // hydro ------------------------------------------------------------------
    scratch.st_w_u_u_.NewAthenaTensor(mbi.nn1);

    // fiducial ---------------------------------------------------------------
    scratch.st_Pfid_ud_.NewAthenaTensor(mbi.nn1);

    // generic quantities (dense) ---------------------------------------------
    scratch.sc_A.NewAthenaTensor(mbi.nn3,mbi.nn2,mbi.nn1);
    scratch.sc_B.NewAthenaTensor(mbi.nn3,mbi.nn2,mbi.nn1);
  }

// idx & constants ============================================================
public:

  // Lab frame variables
  struct ixn_Lab
  {
    enum { E, F_x, F_y, F_z, nG, N };
    static constexpr char const * const names[] = {
      "lab.E",
      "lab.Fx", "lab.Fy", "lab.Fz",
      "lab.nG"
    };
  };

  // Lab frame variables
  struct ixn_Lab_aux
  {
    enum { P_xx, P_xy, P_xz, P_yy, P_yz, P_zz, chi, xi, N };
    static constexpr char const * const names[] = {
      "lab_aux.Pxx", "lab_aux.Pxy", "lab_aux.Pxz",
      "lab_aux.Pyy", "lab_aux.Pyz", "lab_aux.Pzz",
      "lab_aux.chi",
      "lab_aux.xi",
    };
  };

  // Fluid frame radiation variables + P_{ij}, etc.
  struct ixn_Rad
  {
    enum
    {
      n,
      J,
      H_t,
      H_x, H_y, H_z,
      y,
      z,
      N
    };
    static constexpr char const * const names[] = {
      "rad.n",
      "rad.J",
      "rad.Ht", "rad.Hx", "rad.Hy", "rad.Hz",
      "rad.y",
      "rad.z"
    };
  };

  // Radiation matter coupling source terms.
  struct ixn_Src
  {
    enum
    {
      sc_nG,
      sc_E,
      sp_F_0, sp_F_1, sp_F_2,
      N
    };
    static constexpr char const * const names[] = {
      "src.sc_nG",
      "src.sc_E",
      "src.sp_F_0", "src.sp_F_1", "src.sp_F_2"
    };
  };


  // Radiation-matter variables
  struct ixn_RaM
  {
    enum
    {
      eta_0,
      kap_a_0,
      eta,
      kap_a,
      kap_s,
      avg_nrg,
      // abs_0, abs_1,
      // eta_0, eta_1,
      // scat_1,
      // nueave,
      N
    };
    static constexpr char const * const names[] = {
      "rmat.eta_0",
      "rmat.kap_a_0",
      "rmat.eta",
      "rmat.kap_a",
      "rmat.kap_s",
      "rmat.avg_nrg"
      // "rmat.abs_0", "rmat.abs_1",
      // "rmat.eta_0", "rmat.eta_1",
      // "rmat.scat_1",
      // "rmat.nueave",
    };
  };


  // Diagnostic variables
  struct ixn_Diag
  {
    enum
    {
      radflux_0,
      radflux_1,
      y,
      z,
      N
    };
    static constexpr char const * const names[] = {
      "rdia.radial_flux_0",
      "rdia.radial_flux_1",
      "rdia.y",
      "rdia.z",
    };
  };

  // Internal variables (no group dimension)
  struct ixn_Internal
  {
    enum
    {
      fidu_v_u_x, fidu_v_u_y, fidu_v_u_z,
      fidu_v_d_x, fidu_v_d_y, fidu_v_d_z,
      fidu_st_v_t, fidu_st_v_x, fidu_st_v_y, fidu_st_v_z,
      fidu_W,
      netabs,
      netheat,
      mask,
      N
    };
    static constexpr char const * const names[] = {
      "fidu.v_u_x", "fidu.v_u_y", "fidu.v_u_z",
      "fidu.v_d_x", "fidu.v_d_y", "fidu.v_d_z",
      "fidu.st_vt", "fidu.st_vx", "fidu.st_vy", "fidu.st_vz",
      "fidu.W",
      "net.abs",
      "net.heat",
      "mask",
    };
  };

  // BD: TODO - have solvers (including closures) return status codes
  // Source update results
  /*
  struct ixn_Status
  {
    enum
    {
      OK,
      THIN,
      EQUIL,
      SCAT,
      EDDINGTON,
      FAIL,
      RESULTS,
    };
    static constexpr char const * const msg[] = {
      "Ok",
      "explicit update (thin source)",
      "imposed equilibrium",
      "(scattering dominated source)",
      "imposed eddington",
      "failed",
    };
  };
  */

// opacities ==================================================================
public:
  Opacities::Opacities * popac;

// solver / source treatment dispatch =========================================
public:
  struct evolution_strategy {
    enum class opt_solution_regime {noop,
                                    non_stiff,
                                    stiff,
                                    scattering,
                                    equilibrium,
                                    N};
    enum class opt_source_treatment {noop,
                                     full,
                                     set_zero,
                                     N};
    struct {
      AthenaArray<opt_solution_regime>  solution_regime;
      AthenaArray<opt_source_treatment> source_treatment;
    } masks;
  } ev_strat;

  typedef evolution_strategy::opt_solution_regime  t_sln_r;
  typedef evolution_strategy::opt_source_treatment t_src_t;

  // Different solution techniques are employed point-wise according to the
  // current structure of the fields etc. This function sets internal masks
  // that account for that.
  void PrepareEvolutionStrategy(const Real dt,
                                const Real kap_a,
                                const Real kap_s,
                                t_sln_r & mask_sln_r,
                                t_src_t & mask_src_t);
  void PrepareEvolutionStrategy(const Real dt);

// additional methods =========================================================
public:
  void UpdateGeometry(vars_Geom  & geom,
                      vars_Scratch & scratch);
  void UpdateHydro(vars_Hydro & hydro,
                   vars_Geom & geom,
                   vars_Scratch & scratch);

  void CalcCharacteristicSpeedApproximate(const int dir, AT_C_sca & lambda);

  void CalcCharacteristicSpeed(const int dir,
                               const AT_C_sca & sc_E,
                               const AT_N_vec & sp_F_d,
                               const AT_C_sca & sc_chi,
                               AT_C_sca & lambda);

  inline void MaskSet(const bool is_enabled,
                      const int k, const int j, const int i)
  {
    m1_mask(k,j,i) = is_enabled;

    if (!is_enabled)
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      lab.sc_E( ix_g,ix_s)(k,j,i) = 0.0;
      lab.sc_nG(ix_g,ix_s)(k,j,i) = 0.0;
      for (int a=0; a<N; ++a)
      {
        lab.sp_F_d(ix_g,ix_s)(a,k,j,i) = 0.0;
      }
    }
  }

  inline void MaskThreshold(const int k, const int j, const int i)
  {
    // BD: TODO - add threshold
    //
    // Use nearest-neighbour values & threshold to determine whether
    // anything needs to avoid M1 calculations pointwise in ~0 regions.
    std::cout << "MaskThreshold not implemented" << std::endl;
    std::exit(0);

    // const Real C = 10 * opt.fl_E;
    // Real val (0);
    // for (int K=-1; K<=1; ++K)
    // for (int J=-1; J<=1; ++J)
    // for (int I=-1; I<=1; ++I)
    // {
    //   val = std::max(lab.sc_E(0,0)(k+K,j+J,i+I), val);
    // }

    // m1_mask(k,j,i) = (val > C);
    // MaskSet(m1_mask(k,j,i), k, j, i);
    // m1_mask(k,j,i) = (lab.sc_E(k,j,i) < opt.fl_E) ? false : true;
  }

  inline bool MaskGet(const int k, const int j, const int i)
  {
    return m1_mask(k,j,i);
  }


public:
  // These manipulate internal M1 mem-state; don't call external to class
  //
  // Exposed for pgen
  void InitializeGeometry(vars_Geom & geom,
                          vars_Scratch & scratch);
  void DerivedGeometry(vars_Geom & geom,
                       vars_Scratch & scratch);

  void InitializeHydro(vars_Hydro & hydro,
                       vars_Geom & geom,
                       vars_Scratch & scratch);
  void DerivedHydro(vars_Hydro & hydro,
                    vars_Geom & geom,
                    vars_Scratch & scratch);

public:
  // aliases ------------------------------------------------------------------
  template<typename A_tar>
  inline void SetVarAlias(A_tar & tar, AA & src,
                          const int ix_g, // group
                          const int ix_s, // species
                          const int ix_v, // variable idx in src
                          const int Nv)   // number of vars in src
  {
    // Warning: strange bug-
    //
    // Do not write support fcn for N_gs calc. with such template.
    const int N_gs = (ix_s + N_SPCS * (ix_g + 0)) * Nv;
    tar(ix_g,ix_s).InitWithShallowSlice(src, N_gs+ix_v);
  }

  template<typename A_tar>
  inline void SetVarAlias(A_tar & tar, AA (&src)[M1_NDIM],
                          const int ix_g, // group
                          const int ix_s, // species
                          const int ix_f, // flux direction
                          const int ix_v, // variable idx in src
                          const int Nv)   // number of vars in src
  {
    const int N_gs = (ix_s + N_SPCS * (ix_g + 0)) * Nv;
    tar(ix_g,ix_s,ix_f).InitWithShallowSlice(src[ix_f], N_gs+ix_v);
  }

  void SetVarAliasesFluxes(AA (&u_fluxes)[M1_NDIM], vars_Flux   & fluxes);
  void SetVarAliasesLab(   AA  &u,                  vars_Lab    & lab);
  void SetVarAliasesSource(AA  &sources,            vars_Source & src);
  void SetVarAliasesLabAux(AA  &u,                  vars_LabAux & lab_aux);
  void SetVarAliasesRad(   AA  &r,                  vars_Rad    & rad);
  void SetVarAliasesRadMat(AA  &radmat,             vars_RadMat & rmat);
  void SetVarAliasesDiag(  AA  &diagno,             vars_Diag   & rdia);
  void SetVarAliasesFidu(  AA  &intern,             vars_Fidu   & fid);
  void SetVarAliasesNet(   AA  &intern,             vars_Net    & net);


// internal methods ===========================================================
private:
  void PopulateOptions(         ParameterInput *pin);
  void PopulateOptionsClosure(  ParameterInput *pin);
  void PopulateOptionsSolver(   ParameterInput *pin);
  void PopulateOptionsOpacities(ParameterInput *pin);

// internal data ==============================================================
private:
  AA dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep

// debug ======================================================================
public:
  void StatePrintPoint(
    const int ix_g, const int ix_s,
    const int k, const int j, const int i,
    const bool terminate=true);

};

// ============================================================================
} // namespace M1
// ============================================================================

#endif // M1_HPP

//
// :D
//