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
#include "../mesh/mesh.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "m1_containers.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// M1 settings ----------------------------------------------------------------


// Class ======================================================================

class M1
{

// methods ====================================================================
public:
  M1(MeshBlock *pmb, ParameterInput *pin);
  ~M1() { };

  void CalcFiducialVelocity();
  void CalcClosure(AthenaArray<Real> & u);
  void CalcOpacity(Real const dt, AthenaArray<Real> & u);
  void CalcUpdate(Real const dt,
                  AthenaArray<Real> & u_p,
                  AthenaArray<Real> & u_c,
		              AthenaArray<Real> & u_rhs);
  void CalcFluxes(AthenaArray<Real> & u);
  void AddFluxDivergence(AthenaArray<Real> & u_rhs);
  void AddGRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs);

  Real NewBlockTimeStep();

// data =======================================================================
public:
  // Mesh->MeshBlock->M1
  M1 *pm1;
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  Coordinates *pmy_coord;

  // M1-indicial information
  MB_info mbi;

  // Species and groups (from parameter file)
  const int N_GRPS { 1 };
  const int N_SPCS { 1 };

  struct
  {
    AA u;       // solution of M1 evolution system
    AA u1;      // solution at intermediate steps
    AA flux[3]; // flux in the 3 directions
    AA u_rhs;   // M1 rhs
    AA u_rad;   // fluid frame variables + P_{ij} Lab
    AA radmat;  // radiation-matter fields
    AA diagno;  // analysis buffers
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
  enum class opt_fiducial_velocity { fluid, mixed, zero, none };

  struct
  {
    // Prescription for fiducial velocity; zero if not {"fluid","mixed"}
    opt_fiducial_velocity fiducial_velocity;
    Real fiducial_velocity_rho_fluid;
  } opt;

// idx & constants ============================================================
public:

  // Lab frame variables
  struct ixn_Lab
  {
    enum { E, Fx, Fy, Fz, nG, N };
    static constexpr char const * const names[] = {
      "lab.E",
      "lab.Fx", "lab.Fy", "lab.Fz",
      "lab.nG"
    };
  };

  // Fluid frame radiation variables + P_{ij}, etc.
  enum
  {
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
  static constexpr char const * const names_Rad[] = {
    "rad.nnu",
    "rad.J",
    "rad.Ht", "rad.Hx", "rad.Hy", "rad.Hz",
    "rad.Pxx", "rad.Pxy", "rad.Pxz", "rad.Pyy", "rad.Pyz", "rad.Pzz",
    "rad.chi",
  };

  // Radiation-matter variables
  enum
  {
    I_RadMat_abs_0, I_RadMat_abs_1,
    I_RadMat_eta_0, I_RadMat_eta_1,
    I_RadMat_scat_1,
    I_RadMat_nueave,
    N_RadMat
  };
  static constexpr char const * const names_RadMat[] = {
    "rmat.abs_0", "rmat.abs_1",
    "rmat.eta_0", "rmat.eta_1",
    "rmat.scat_1",
    "rmat.nueave",
  };

  // Diagnostic variables
  enum
  {
    I_Diagno_radflux_0,
    I_Diagno_radflux_1,
    I_Diagno_ynu,
    I_Diagno_znu,
    N_Diagno
  };
  static constexpr char const * const names_Diagno[] = {
    "rdia.radial_flux_0",
    "rdia.radial_flux_1",
    "rdia.ynu",
    "rdia.znu",
  };

  // Internal variables (no group dimension)
  enum
  {
    I_Intern_fidu_vx, I_Intern_fidu_vy, I_Intern_fidu_vz,
    I_Intern_fidu_Wlorentz,
    I_Intern_netabs,
    I_Intern_netheat,
    I_Intern_mask,
    N_Intern
  };
  static constexpr char const * const names_Intern[] = {
    "fidu.vx", "fidu.vy", "fidu.vz",
    "fidu.Wlorentz",
    "net.abs",
    "net.heat",
    "mask",
  };

  // Source update results
  enum
  {
    M1_SRC_UPDATE_OK,
    M1_SRC_UPDATE_THIN,
    M1_SRC_UPDATE_EQUIL,
    M1_SRC_UPDATE_SCAT,
    M1_SRC_UPDATE_EDDINGTON,
    M1_SRC_UPDATE_FAIL,
    M1_SRC_UPDATE_RESULTS,
  };
  static constexpr char const * const source_update_msg[] = {
    "Ok",
    "explicit update (thin source)",
    "imposed equilibrium",
    "(scattering dominated source)",
    "imposed eddington",
    "failed",
  };

  // variable alias / storage -------------------------------------------------
  typedef AthenaTensor<Real, TensorSymm::NONE, ixn_Lab::N, 1> AT_F_vec;

  struct vars_Flux
  {
    GroupSpeciesFluxContainer<AT_N_sca> E;
    GroupSpeciesFluxContainer<AT_N_vec> F_d;
    GroupSpeciesFluxContainer<AT_N_sca> nG;
    GroupSpeciesFluxContainer<AT_F_vec> all;
  };
  vars_Flux fluxes;

  // Lab variables and RHS
  struct vars_Lab {
    GroupSpeciesContainer<AT_N_sca> E;
    GroupSpeciesContainer<AT_N_vec> F_d;
    GroupSpeciesContainer<AT_N_sca> nG;
  };
  vars_Lab lab, rhs;

  // fluid variables + P_ij Lab, etc.
  struct vars_Rad {
    AT_N_sca nnu;
    AT_N_sca J;
    AT_N_sca Ht;
    AT_N_vec H;
    AT_N_sym P_dd; // Lab frame (normalized by E)
    AT_N_sca chi;
    AT_N_sca ynu;
    AT_N_sca znu;
  };
  vars_Rad rad;

  // radiation-matter variables
  struct vars_RadMat {
    AT_N_sca abs_0;
    AT_N_sca abs_1;
    AT_N_sca eta_0;
    AT_N_sca eta_1;
    AT_N_sca scat_1;
    AT_N_sca nueave;
  };
  vars_RadMat rmat;

  // diagnostic variables
  struct vars_Diagno {
    AT_N_sca radflux_0;
    AT_N_sca radflux_1;
    AT_N_sca ynu;
    AT_N_sca znu;
  };
  vars_Diagno rdia;

  // fiducial vel. variables (no group dependency)
  struct vars_Fidu {
    AT_N_vec vel_u;
    AT_N_sca Wlorentz;
  };
  vars_Fidu fidu;

  // net heat and abs (no group dependency)
  struct vars_Net {
    AT_N_sca abs;
    AT_N_sca heat;
  };
  vars_Net net;

  // geometric quantities (storage)
  struct vars_Geom {
    // (sc)alar fields
    AT_N_sca sc_alpha;

    // (sp)atial quantities
    AT_N_vec   sp_beta_u;
    AT_N_sym   sp_g_dd;
    AT_N_sym   sp_K_dd;

    // derived quantities
    AT_N_sca sc_sqrt_det_g;
    AT_N_sym sp_g_uu;

    AT_N_D1sym sp_dg_ddd;

    // (s)pace-(t)ime quantities (scratch assembled as required)
    AT_D_vec st_beta_u_;
    AT_D_sym st_g_dd_;
    AT_D_sym st_g_uu_;
  };
  vars_Geom geom;

  // hydrodynamical quantities (storage)
  struct vars_Hydro {
    // (sc)alar fields
    AT_N_sca sc_w_rho;
    AT_N_sca sc_w_p;
    AT_N_sca sc_W;

    // (sp)atial quantities
    AT_N_vec sp_w_util_u;

    // (s)pace-(t)ime quantities (scratch assembled as required)
    AT_D_vec st_w_u_u_;
  };
  vars_Hydro hydro;

  // various persistent scratch quantities not fitting elsewhere
  struct vars_Scratch {
    AA dflx;
  };
  vars_Scratch scratch;

  AT_N_sca m1_mask;  // Excision mask

// additional methods =========================================================
public:
  void UpdateGeometry(vars_Geom  & geom);
  void UpdateHydro(vars_Hydro & hydro, vars_Geom & geom);

private:
  // These manipulate internal M1 mem-state; don't call external to class
  void InitializeGeometry(vars_Geom  & geom);
  void DerivedGeometry(   vars_Geom  & geom);

  void InitializeHydro(vars_Hydro & hydro, vars_Geom & geom);
  void DerivedHydro(   vars_Hydro & hydro, vars_Geom & geom);

  void InitializeScratch(vars_Scratch & scratch)
  {
    scratch.dflx.NewAthenaArray(N_GRPS, N_SPCS, ixn_Lab::N, mbi.nn1);
  }

public:
  // aliases ------------------------------------------------------------------
  inline int ix_map_GS(const int ix_g, const int ix_s, const int N)
  {
    return (ix_s + N_SPCS * (ix_g + 0)) * N;
  }

  template<typename A_tar>
  inline void SetVarAlias(A_tar & tar, AA & src,
                          const int ix_g, // group
                          const int ix_s, // species
                          const int ix_v, // variable idx in src
                          const int N)    // number of vars in src
  {
    const int N_gs = ix_map_GS(ix_s, ix_g, N);
    tar(ix_g,ix_s).InitWithShallowSlice(src, N_gs+ix_v);
  }

  template<typename A_tar>
  inline void SetVarAlias(A_tar & tar, AA (&src)[M1_NDIM],
                          const int ix_g, // group
                          const int ix_s, // species
                          const int ix_f, // flux direction
                          const int ix_v, // variable idx in src
                          const int N)    // number of vars in src
  {
    const int N_gs = ix_map_GS(ix_s, ix_g, N);
    tar(ix_g,ix_s,ix_f).InitWithShallowSlice(src[ix_f], N_gs+ix_v);
  }

  void SetVarAliasesFluxes(AA (&u_fluxes)[M1_NDIM], vars_Flux   & fluxes);
  void SetVarAliasesLab(   AA  &u,                  vars_Lab    & lab);
  void SetVarAliasesRad(   AA  &r,                  vars_Rad    & rad);
  void SetVarAliasesRadMat(AA  &radmat,             vars_RadMat & rmat);
  void SetVarAliasesDiagno(AA  &diagno,             vars_Diagno & rdia);
  void SetVarAliasesFidu(  AA  &intern,             vars_Fidu   & fid);
  void SetVarAliasesNet(   AA  &intern,             vars_Net    & net);


// internal methods ===========================================================
private:
  void PopulateOptions(ParameterInput *pin);

// internal data ==============================================================
private:
  AA dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep


};

// ============================================================================
} // namespace M1
// ============================================================================

#endif // M1_HPP

//
// :D
//