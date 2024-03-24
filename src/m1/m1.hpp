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
#include "../athena_aliases.hpp"
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
  const int N_GRPS;
  const int N_SPCS;

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


  // variable alias / storage -------------------------------------------------
  // Conventions for fields:
  // sc: (s)calar-(f)ield
  // sp: (sp)atial
  // st: (s)pace-time
  // Appended "_" indicates scratch (in i)

  struct vars_Flux
  {
    GroupSpeciesFluxContainer<AT_N_sca> sc_E;
    GroupSpeciesFluxContainer<AT_N_vec> sp_F_d;
    GroupSpeciesFluxContainer<AT_N_sca> sc_nG;
  };
  vars_Flux fluxes;

  // Lab variables and RHS
  struct vars_Lab {
    GroupSpeciesContainer<AT_N_sca> sc_E;
    GroupSpeciesContainer<AT_N_vec> sp_F_d;
    GroupSpeciesContainer<AT_N_sca> sc_nG;
  };
  vars_Lab lab, rhs;

  // fluid variables + P_ij Lab, etc.
  struct vars_Rad {
    GroupSpeciesContainer<AT_N_sca> sc_nnu;
    GroupSpeciesContainer<AT_N_sca> sc_J;
    GroupSpeciesContainer<AT_N_sca> sc_H_t;
    GroupSpeciesContainer<AT_N_vec> sp_H_d;
    GroupSpeciesContainer<AT_N_sym> sp_P_dd; // Lab frame (normalized by E)
    GroupSpeciesContainer<AT_N_sca> sc_chi;
    GroupSpeciesContainer<AT_N_sca> sc_ynu;
    GroupSpeciesContainer<AT_N_sca> sc_znu;
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
  struct vars_Diag {
    AT_N_sca radflux_0;
    AT_N_sca radflux_1;
    AT_N_sca ynu;
    AT_N_sca znu;
  };
  vars_Diag rdia;

  // fiducial vel. variables (no group dependency)
  struct vars_Fidu {
    AT_N_vec sp_v_u;
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
    AT_N_vec sp_beta_u;
    AT_N_sym sp_g_dd;
    AT_N_sym sp_K_dd;

    // derived quantities
    AT_N_vec sp_beta_d;
    AT_N_sca sc_sqrt_det_g;
    AT_N_sym sp_g_uu;

    AT_N_D1sca sp_dalpha_d;
    AT_N_D1vec sp_dbeta_du;
    AT_N_D1sym sp_dg_ddd;

    // (s)pace-(t)ime quantities (scratch: assembled as required)
    AT_D_vec st_n_u_;
    AT_D_vec st_n_d_;

    AT_D_vec st_beta_u_;
    AT_D_sym st_g_dd_;
    AT_D_sym st_g_uu_;

    AT_D_bil st_P_ud_;  // projector
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

    // (s)pace-(t)ime quantities (scratch: assembled as required)
    AT_D_vec st_w_u_u_;
  };
  vars_Hydro hydro;

  // various persistent scratch quantities not fitting elsewhere
  struct vars_Scratch {
    AA dflx;

    // for general quantities that need specific valence
    AT_N_sca sc_;
    AT_N_vec sp_vec_;

    AT_D_sym st_T_rad_;
  };
  vars_Scratch scratch;

  AT_N_sca m1_mask;  // Excision mask

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

  // Fluid frame radiation variables + P_{ij}, etc.
  struct ixn_Rad
  {
    enum
    {
      nnu,
      J,
      H_t,
      H_x, H_y, H_z,
      P_xx, P_xy, P_xz, P_yy, P_yz, P_zz,
      chi,
      ynu,
      znu,
      N
    };
    static constexpr char const * const names[] = {
      "rad.nnu",
      "rad.J",
      "rad.Ht", "rad.Hx", "rad.Hy", "rad.Hz",
      "rad.Pxx", "rad.Pxy", "rad.Pxz", "rad.Pyy", "rad.Pyz", "rad.Pzz",
      "rad.chi",
    };
  };

  // Radiation-matter variables
  struct ixn_RaM
  {
    enum
    {
      abs_0, abs_1,
      eta_0, eta_1,
      scat_1,
      nueave,
      N
    };
    static constexpr char const * const names[] = {
      "rmat.abs_0", "rmat.abs_1",
      "rmat.eta_0", "rmat.eta_1",
      "rmat.scat_1",
      "rmat.nueave",
    };
  };


  // Diagnostic variables
  struct ixn_Diag
  {
    enum
    {
      radflux_0,
      radflux_1,
      ynu,
      znu,
      N
    };
    static constexpr char const * const names[] = {
      "rdia.radial_flux_0",
      "rdia.radial_flux_1",
      "rdia.ynu",
      "rdia.znu",
    };
  };

  // Internal variables (no group dimension)
  struct ixn_Internal
  {
    enum
    {
      fidu_v_x, fidu_v_y, fidu_v_z,
      netabs,
      netheat,
      mask,
      N
    };
    static constexpr char const * const names[] = {
      "fidu.vx", "fidu.vy", "fidu.vz",
      "net.abs",
      "net.heat",
      "mask",
    };
  };

  // Source update results
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

// additional methods =========================================================
public:
  void UpdateGeometry(vars_Geom  & geom,
                      vars_Scratch & scratch);
  void UpdateHydro(vars_Hydro & hydro,
                   vars_Geom & geom,
                   vars_Scratch & scratch);

private:
  // These manipulate internal M1 mem-state; don't call external to class
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

  void InitializeScratch(vars_Scratch & scratch)
  {
    scratch.dflx.NewAthenaArray(N_GRPS, N_SPCS, ixn_Lab::N, mbi.nn1);
    scratch.sc_.NewAthenaTensor(mbi.nn1);
    scratch.sp_vec_.NewAthenaTensor(mbi.nn1);

    scratch.st_T_rad_.NewAthenaTensor(mbi.nn1);
  }

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
  void SetVarAliasesRad(   AA  &r,                  vars_Rad    & rad);
  void SetVarAliasesRadMat(AA  &radmat,             vars_RadMat & rmat);
  void SetVarAliasesDiagno(AA  &diagno,             vars_Diag   & rdia);
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