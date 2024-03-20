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
#include "m1_containers.hpp"

// M1 settings ----------------------------------------------------------------


// Class ======================================================================

class MeshBlock;
class ParameterInput;

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

  Real NewBlockTimeStep(void);

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
  const int N_SPCS { 1 };
  const int N_GRPS { 1 };

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

  // variable alias -----------------------------------------------------------

  // Lab variables and RHS
  struct vars_Lab {
    GroupSpeciesContainer<AT_N_sca> E;
    GroupSpeciesContainer<AT_N_vec> F_d;
    GroupSpeciesContainer<AT_N_sca> N;
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

  AT_N_sca m1_mask;  // Excision mask

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
  enum
  {
    I_Lab_E,
    I_Lab_Fx, I_Lab_Fy, I_Lab_Fz,
    I_Lab_N,
    N_Lab
  };
  static constexpr char const * const names_Lab[] = {
    "lab.E",
    "lab.Fx", "lab.Fy", "lab.Fz",
    "lab.N"
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


// additional methods =========================================================
public:
  // aliases ------------------------------------------------------------------
  void SetVarAliasesLab(   AA & u,      vars_Lab    & lab);
  void SetVarAliasesRad(   AA & r,      vars_Rad    & rad);
  void SetVarAliasesRadMat(AA & radmat, vars_RadMat & rmat);
  void SetVarAliasesDiagno(AA & diagno, vars_Diagno & rdia);
  void SetVarAliasesFidu(  AA & intern, vars_Fidu   & fid);
  void SetVarAliasesNet(   AA & intern, vars_Net    & net);


// internal methods ===========================================================
private:
  void PopulateOptions(ParameterInput *pin);

// internal data ==============================================================
private:
  AA dt1_, dt2_, dt3_;  // scratch arrays used in NewTimeStep


};

#endif // M1_HPP
