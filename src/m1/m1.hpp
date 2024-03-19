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

// M1 settings ----------------------------------------------------------------
#ifndef M1_NSPECIES
#define M1_NSPECIES (1)
#endif

#ifndef M1_NGROUPS
#define M1_NGROUPS (1)
#endif

// Class ----------------------------------------------------------------------

class MeshBlock;
class ParameterInput;

class M1
{

// methods --------------------------------------------------------------------
public:
  M1(MeshBlock *pmb, ParameterInput *pin);
  ~M1();

  // new time-step for current block
  Real NewBlockTimeStep(void);

// data -----------------------------------------------------------------------
public:
  // Mesh->MeshBlock->M1
  Mesh *pmy_mesh;
  MeshBlock *pmy_block;
  // M1-indicial information
  MB_info mbi;

  struct
  {
    AthenaArray<Real> u;       // solution of M1 evolution system
    AthenaArray<Real> u1;      // solution at intermediate steps
    AthenaArray<Real> flux[3]; // flux in the 3 directions
    AthenaArray<Real> u_rhs;   // M1 rhs
    AthenaArray<Real> u_rad;   // fluid frame variables + P_{ij} Lab
    AthenaArray<Real> radmat;  // radiation-matter fields
    AthenaArray<Real> diagno;  // analysis buffers
    // "internals": fiducial velocity, netabs, ..
    // N.B. these do not have group dimension!
    AthenaArray<Real> intern;
  } storage;

  struct
  {
    // options will go here ...
  } opt;

// idx & constants ------------------------------------------------------------
public:
  // Lab frame variables
  enum
  {
    I_Lab_E,
    I_Lab_Fx, I_Lab_Fy, I_Lab_Fz,
    I_Lab_N,
    N_Lab
  };
  static char const * const Lab_names[N_Lab];

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
  static char const * const Rad_names[N_Rad];

  // Radiation-matter variables
  enum
  {
    I_RadMat_abs_0, I_RadMat_abs_1,
    I_RadMat_eta_0, I_RadMat_eta_1,
    I_RadMat_scat_1,
    I_RadMat_nueave,
    N_RadMat
  };
  static char const * const RadMat_names[N_RadMat];

  // Diagnostic variables
  enum
  {
    I_Diagno_radflux_0,
    I_Diagno_radflux_1,
    I_Diagno_ynu,
    I_Diagno_znu,
    N_Diagno
  };
  static char const * const Diagno_names[N_Diagno];

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
  static char const * const Intern_names[N_Intern];

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
  static char const * const source_update_msg[M1_SRC_UPDATE_RESULTS];


// internal methods -----------------------------------------------------------
private:
  // ...

// internal data --------------------------------------------------------------
private:
  AthenaArray<Real> dt1_,dt2_,dt3_;  // scratch arrays used in NewTimeStep


};

#endif // M1_HPP
