//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file M1.cpp
//  \brief implementation of functions in the M1 class

// c++
#include <iostream>
#include <fstream>
#include <sstream>

// Athena++ headers
#include "m1.hpp"

// populate static constants --------------------------------------------------
char const * const M1::Lab_names[M1::N_Lab] = {
  "lab.E",
  "lab.Fx", "lab.Fy", "lab.Fz",
  "lab.N"
};

char const * const M1::Rad_names[M1::N_Rad] = {
  "rad.nnu",
  "rad.J",
  "rad.Ht", "rad.Hx", "rad.Hy", "rad.Hz",
  "rad.Pxx", "rad.Pxy", "rad.Pxz", "rad.Pyy", "rad.Pyz", "rad.Pzz",
  "rad.chi",
};

char const * const M1::RadMat_names[M1::N_RadMat] = {
  "rmat.abs_0", "rmat.abs_1",
  "rmat.eta_0", "rmat.eta_1",
  "rmat.scat_1",
  "rmat.nueave",
};

char const * const M1::Diagno_names[M1::N_Diagno] = {
  "rdia.radial_flux_0",
  "rdia.radial_flux_1",
  "rdia.ynu",
  "rdia.znu",
};

char const * const M1::Intern_names[M1::N_Intern] = {
  "fidu.vx", "fidu.vy", "fidu.vz",
  "fidu.Wlorentz",
  "net.abs",
  "net.heat",
  "mask",
};

char const * const M1::source_update_msg[M1::M1_SRC_UPDATE_RESULTS] = {
  "Ok",
  "explicit update (thin source)",
  "imposed equilibrium",
  "(scattering dominated source)",
  "imposed eddington",
  "failed",
};
// ----------------------------------------------------------------------------

M1::M1(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb),
  pmy_mesh(pmb->pmy_mesh),
  mbi{
    1, pmy_mesh->f2, pmy_mesh->f3,  // f1, f2, f3
    pmb->is,                        // il
    pmb->ie,                        // iu
    pmb->js,                        // jl
    pmb->je,                        // ju
    pmb->ks,                        // kl
    pmb->ke,                        // ku
    pmb->ncells1,                   // nn1
    pmb->ncells2,                   // nn2
    pmb->ncells3,                   // nn3
    pmb->ncc1,                      // cnn1
    pmb->ncc2,                      // cnn2
    pmb->ncc3,                      // cnn3
    NGHOST,                         // ng
    NCGHOST,                        // cng
    (pmy_mesh->f3) ? 3 : (pmy_mesh->f2) ? 2 : 1    // ndim
  },
  storage{
    { N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},  // u
    { N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},  // u1
    { // flux
     {N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1 + 1},
     {N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2 + 1, mbi.nn1,
	    (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated
                         : AthenaArray<Real>::DataStatus::empty)},
     {N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3 + 1, mbi.nn2, mbi.nn1,
	    (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated
                         : AthenaArray<Real>::DataStatus::empty)},
    },
    {N_Lab*M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},     // u_rhs
    {N_Rad,M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},     // u_rad
    {N_RadMat,M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},  // radmat
    {N_Diagno,M1_NSPECIES*M1_NGROUPS, mbi.nn3, mbi.nn2, mbi.nn1},  // diagno
	  {N_Intern, mbi.nn3, mbi.nn2, pmb->ncells1},                    // internal
  }
{

  // should all be of size mbi.nn1
  dt1_.NewAthenaArray(mbi.nn1);
  dt2_.NewAthenaArray(mbi.nn1);
  dt3_.NewAthenaArray(mbi.nn1);

}

M1::~M1()
{

}