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
#include "../coordinates/coordinates.hpp"

// ----------------------------------------------------------------------------
M1::M1(MeshBlock *pmb, ParameterInput *pin) :
  pm1(this),
  pmy_block(pmb),
  pmy_mesh(pmb->pmy_mesh),
  pmy_coord(pmy_block->pcoord),
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
  NSPCS(pin->GetOrAddInteger("M1", "nspecies", 1)),
  NGRPS(pin->GetOrAddInteger("M1", "ngroups",  1)),
  storage{
    { N_Lab*NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},  // u
    { N_Lab*NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},  // u1
    { // flux
     {N_Lab*NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1 + 1},
     {N_Lab*NSPCS*NGRPS, mbi.nn3, mbi.nn2 + 1, mbi.nn1,
	    (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated
                         : AthenaArray<Real>::DataStatus::empty)},
     {N_Lab*NSPCS*NGRPS, mbi.nn3 + 1, mbi.nn2, mbi.nn1,
	    (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated
                         : AthenaArray<Real>::DataStatus::empty)},
    },
    {N_Lab*NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},     // u_rhs
    {N_Rad,NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},     // u_rad
    {N_RadMat,NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},  // radmat
    {N_Diagno,NSPCS*NGRPS, mbi.nn3, mbi.nn2, mbi.nn1},  // diagno
	  {N_Intern, mbi.nn3, mbi.nn2, pmb->ncells1},         // internal
  }
{
  Coordinates * pco = pmb->pcoord;

  //---------------------------------------------------------------------------
  // set up sampling
  //---------------------------------------------------------------------------
  mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, mbi.nn1);
  mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, mbi.nn2);
  mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, mbi.nn3);

  // sizes are the same in either case
  mbi.dx1.InitWithShallowSlice(pco->dx1v, 1, 0, mbi.nn1);
  mbi.dx2.InitWithShallowSlice(pco->dx2v, 1, 0, mbi.nn2);
  mbi.dx3.InitWithShallowSlice(pco->dx3v, 1, 0, mbi.nn3);
  //---------------------------------------------------------------------------

  // should all be of size mbi.nn1
  dt1_.NewAthenaArray(mbi.nn1);
  dt2_.NewAthenaArray(mbi.nn1);
  dt3_.NewAthenaArray(mbi.nn1);

  // Populate options
  PopulateOptions(pin);


  // Point variables to correct locations -------------------------------------
  SetLabVarsAliases(storage.u, lab);
  SetRadVarsAliases(storage.u_rad, rad);
  SetRadMatVarsAliases(storage.radmat, rmat);
  SetDiagnoVarsAliases(storage.diagno, rdia);
  SetFiduVarsAliases(storage.intern, fidu);
  SetNetVarsAliases(storage.intern, net);

  m1_mask.InitWithShallowSlice(storage.intern, I_Intern_mask);
}

M1::~M1()
{

}

void M1::PopulateOptions(ParameterInput *pin)
{
  std::string tmp;
  std::ostringstream msg;

  { // fiducial
    tmp = pin->GetOrAddString("M1", "fiducial_velocity", "fluid");
    if (tmp == "fluid")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::fluid;
    }
    else if (tmp == "mixed")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::mixed;
    }
    else if (tmp == "zero")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::zero;
    }
    else if (tmp == "none")
    {
      opt.fiducial_velocity = opt_fiducial_velocity::none;
    }
    else
    {
      msg << "M1/fiducial_velocity unknown" << std::endl;
      ATHENA_ERROR(msg);
    }

    opt.fiducial_velocity_rho_fluid = pin->GetOrAddReal(
      "M1", "fiducial_velocity_rho_fluid", 0.0) * CGS_GCC;
  }

}

// set aliases for variables ==================================================

void M1::SetLabVarsAliases(AthenaArray<Real> & u, M1::vars_Lab & lab)
{
  lab.E.InitWithShallowSliceVar(  u, I_Lab_E,  NGRPS*NSPCS);
  lab.F_d.InitWithShallowSliceVar(u, I_Lab_Fx, NGRPS*NSPCS);
  lab.N.InitWithShallowSliceVar(  u, I_Lab_N,  NGRPS*NSPCS);
}

void M1::SetRadVarsAliases(AthenaArray<Real> & u, M1::vars_Rad & rad)
{
  rad.nnu.InitWithShallowSlice(u, I_Rad_nnu);
  rad.J.InitWithShallowSlice(u, I_Rad_J);
  rad.Ht.InitWithShallowSlice(u, I_Rad_Ht);
  rad.H.InitWithShallowSlice(u, I_Rad_Hx);
  rad.P_dd.InitWithShallowSlice(u, I_Rad_Pxx);
  rad.chi.InitWithShallowSlice(u, I_Rad_chi);
  rad.ynu.InitWithShallowSlice(u, I_Rad_ynu);
  rad.znu.InitWithShallowSlice(u, I_Rad_znu);
}

void M1::SetRadMatVarsAliases(AthenaArray<Real> & u, M1::vars_RadMat & rmat)
{
  rmat.abs_0.InitWithShallowSlice(u, I_RadMat_abs_0);
  rmat.abs_1.InitWithShallowSlice(u, I_RadMat_abs_1);
  rmat.eta_0.InitWithShallowSlice(u, I_RadMat_eta_0);
  rmat.eta_1.InitWithShallowSlice(u, I_RadMat_eta_1);
  rmat.scat_1.InitWithShallowSlice(u, I_RadMat_scat_1);
  rmat.nueave.InitWithShallowSlice(u, I_RadMat_nueave);
}

void M1::SetDiagnoVarsAliases(AthenaArray<Real> & u, M1::vars_Diagno & rdia)
{
  rdia.radflux_0.InitWithShallowSlice(u, I_Diagno_radflux_0);
  rdia.radflux_1.InitWithShallowSlice(u, I_Diagno_radflux_1);
  rdia.ynu.InitWithShallowSlice(u, I_Diagno_ynu);
  rdia.znu.InitWithShallowSlice(u, I_Diagno_znu);
}

void M1::SetFiduVarsAliases(AthenaArray<Real> & u, M1::vars_Fidu & fidu)
{
  fidu.vel_u.InitWithShallowSlice(u, I_Intern_fidu_vx);
  fidu.Wlorentz.InitWithShallowSlice(u, I_Intern_fidu_Wlorentz);
}

void M1::SetNetVarsAliases(AthenaArray<Real> & u, M1::vars_Net & net)
{
  net.abs.InitWithShallowSlice(u, I_Intern_netabs);
  net.heat.InitWithShallowSlice(u, I_Intern_netheat);
}
