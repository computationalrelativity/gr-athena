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
#include <vector>

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
  N_SPCS(pin->GetOrAddInteger("M1", "nspecies", 1)),
  N_GRPS(pin->GetOrAddInteger("M1", "ngroups",  1)),
  storage{
    { N_Lab*N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u
    { N_Lab*N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u1
    { // flux
     { N_Lab*N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 + 1 },
     { N_Lab*N_SPCS*N_GRPS, mbi.nn3, mbi.nn2 + 1, mbi.nn1,
	    (pmy_mesh->f2 ? AA::DataStatus::allocated
                    : AA::DataStatus::empty) },
     { N_Lab*N_SPCS*N_GRPS, mbi.nn3 + 1, mbi.nn2, mbi.nn1,
	    (pmy_mesh->f3 ? AA::DataStatus::allocated
                    : AA::DataStatus::empty) },
    },
    { N_Lab*N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rhs
    { N_Rad,N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rad
    { N_RadMat,N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },  // radmat
    { N_Diagno,N_SPCS*N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1 },  // diagno
	  { N_Intern, mbi.nn3, mbi.nn2, mbi.nn1 },         // internal
  },
  coarse_u_(N_Lab*N_SPCS*N_GRPS, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmy_mesh->multilevel ? AA::DataStatus::allocated
                                  : AA::DataStatus::empty)),
  empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
  ubvar(pmb, &storage.u, &coarse_u_, storage.flux),
  // alias storage
  lab{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  rhs{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  }
{

  // Registration of boundary variables for refinement etc. -------------------
  pmb->RegisterMeshBlockDataCC(storage.u);
  if (pmy_mesh->multilevel)
  {
    pmb->pmr->AddToRefinementM1CC(&storage.u, &coarse_u_);
  }

  ubvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ubvar);
  pmb->pbval->bvars_m1.push_back(&ubvar);
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // set up sampling
  // --------------------------------------------------------------------------
  Coordinates * pco = pmy_coord;

  mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, mbi.nn1);
  mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, mbi.nn2);
  mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, mbi.nn3);

  // sizes are the same in either case
  mbi.dx1.InitWithShallowSlice(pco->dx1v, 1, 0, mbi.nn1);
  mbi.dx2.InitWithShallowSlice(pco->dx2v, 1, 0, mbi.nn2);
  mbi.dx3.InitWithShallowSlice(pco->dx3v, 1, 0, mbi.nn3);
  // --------------------------------------------------------------------------

  // should all be of size mbi.nn1
  dt1_.NewAthenaArray(mbi.nn1);
  dt2_.NewAthenaArray(mbi.nn1);
  dt3_.NewAthenaArray(mbi.nn1);

  // Populate options
  PopulateOptions(pin);

  // Point variables to correct locations -------------------------------------
  SetVarAliasesLab(storage.u, lab);
  SetVarAliasesRad(storage.u_rad, rad);
  SetVarAliasesRadMat(storage.radmat, rmat);
  SetVarAliasesDiagno(storage.diagno, rdia);
  SetVarAliasesFidu(storage.intern, fidu);
  SetVarAliasesNet(storage.intern, net);

  m1_mask.InitWithShallowSlice(storage.intern, I_Intern_mask);

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

void M1::SetVarAliasesLab(AthenaArray<Real> & u, vars_Lab & lab)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    const int N_gs = (ix_s + N_SPCS * (ix_g + 0)) * N_Lab;
    lab.E(  ix_g,ix_s).InitWithShallowSlice(u, N_gs+I_Lab_E);
    lab.F_d(ix_g,ix_s).InitWithShallowSlice(u, N_gs+I_Lab_Fx);
    lab.N(  ix_g,ix_s).InitWithShallowSlice(u, N_gs+I_Lab_N);
  }
}

void M1::SetVarAliasesRad(AthenaArray<Real> & u, vars_Rad & rad)
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

void M1::SetVarAliasesRadMat(AthenaArray<Real> & u, vars_RadMat & rmat)
{
  rmat.abs_0.InitWithShallowSlice(u, I_RadMat_abs_0);
  rmat.abs_1.InitWithShallowSlice(u, I_RadMat_abs_1);
  rmat.eta_0.InitWithShallowSlice(u, I_RadMat_eta_0);
  rmat.eta_1.InitWithShallowSlice(u, I_RadMat_eta_1);
  rmat.scat_1.InitWithShallowSlice(u, I_RadMat_scat_1);
  rmat.nueave.InitWithShallowSlice(u, I_RadMat_nueave);
}

void M1::SetVarAliasesDiagno(AthenaArray<Real> & u, vars_Diagno & rdia)
{
  rdia.radflux_0.InitWithShallowSlice(u, I_Diagno_radflux_0);
  rdia.radflux_1.InitWithShallowSlice(u, I_Diagno_radflux_1);
  rdia.ynu.InitWithShallowSlice(u, I_Diagno_ynu);
  rdia.znu.InitWithShallowSlice(u, I_Diagno_znu);
}

void M1::SetVarAliasesFidu(AthenaArray<Real> & u, vars_Fidu & fidu)
{
  fidu.vel_u.InitWithShallowSlice(u, I_Intern_fidu_vx);
  fidu.Wlorentz.InitWithShallowSlice(u, I_Intern_fidu_Wlorentz);
}

void M1::SetVarAliasesNet(AthenaArray<Real> & u, vars_Net & net)
{
  net.abs.InitWithShallowSlice(u, I_Intern_netabs);
  net.heat.InitWithShallowSlice(u, I_Intern_netheat);
}
