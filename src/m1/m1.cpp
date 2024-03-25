//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file M1.cpp
//  \brief implementation of functions in the M1 class

// c++
#include <codecvt>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Athena++ headers
#include "m1.hpp"
#include "../coordinates/coordinates.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

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
  N_GRPS(pin->GetOrAddInteger("M1", "ngroups",  1)),
  N_SPCS(pin->GetOrAddInteger("M1", "nspecies", 1)),
  storage{
    { ixn_Lab::N*N_GRPS*N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u
    { ixn_Lab::N*N_GRPS*N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u1
    { // flux
     { ixn_Lab::N*N_GRPS*N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1 + 1 },
     { ixn_Lab::N*N_GRPS*N_SPCS, mbi.nn3, mbi.nn2 + 1, mbi.nn1 },
     { ixn_Lab::N*N_GRPS*N_SPCS, mbi.nn3 + 1, mbi.nn2, mbi.nn1 },
    },
    { ixn_Lab::N*N_GRPS*N_SPCS,  mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rhs
    { ixn_Rad::N*N_GRPS*N_SPCS,  mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rad
    { ixn_RaM::N*N_GRPS*N_SPCS,  mbi.nn3, mbi.nn2, mbi.nn1 },     // radmat
    { ixn_Diag::N*N_GRPS*N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1 },     // diagno
	  { ixn_Internal::N,           mbi.nn3, mbi.nn2, mbi.nn1 },     // internal
  },
  coarse_u_(ixn_Lab::N*N_GRPS*N_SPCS, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmy_mesh->multilevel ? AA::DataStatus::allocated
                                  : AA::DataStatus::empty)),
  ubvar(pmb, &storage.u, &coarse_u_, storage.flux),
  // alias storage
  fluxes{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  lab{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  rhs{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  rad{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
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

  // initialize storage for auxiliary fields ----------------------------------
  InitializeScratch(scratch, lab, rad, geom, hydro, fidu);
  InitializeGeometry(geom, scratch);
  InitializeHydro(hydro, geom, scratch);

  // Point variables to correct locations -------------------------------------
  SetVarAliasesFluxes(storage.flux, fluxes);
  SetVarAliasesLab(storage.u,     lab);
  SetVarAliasesLab(storage.u_rhs, rhs);
  SetVarAliasesRad(storage.u_rad, rad);
  // SetVarAliasesRadMat(storage.radmat, rmat);
  // SetVarAliasesDiagno(storage.diagno, rdia);
  SetVarAliasesFidu(storage.intern, fidu);
  // SetVarAliasesNet(storage.intern, net);

  m1_mask.InitWithShallowSlice(storage.intern, ixn_Internal::mask);


  // debug
  // GroupSpeciesContainer<AT_N_sca> E(1,2);
  // lab.E(0,0).InitWithShallowSlice(storage.u,I_Lab_E);
  // lab.E(0,1).InitWithShallowSlice(storage.u,I_Lab_E+1);

  // E(0,0).array().print_all();

  // GroupSpeciesFluxContainer<AT_N_sca> fluxes_E;

  // x1dir = fluxes(ix_g,ix_s)[0](ix_lab::E,k,j,i)
  // x2dir = fluxes(ix_g,ix_s)[1](ix_lab::E,k,j,i)
  // x3dir = fluxes(ix_g,ix_s)[2](ix_lab::E,k,j,i)
  //
  // x1dir = fluxes(ix_g,ix_s,0)(ix_lab::E,k,j,i)
  // x2dir = fluxes(ix_g,ix_s,1)(ix_lab::E,k,j,i)
  // x3dir = fluxes(ix_g,ix_s,2)(ix_lab::E,k,j,i)

  /*
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    lab.E(ix_g,ix_s).Fill(ix_g+ix_s);
  }

  const int ix_g = 1;
  const int ix_s = 2;
  lab.E(ix_g,ix_s).array().print_all();

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  for (int ix_f=0; ix_f<M1_NDIM; ++ix_f)
  {
    fluxes.E(ix_g,ix_s,ix_f).array().print_all();
  }
  */

  /*
  GroupSpeciesFluxContainer<AT_N_sca> fluxes_E(1,1);

  for (int ix_f=0; ix_f<M1_NDIM; ++ix_f)
    fluxes_E(0,0,ix_f).InitWithShallowSlice(storage.flux[ix_f], I_Lab_E);

  fluxes_E(0,0,0).array().print_all();
  fluxes.E(0,0,0).array().print_all();

  fluxes_E(0,0,1).array().print_all();
  // fluxes.E(0,0,1).array().print_all();

  std::exit(0);
  */
  /*

  // const int N_gs = (ix_s + N_SPCS * (ix_g + 0)) * N_Lab;
  // fluxes_E(ix_g,ix_s,0).InitWithShallowSlice(
  //   storage.flux[0], N_gs+I_Lab_E);


  fluxes.E(ix_g,ix_s,1).array().print_all();
  */
  // lab.E(ix_g,ix_s).array().print_all();

  // const int ix_g = 1;
  // const int ix_s = 2;
  // fluxes.E(ix_g,ix_s,0).array().print_all();
  // std::exit(0);

  // debug
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  M1_ILOOP3(k,j,i)
  {
    lab.sc_E(ix_g,ix_s)(k,j,i) = mbi.x1(i) + ix_g + ix_s;
  }

  // lab.sp_F_d(0, 1).array().print_all();
  // rad.sp_P_dd(0, 1).array().print_all();

  // std::exit(0);
  // M1_GLOOP2(k,j)
  // {
  //   std::cout << k << "," << j << std::endl;
  //   Assemble::st_beta_u_(geom.st_beta_u_, geom.sp_beta_u,
  //                        k, j, 0, mbi.nn1-1);
  // }
  // std::exit(0);

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

  { // tol / ad-hoc
    opt.floor_rad_E = pin->GetOrAddReal("M1", "floor_rad_E", 1e-15);
    opt.eps_rad_E   = pin->GetOrAddReal("M1", "eps_rad_E",   1e-5);
    opt.eps_f_J     = pin->GetOrAddReal("M1", "eps_f_J",     1e-10);
    // ...
  }

}

// set aliases for variables ==================================================

void M1::SetVarAliasesFluxes(AA (&u_flux)[M1_NDIM], vars_Flux & fluxes)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  for (int ix_f=0; ix_f<M1_NDIM; ++ix_f)
  {
    SetVarAlias(fluxes.sc_E, u_flux, ix_g, ix_s, ix_f,
                ixn_Lab::E,  ixn_Lab::N);
    SetVarAlias(fluxes.sp_F_d, u_flux, ix_g, ix_s, ix_f,
                ixn_Lab::F_x,  ixn_Lab::N);
    SetVarAlias(fluxes.sc_nG, u_flux, ix_g, ix_s, ix_f,
                ixn_Lab::nG,  ixn_Lab::N);
  }
}

void M1::SetVarAliasesLab(AthenaArray<Real> & u, vars_Lab & lab)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(lab.sc_E,   u, ix_g, ix_s, ixn_Lab::E,   ixn_Lab::N);
    SetVarAlias(lab.sp_F_d, u, ix_g, ix_s, ixn_Lab::F_x, ixn_Lab::N);
    SetVarAlias(lab.sc_nG,  u, ix_g, ix_s, ixn_Lab::nG,  ixn_Lab::N);
  }
}

void M1::SetVarAliasesRad(AthenaArray<Real> & u, vars_Rad & rad)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(rad.sc_nnu,  u, ix_g, ix_s, ixn_Rad::nnu,  ixn_Rad::N);
    SetVarAlias(rad.sc_J,    u, ix_g, ix_s, ixn_Rad::J,    ixn_Rad::N);
    SetVarAlias(rad.sc_H_t,  u, ix_g, ix_s, ixn_Rad::H_t,  ixn_Rad::N);
    SetVarAlias(rad.sp_H_d,  u, ix_g, ix_s, ixn_Rad::H_x,  ixn_Rad::N);
    SetVarAlias(rad.sp_P_dd, u, ix_g, ix_s, ixn_Rad::P_xx, ixn_Rad::N);
    SetVarAlias(rad.sc_chi,  u, ix_g, ix_s, ixn_Rad::chi,  ixn_Rad::N);
    SetVarAlias(rad.sc_ynu,  u, ix_g, ix_s, ixn_Rad::ynu,  ixn_Rad::N);
    SetVarAlias(rad.sc_znu,  u, ix_g, ix_s, ixn_Rad::znu,  ixn_Rad::N);
  }
}

void M1::SetVarAliasesRadMat(AthenaArray<Real> & u, vars_RadMat & rmat)
{
  // rmat.abs_0.InitWithShallowSlice(u, I_RadMat_abs_0);
  // rmat.abs_1.InitWithShallowSlice(u, I_RadMat_abs_1);
  // rmat.eta_0.InitWithShallowSlice(u, I_RadMat_eta_0);
  // rmat.eta_1.InitWithShallowSlice(u, I_RadMat_eta_1);
  // rmat.scat_1.InitWithShallowSlice(u, I_RadMat_scat_1);
  // rmat.nueave.InitWithShallowSlice(u, I_RadMat_nueave);
}

void M1::SetVarAliasesDiagno(AthenaArray<Real> & u, vars_Diag & rdia)
{
  // rdia.radflux_0.InitWithShallowSlice(u, I_Diagno_radflux_0);
  // rdia.radflux_1.InitWithShallowSlice(u, I_Diagno_radflux_1);
  // rdia.ynu.InitWithShallowSlice(u, I_Diagno_ynu);
  // rdia.znu.InitWithShallowSlice(u, I_Diagno_znu);
}

void M1::SetVarAliasesFidu(AthenaArray<Real> & u, vars_Fidu & fidu)
{
  fidu.sp_v_u.InitWithShallowSlice(u, ixn_Internal::fidu_v_x);
  fidu.sc_W.InitWithShallowSlice(  u, ixn_Internal::fidu_W);
}

void M1::SetVarAliasesNet(AthenaArray<Real> & u, vars_Net & net)
{
  // net.abs.InitWithShallowSlice(u, I_Intern_netabs);
  // net.heat.InitWithShallowSlice(u, I_Intern_netheat);
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//