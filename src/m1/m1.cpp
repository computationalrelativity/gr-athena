//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file M1.cpp
//  \brief implementation of functions in the M1 class

// c++
#include <codecvt>
#include <iomanip>
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
  N_GS(N_GRPS*N_SPCS),
  storage{
    { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u
    { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u1
    { // flux
     { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 + 1 },
     { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2 + 1, mbi.nn1 },
     { ixn_Lab::N*N_GS, mbi.nn3 + 1, mbi.nn2, mbi.nn1 },
    },
    { ixn_Lab::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rhs
    { ixn_Lab_aux::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 },     // u_lab_aux
    { ixn_Rad::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rad
    { ixn_RaM::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // radmat
    { ixn_Diag::N*N_GS,    mbi.nn3, mbi.nn2, mbi.nn1 },     // diagno
	  { ixn_Internal::N,     mbi.nn3, mbi.nn2, mbi.nn1 },     // internal
  },
  coarse_u_(ixn_Lab::N*N_GS, mbi.cnn3, mbi.cnn2, mbi.cnn1,
            (pmy_mesh->multilevel ? AA::DataStatus::allocated
                                  : AA::DataStatus::empty)),
  ubvar(pmb, &storage.u, &coarse_u_, storage.flux),
  // alias storage (size specs. must match number of containers in struct)
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
  lab_aux{
    {N_GRPS,N_SPCS},
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
    {N_GRPS,N_SPCS}
  },
  radmat{
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
  SetVarAliasesFluxes(storage.flux,      fluxes);
  SetVarAliasesLab(   storage.u,         lab);
  SetVarAliasesLab(   storage.u_rhs,     rhs);
  SetVarAliasesLabAux(storage.u_lab_aux, lab_aux);
  SetVarAliasesRad(   storage.u_rad,     rad);

  SetVarAliasesRadMat(storage.radmat, radmat);
  // SetVarAliasesDiagno(storage.diagno, rdia);
  SetVarAliasesFidu(storage.intern, fidu);
  // SetVarAliasesNet(storage.intern, net);

  m1_mask.InitWithShallowSlice(storage.intern, ixn_Internal::mask);
  m1_mask.Fill(true);

  // --------------------------------------------------------------------------
  // general setup
  if ((opt.closure_variety == opt_closure_variety::MinerboP) ||
      (opt.closure_variety == opt_closure_variety::MinerboN))
  {
    for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
    for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
    {
      lab_aux.sc_xi( ix_g,ix_s).Fill(1.0);  // init. with thin limit
      lab_aux.sc_chi(ix_g,ix_s).Fill(1.0);
    }
  }

}

void M1::PopulateOptions(ParameterInput *pin)
{
  std::string tmp;
  std::ostringstream msg;

  { // integration strategy
    tmp = pin->GetOrAddString("M1", "integration_strategy", "full_explicit");
    if (tmp == "full_explicit")
    {
      opt.integration_strategy = opt_integration_strategy::full_explicit;
    }
    else if (tmp == "semi_implicit_PicardFrozenP")
    {
      opt.integration_strategy = \
        opt_integration_strategy::semi_implicit_PicardFrozenP;
    }
    else if (tmp == "semi_implicit_PicardMinerboP")
    {
      opt.integration_strategy = \
        opt_integration_strategy::semi_implicit_PicardMinerboP;
    }
    else if (tmp == "semi_implicit_PicardMinerboPC")
    {
      opt.integration_strategy = \
        opt_integration_strategy::semi_implicit_PicardMinerboPC;
    }
    else
    {
      msg << "M1/integration_strategy unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  { // fluxes
    tmp = pin->GetOrAddString("M1", "characteristics_variety", "approximate");
    if (tmp == "approximate")
    {
      opt.characteristics_variety = opt_characteristics_variety::approximate;
    }
    else if (tmp == "exact_thin")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thin;
    }
    else if (tmp == "exact_thick")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_thick;
    }
    else if (tmp == "exact_Minerbo")
    {
      opt.characteristics_variety = opt_characteristics_variety::exact_Minerbo;
    }
    else
    {
      msg << "M1/characteristics_variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

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
      "M1", "fiducial_velocity_rho_fluid", 0.0) * M1_UNITS_CGS_GCC;
  }

  { // closure
    tmp = pin->GetOrAddString("M1", "closure_variety", "thin");
    if (tmp == "thin")
    {
      opt.closure_variety = opt_closure_variety::thin;
    }
    else if (tmp == "thick")
    {
      opt.closure_variety = opt_closure_variety::thick;
    }
    else if (tmp == "Minerbo")
    {
      opt.closure_variety = opt_closure_variety::Minerbo;
    }
    else if (tmp == "MinerboP")
    {
      opt.closure_variety = opt_closure_variety::MinerboP;
    }
    else if (tmp == "MinerboN")
    {
      opt.closure_variety = opt_closure_variety::MinerboN;
    }
    else
    {
      msg << "M1/closure_variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  { // opacities
    tmp = pin->GetOrAddString("M1_opacities", "variety", "none");
    if (tmp == "none")
    {
      opt.opacity_variety = opt_opacity_variety::none;
    }
    else if (tmp == "zero")
    {
      opt.opacity_variety = opt_opacity_variety::zero;
    }
    else
    {
      msg << "M1_opacities/variety unknown" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  { // tol / ad-hoc
    opt.fl_E = pin->GetOrAddReal("M1", "fl_E", 1e-15);
    opt.fl_J = pin->GetOrAddReal("M1", "fl_J", 1e-15);
    opt.eps_E = pin->GetOrAddReal("M1", "eps_E", 1e-5);
    opt.eps_J = pin->GetOrAddReal("M1", "eps_J", 1e-10);
    opt.min_flux_A = pin->GetOrAddReal("M1", "min_flux_A", 0);

    opt.eps_C      = pin->GetOrAddReal(   "M1", "eps_C",      1e-6);
    opt.eps_C_N    = pin->GetOrAddReal(   "M1", "eps_C_N",    1e-10);
    opt.max_iter_C = pin->GetOrAddInteger("M1", "max_iter_C", 64);
    opt.max_iter_C_rst = pin->GetOrAddInteger("M1", "max_iter_C_rst", 10);
    opt.w_opt_ini_C = pin->GetOrAddReal("M1", "w_opt_ini_C", 0.5);
    opt.reset_thin = pin->GetOrAddBoolean("M1", "reset_thin", true);
    opt.fac_amp_C = pin->GetOrAddReal("M1", "fac_amp_C", 1.11);
    opt.verbose_iter_C = pin->GetOrAddBoolean("M1", "verbose_iter_C", false);
  }

  { // semi-implicit iteration settings
    opt.max_iter_P = pin->GetOrAddInteger("M1", "max_iter_P", 128);
    opt.max_iter_P_rst = pin->GetOrAddInteger("M1", "max_iter_P_rst", 10);
    opt.w_opt_ini = pin->GetOrAddReal("M1", "w_opt_ini", 1.0);
    opt.eps_P_abs_tol = pin->GetOrAddReal("M1", "eps_P_abs_tol", 1e-8);
    opt.fac_amp_P = pin->GetOrAddReal("M1", "fac_amp_P", 1.11);
    opt.verbose_iter_P = pin->GetOrAddBoolean("M1", "verbose_iter_P", false);
  }

  // debugging
  opt.value_inject = pin->GetOrAddBoolean("problem", "value_inject", false);
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

void M1::SetVarAliasesLabAux(AthenaArray<Real> & u, vars_LabAux & lab_aux)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(lab_aux.sp_P_dd, u, ix_g, ix_s,
                ixn_Lab_aux::P_xx, ixn_Lab_aux::N);
    SetVarAlias(lab_aux.sc_n,    u, ix_g, ix_s,
                ixn_Lab_aux::n,    ixn_Lab_aux::N);
    SetVarAlias(lab_aux.sc_chi,  u, ix_g, ix_s,
                ixn_Lab_aux::chi,  ixn_Lab_aux::N);
    SetVarAlias(lab_aux.sc_xi,   u, ix_g, ix_s,
                ixn_Lab_aux::xi,   ixn_Lab_aux::N);
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
    SetVarAlias(rad.sc_ynu,  u, ix_g, ix_s, ixn_Rad::ynu,  ixn_Rad::N);
    SetVarAlias(rad.sc_znu,  u, ix_g, ix_s, ixn_Rad::znu,  ixn_Rad::N);
  }
}

void M1::SetVarAliasesRadMat(AthenaArray<Real> & u, vars_RadMat & rmat)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(radmat.sc_eta_0,   u, ix_g, ix_s,
                ixn_RaM::eta_0,   ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_a_0, u, ix_g, ix_s,
                ixn_RaM::kap_a_0, ixn_RaM::N);

    SetVarAlias(radmat.sc_eta,   u, ix_g, ix_s,
                ixn_RaM::eta,   ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_a, u, ix_g, ix_s,
                ixn_RaM::kap_a, ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_s, u, ix_g, ix_s,
                ixn_RaM::kap_s, ixn_RaM::N);

  }
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
  fidu.sp_v_u.InitWithShallowSlice(u, ixn_Internal::fidu_v_u_x);
  fidu.sp_v_d.InitWithShallowSlice(u, ixn_Internal::fidu_v_d_x);
  fidu.st_v_u.InitWithShallowSlice(u, ixn_Internal::fidu_st_v_t);
  fidu.sc_W.InitWithShallowSlice(  u, ixn_Internal::fidu_W);
}

void M1::SetVarAliasesNet(AthenaArray<Real> & u, vars_Net & net)
{
  // net.abs.InitWithShallowSlice(u, I_Intern_netabs);
  // net.heat.InitWithShallowSlice(u, I_Intern_netheat);
}


void M1::StatePrintPoint(
  const int ix_g, const int ix_s,
  const int k, const int j, const int i,
  const bool terminate)
{
  #pragma omp critical
  {
    std::cout << "M1::DebugState" << std::endl;
    std::cout << std::setprecision(14) << std::endl;
    std::cout << "ix_g, ix_s:  " << ix_g << ", " << ix_s << "\n";
    std::cout << "k, j, i:     " << k << ", " << j << ", " << i << "\n";

    std::cout << "mbi.x3(k):   " << mbi.x3(k) << "\n";
    std::cout << "mbi.x2(j):   " << mbi.x2(j) << "\n";
    std::cout << "mbi.x1(i):   " << mbi.x1(i) << "\n";


    std::cout << "geometric fields=========================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    geom.sc_alpha.PrintPoint("geom.sc_alpha", k,j,i);
    geom.sc_sqrt_det_g.PrintPoint("geom.sc_sqrt_det_g", k,j,i);

    std::cout << "vec================: " << "\n";
    geom.sp_beta_u.PrintPoint("geom.sp_beta_u", k,j,i);
    geom.sp_beta_d.PrintPoint("geom.sp_beta_d", k,j,i);

    std::cout << "sym2===============: " << "\n";
    geom.sp_g_dd.PrintPoint("geom.sp_g_dd", k,j,i);
    geom.sp_K_dd.PrintPoint("geom.sp_K_dd", k,j,i);
    geom.sp_g_uu.PrintPoint("geom.sp_g_uu", k,j,i);

    std::cout << "dsc================: " << "\n";
    geom.sp_dalpha_d.PrintPoint("geom.sp_dalpha_d", k,j,i);


    std::cout << "dvec===============: " << "\n";
    geom.sp_dbeta_du.PrintPoint("geom.sp_dbeta_du", k,j,i);

    std::cout << "dsym2==============: " << "\n";
    geom.sp_dg_ddd.PrintPoint("geom.sp_dg_ddd", k,j,i);

    std::cout << "fiducial fields=========================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    fidu.sc_W.PrintPoint("fidu.sc_W", k,j,i);

    std::cout << "vec================: " << "\n";
    fidu.sp_v_u.PrintPoint("fidu.sp_v_u", k,j,i);
    fidu.sp_v_d.PrintPoint("fidu.sp_v_d", k,j,i);

    std::cout << "radiation fields========================: " << "\n\n";

    std::cout << "sc=================: " << "\n";
    lab.sc_nG(ix_g,ix_s).PrintPoint("lab.sc_nG(ix_s,ix_g)", k,j,i);
    lab.sc_E(ix_g,ix_s).PrintPoint("lab.sc_E(ix_s,ix_g)", k,j,i);

    std::cout << "vec================: " << "\n";
    lab.sp_F_d(ix_g,ix_s).PrintPoint("lab.sp_F_d(ix_s,ix_g)", k,j,i);

    std::cout << "sc=================: " << "\n";
    lab_aux.sc_n(ix_g,ix_s).PrintPoint("lab_aux.sc_n(ix_s,ix_g)", k,j,i);
    lab_aux.sc_chi(ix_g,ix_s).PrintPoint("lab_aux.sc_chi(ix_s,ix_g)", k,j,i);
    lab_aux.sc_xi(ix_g,ix_s).PrintPoint("lab_aux.sc_xi(ix_s,ix_g)", k,j,i);

    std::cout << "sym2===============: " << "\n";
    lab_aux.sp_P_dd(ix_g,ix_s).PrintPoint("lab_aux.sp_P_dd(ix_s,ix_g)", k,j,i);


  }

  if (terminate)
  {
    std::exit(0);
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//