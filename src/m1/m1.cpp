// See m1.hpp for description / references.

// c++
#include <codecvt>
#include <gsl/gsl_roots.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

// Athena++ headers
#include "m1.hpp"
#include "../coordinates/coordinates.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "opacities/m1_opacities.hpp"

// ============================================================================
namespace M1 {
// ============================================================================

// ----------------------------------------------------------------------------
M1::M1(MeshBlock *pmb, ParameterInput *pin) :
  pm1(this),
  pmy_mesh(pmb->pmy_mesh),
  pmy_block(pmb),
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
    pmb->cis,                       // cil
    pmb->cie,                       // ciu
    pmb->cjs,                       // cjl
    pmb->cje,                       // cju
    pmb->cks,                       // ckl
    pmb->cke,                       // cku
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
    // { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 },  // u2
    {   // flux
     { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 + 1 },
     { ixn_Lab::N*N_GS, mbi.nn3, mbi.nn2 + 1, mbi.nn1 },
     { ixn_Lab::N*N_GS, mbi.nn3 + 1, mbi.nn2, mbi.nn1 },
    },
    {}, // flux_lo
    { ixn_Lab::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rhs
    { ixn_Lab_aux::N*N_GS, mbi.nn3, mbi.nn2, mbi.nn1 },     // u_lab_aux
    { ixn_Rad::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // u_rad
    { ixn_Src::N*N_GS,     mbi.nn3, mbi.nn2, mbi.nn1 },     // u_sources
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
  fluxes_lo{
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
  lab_aux{
    // {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  rad{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  radmat{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  eql{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  sources{
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS},
    {N_GRPS,N_SPCS}
  },
  rdiag{
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
    // For internal points (LoadBalancingAndAdaptiveMeshRefinement)
    pmb->pmr->AddToRefinementCC(&storage.u, &coarse_u_);
    // For inter-MeshBlock boundaries
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
  SetVarAliasesSource(storage.u_sources, sources);
  SetVarAliasesLabAux(storage.u_lab_aux, lab_aux);
  SetVarAliasesRad(   storage.u_rad,     rad);

  SetVarAliasesRadMat(storage.radmat, radmat);
  SetVarAliasesDiag(  storage.diagno, rdiag);
  SetVarAliasesFidu(  storage.intern, fidu);
  SetVarAliasesNet(   storage.intern, net);

  if (opt.retain_equilibrium)
  {
    storage_eql.NewAthenaArray(ixn_Eql::N*N_GS,
                               mbi.nn3, mbi.nn2, mbi.nn1);
    SetVarAliasesEql(storage_eql, eql);
  }

  // storage for misc. quantities ---------------------------------------------
  if ((opt_solver.src_lim >= 0) ||
      (opt_solver.full_lim >= 0))
  {
    sources.theta.NewAthenaTensor(mbi.nn3, mbi.nn2, mbi.nn1);
  }

  if (opt_solver.solver_reduce_to_common)
  {
    ev_strat.masks.solution_regime.NewAthenaArray(
      N_GRPS, mbi.nn3, mbi.nn2, mbi.nn1);
  }
  else
  {
    ev_strat.masks.solution_regime.NewAthenaArray(
      N_GRPS, N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1);
  }
  ev_strat.masks.solution_regime.Fill(t_sln_r::noop);

  ev_strat.masks.excised.NewAthenaArray(
    mbi.nn3, mbi.nn2, mbi.nn1);
  ev_strat.masks.excised.Fill(false);

  ev_strat.status.clear();


  if (opt.flux_limiter_use_mask)
  {
    ev_strat.masks.flux_limiter.NewAthenaArray(
      N, mbi.nn3, mbi.nn2, mbi.nn1);
    ev_strat.masks.flux_limiter.Fill(0);
  }

  if (opt.flux_lo_fallback)
  {
    storage.flux_lo[0].NewAthenaArray(ixn_Lab::N*N_GS,
                                      mbi.nn3, mbi.nn2, mbi.nn1 + 1);
    storage.flux_lo[1].NewAthenaArray(ixn_Lab::N*N_GS,
                                      mbi.nn3, mbi.nn2 + 1, mbi.nn1);
    storage.flux_lo[2].NewAthenaArray(ixn_Lab::N*N_GS,
                                      mbi.nn3 + 1, mbi.nn2, mbi.nn1);

    SetVarAliasesFluxes(storage.flux_lo, fluxes_lo);

    if (opt.flux_lo_fallback_species)
    {
      ev_strat.masks.pp.NewAthenaArray(N_SPCS, mbi.nn3, mbi.nn2, mbi.nn1);
      ev_strat.masks.compute_point.NewAthenaArray(N_SPCS,
                                                  mbi.nn3, mbi.nn2, mbi.nn1);

    }
    else
    {
      ev_strat.masks.pp.NewAthenaArray(mbi.nn3, mbi.nn2, mbi.nn1);
      ev_strat.masks.compute_point.NewAthenaArray(mbi.nn3, mbi.nn2, mbi.nn1);
    }
  }

  ev_strat.masks.compute_point.Fill(true);

  // --------------------------------------------------------------------------
  // general setup
  /*
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

  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    lab.sc_E(ix_g,ix_s).Fill(opt.fl_E);
    lab.sp_F_d(ix_g,ix_s).Fill(0);
    lab_aux.sp_P_dd(ix_g,ix_s).Fill(std::numeric_limits<Real>::infinity());
  }
  */

  // --------------------------------------------------------------------------
  // (GSL) solver setup
  gsl_set_error_handler_off();

  gsl_brent_solver = gsl_root_fsolver_alloc(
    gsl_root_fsolver_brent
  );

  gsl_newton_solver = gsl_root_fdfsolver_alloc(
    gsl_root_fdfsolver_newton
  );

  // --------------------------------------------------------------------------
  // deal with opacities
  popac = new Opacities::Opacities(pmb, this, pin);
}

M1::~M1()
{
  gsl_root_fsolver_free(gsl_brent_solver);
  gsl_root_fdfsolver_free(gsl_newton_solver);
  // TODO: bug - this can't be deactivated properly with OMP
  // gsl_set_error_handler(NULL);  // restore default handler

  delete popac;
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

void M1::SetVarAliasesSource(AthenaArray<Real> & sources, vars_Source & src)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(src.sc_nG,  sources, ix_g, ix_s, ixn_Src::sc_nG,  ixn_Src::N);
    SetVarAlias(src.sc_E,   sources, ix_g, ix_s, ixn_Src::sc_E,   ixn_Src::N);
    SetVarAlias(src.sp_F_d, sources, ix_g, ix_s, ixn_Src::sp_F_0, ixn_Src::N);
  }
}

void M1::SetVarAliasesLabAux(AthenaArray<Real> & u, vars_LabAux & lab_aux)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    // SetVarAlias(lab_aux.sp_P_dd, u, ix_g, ix_s,
    //             ixn_Lab_aux::P_xx, ixn_Lab_aux::N);
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
    SetVarAlias(rad.sc_n,     u, ix_g, ix_s, ixn_Rad::n,     ixn_Rad::N);
    SetVarAlias(rad.sc_J,     u, ix_g, ix_s, ixn_Rad::J,     ixn_Rad::N);
    SetVarAlias(rad.st_H_u,   u, ix_g, ix_s, ixn_Rad::st_H_u_t, ixn_Rad::N);
  }
}

void M1::SetVarAliasesRadMat(AthenaArray<Real> & u, vars_RadMat & radmat)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(radmat.sc_eta_0,   u, ix_g, ix_s,
                ixn_RaM::eta_0,    ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_a_0, u, ix_g, ix_s,
                ixn_RaM::kap_a_0,  ixn_RaM::N);

    SetVarAlias(radmat.sc_eta,   u, ix_g, ix_s,
                ixn_RaM::eta,    ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_a, u, ix_g, ix_s,
                ixn_RaM::kap_a,  ixn_RaM::N);
    SetVarAlias(radmat.sc_kap_s, u, ix_g, ix_s,
                ixn_RaM::kap_s,  ixn_RaM::N);

    SetVarAlias(radmat.sc_avg_nrg, u, ix_g, ix_s,
                ixn_RaM::avg_nrg,  ixn_RaM::N);
  }
}

void M1::SetVarAliasesEql(AthenaArray<Real> & u, vars_Eql & eq)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(eql.sc_J, u, ix_g, ix_s, ixn_Eql::sc_J, ixn_Eql::N);
    SetVarAlias(eql.sc_n, u, ix_g, ix_s, ixn_Eql::sc_n, ixn_Eql::N);
  }
}

void M1::SetVarAliasesDiag(AthenaArray<Real> & u, vars_Diag & rdiag)
{
  for (int ix_g=0; ix_g<N_GRPS; ++ix_g)
  for (int ix_s=0; ix_s<N_SPCS; ++ix_s)
  {
    SetVarAlias(rdiag.sc_radflux_0,  u, ix_g, ix_s,
                ixn_Diag::radflux_0, ixn_Diag::N);
    SetVarAlias(rdiag.sc_radflux_1,  u, ix_g, ix_s,
                ixn_Diag::radflux_1, ixn_Diag::N);

    SetVarAlias(rdiag.sc_y, u, ix_g, ix_s,
                ixn_Diag::y, ixn_Diag::N);
    SetVarAlias(rdiag.sc_z, u, ix_g, ix_s,
                ixn_Diag::z, ixn_Diag::N);
  }
}

void M1::SetVarAliasesFidu(AthenaArray<Real> & u, vars_Fidu & fidu)
{
  fidu.sp_v_u.InitWithShallowSlice(u, ixn_Internal::fidu_v_u_x);
  fidu.sp_v_d.InitWithShallowSlice(u, ixn_Internal::fidu_v_d_x);
  fidu.sc_W.InitWithShallowSlice(  u, ixn_Internal::fidu_W);
}

void M1::SetVarAliasesNet(AthenaArray<Real> & u, vars_Net & net)
{
  net.abs.InitWithShallowSlice( u, ixn_Internal::netabs);
  net.heat.InitWithShallowSlice(u, ixn_Internal::netheat);
}

void M1::StatePrintPoint(
  const std::string & tag,
  const int ix_g, const int ix_s,
  const int k, const int j, const int i,
  const bool terminate)
{
  #pragma omp critical
  {
    std::cout << "M1::DebugState" << std::endl;
    std::cout << "Tag: \n";
    std::cout << tag << "\n";
    std::cout << std::setprecision(14) << std::endl;
    std::cout << "ix_g, ix_s:  " << ix_g << ", " << ix_s << "\n";
    std::cout << "k, j, i:     " << k << ", " << j << ", " << i << "\n";

    std::cout << "mbi.x3(k):   " << mbi.x3(k) << "\n";
    std::cout << "mbi.x2(j):   " << mbi.x2(j) << "\n";
    std::cout << "mbi.x1(i):   " << mbi.x1(i) << "\n";


    std::cout << "geometric fields=========================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    geom.sc_oo_sqrt_det_g.PrintPoint("geom.sc_oo_sqrt_det_g", k,j,i);
    geom.sc_sqrt_det_g.PrintPoint("geom.sc_sqrt_det_g", k,j,i);
    geom.sc_alpha.PrintPoint("geom.sc_alpha", k,j,i);

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

    std::cout << "hydro fields============================: " << "\n\n";
    std::cout << "sc=================: " << "\n";
    hydro.sc_W.PrintPoint("hydro.sc_W", k,j,i);
    hydro.sc_T.PrintPoint("hydro.sc_T", k,j,i);
    hydro.sc_w_p.PrintPoint("hydro.sc_w_p", k,j,i);
    hydro.sc_w_rho.PrintPoint("hydro.sc_w_rho", k,j,i);
    hydro.sc_w_Ye.PrintPoint("hydro.sc_w_Ye", k,j,i);

    std::cout << "vec================: " << "\n";
    hydro.sp_w_util_u.PrintPoint("hydro.sp_w_util_u", k,j,i);

    std::cout << "radiation fields========================: " << "\n\n";

    std::cout << "sc=================: " << "\n";
    lab.sc_nG(ix_g,ix_s).PrintPoint("lab.sc_nG(ix_g,ix_s)", k,j,i);
    lab.sc_E(ix_g,ix_s).PrintPoint("lab.sc_E(ix_g,ix_s)", k,j,i);

    std::cout << "vec================: " << "\n";
    lab.sp_F_d(ix_g,ix_s).PrintPoint("lab.sp_F_d(ix_g,ix_s)", k,j,i);

    std::cout << "sc=================: " << "\n";
    rad.sc_n(ix_g,ix_s).PrintPoint("rad.sc_n(ix_g,ix_s)", k,j,i);
    rad.sc_J(ix_g,ix_s).PrintPoint("rad.sc_J(ix_g,ix_s)", k,j,i);
    lab_aux.sc_chi(ix_g,ix_s).PrintPoint("lab_aux.sc_chi(ix_g,ix_s)", k,j,i);
    lab_aux.sc_xi(ix_g,ix_s).PrintPoint("lab_aux.sc_xi(ix_g,ix_s)", k,j,i);

    // std::cout << "sym2===============: " << "\n";
    // lab_aux.sp_P_dd(ix_g,ix_s).PrintPoint("lab_aux.sp_P_dd(ix_g,ix_s)", k,j,i);

    std::cout << "src=================: " << "\n";
    sources.sc_nG(ix_g,ix_s).PrintPoint("sources.sc_nG", k, j, i);
    sources.sc_E(ix_g,ix_s).PrintPoint("sources.sc_E", k, j, i);
    sources.sp_F_d(ix_g,ix_s).PrintPoint("sources.sp_F_d", k, j, i);
    radmat.sc_eta(ix_g,ix_s).PrintPoint("radmat.sc_eta", k, j, i);
    radmat.sc_kap_a(ix_g,ix_s).PrintPoint("radmat.sc_kap_a", k, j, i);
    radmat.sc_kap_s(ix_g,ix_s).PrintPoint("radmat.sc_kap_s", k, j, i);

    const Real kap_as = (
      radmat.sc_kap_a(ix_g,ix_s)(k,j,i) +
      radmat.sc_kap_s(ix_g,ix_s)(k,j,i)
    );

    std::printf("OO(kap_as * dx1) = %.3g\n", OO(kap_as) * mbi.dx1(i));
    std::printf("OO(kap_as * dx2) = %.3g\n", OO(kap_as) * mbi.dx2(j));
    std::printf("OO(kap_as * dx3) = %.3g\n", OO(kap_as) * mbi.dx3(k));

    std::cout << "opt_solution_regime: ";
    std::cout << static_cast<int>(GetMaskSolutionRegime(ix_g,ix_s,k,j,i));
    std::cout << "\n";

    if (opt.flux_lo_fallback)
    {
      std::cout << "opt_flux_lo_fallback: ";
      std::cout << sources.theta(k,j,i);
      std::cout << "\n";
    }
  }

  if (terminate)
  {
    assert(false);
  }
}

// ============================================================================
} // namespace M1
// ============================================================================

//
// :D
//
