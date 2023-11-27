//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file M1.cpp
//  \brief implementation of functions in the M1 class

#include <iostream>
#include <fstream>

// Athena++ headers
#include "m1.hpp"
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/interp_intergrid.hpp"

#if Z4C_ENABLED
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#endif

// constructor, initializes data structures and parameters

char const * const M1::Lab_names[M1::N_Lab] = {
  "lab.N",
  "lab.E",
  "lab.Fx", "lab.Fy", "lab.Fz", 
};

char const * const M1::Rad_names[M1::N_Rad] = {
  "rad.nnu",
  "rad.J",
  "rad.Ht", "rad.Hx", "rad.Hy", "rad.Hz",
  "rad.Pxx", "rad.Pxy", "rad.Pxz", "rad.Pyy", "rad.Pyz", "rad.Pzz",
  "rad.chi",
  "rad.mask",
};

char const * const M1::RadMat_names[M1::N_RadMat] = {
  "rmat.abs_0", "rmat.abs_1",
  "rmat.eta_0", "rmat.eta_1",
  "rmat.scat_1",
};

char const * const M1::Diagno_names[M1::N_Diagno] = {
  "rdia.radial_flux_0",
  "rdia.radial_flux_1",
  "rdia.ynu",
  "rdia.znu",
};

char const * const M1::Intern_names[M1::N_Intern] = {
  "fidu.vx", "fidu.vy", "fidu.vz",
  "fidu.Wlorents", 
  "net.abs",
  "net.heat",
};

char const * const M1::source_update_msg[M1::M1_SRC_UPDATE_RESULTS] = {
  "Ok",
  "explicit update (thin source)",
  "imposed equilibrium",
  "(scattering dominated source)",
  "imposed eddington",
  "failed"
};

M1::M1(MeshBlock *pmb, ParameterInput *pin) :
  pmy_block(pmb),
  coarse_u_(N_Lab, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
	     AthenaArray<Real>::DataStatus::empty)),
  storage{{N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1},  // u
          {N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1},  // u1
          {                                                                           // flux
            {N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1 + 1},
            {N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2 + 1, pmb->ncells1,
	     (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)},
            {N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3 + 1, pmb->ncells2, pmb->ncells1,
	     (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
              AthenaArray<Real>::DataStatus::empty)},
          },
          {N_Lab, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // u_rhs
          {N_Rad, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // u_rad
          {N_RadMat, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // radmat
          {N_Diagno, M1_NSPECIES*M1_NGROUPS, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // diagno
	  {N_Intern, pmb->ncells3, pmb->ncells2, pmb->ncells1}, // internal
  },
  ubvar(pmb, &storage.u, &coarse_u_, storage.flux)
{
  Mesh *pm = pmy_block->pmy_mesh;
  Coordinates * pco = pmb->pcoord;
  
  mbi.nn1 = pmb->ncells1;
  mbi.nn2 = pmb->ncells2;
  mbi.nn3 = pmb->ncells3;
  
  int nn1 = mbi.nn1, nn2 = mbi.nn2, nn3 = mbi.nn3;
  
  // convenience for per-block iteration (private Wave scope)
  mbi.il = pmb->is; mbi.jl = pmb->js; mbi.kl = pmb->ks;
  mbi.iu = pmb->ie; mbi.ju = pmb->je; mbi.ku = pmb->ke;

  // point to appropriate grid
  mbi.x1.InitWithShallowSlice(pco->x1v, 1, 0, nn1);
  mbi.x2.InitWithShallowSlice(pco->x2v, 1, 0, nn2);
  mbi.x3.InitWithShallowSlice(pco->x3v, 1, 0, nn3);

  //---------------------------------------------------------------------------

  // inform MeshBlock that this array is the "primary" representation
  // Used for:
  // (1) load-balancing
  // (2) (future) dumping to restart file
  pmb->RegisterMeshBlockDataCC(storage.u);

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementVC(&storage.u, &coarse_u_);
  }

  //TODO:
  // If user-requested time integrator is type 3S* allocate additional memory
  // std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  // if (integrator == "ssprk5_4")
  //   storage.u2.NewAthenaArray(N_Z4c, nn3, nn2, nn1);

  // enroll CellCenteredBoundaryVariabl object
  ubvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&ubvar);
  pmb->pbval->bvars_m1_int.push_back(&ubvar);

  dt1_.NewAthenaArray(nn1);
  dt2_.NewAthenaArray(nn1);
  dt3_.NewAthenaArray(nn1);
  x1face_area_.NewAthenaArray(mbi.nn1+1);
  if (pm->f2) {
    x2face_area_.NewAthenaArray(mbi.nn1);
    x2face_area_p1_.NewAthenaArray(mbi.nn1);
  }
  if (pm->f3) {
    x3face_area_.NewAthenaArray(mbi.nn1);
    x3face_area_p1_.NewAthenaArray(mbi.nn1);
  }
  cell_volume_.NewAthenaArray(mbi.nn1);
  dflx_.NewAthenaArray(N_Lab, ngroups*nspecies, mbi.nn1);

  //---------------------------------------------------------------------------
  
  // Parameters
  nspecies = M1_NSPECIES;
  ngroups = M1_NGROUPS;

  closure = pin->GetString("M1", "closure");  
  fiducial_velocity = pin->GetOrAddString("M1", "fiducial_velocity", "");
  fiducial_vel_rho_fluid = pin->GetOrAddReal("M1", "fiducial_velocity_rho_fluid", 0.0) * CGS_GCC;
  opacity_equil_depth = pin->GetOrAddReal("M1", "opacity_equil_depth", 1.0);
  opacity_corr_fac_max = pin->GetOrAddReal("M1", "opacity_corr_fac_max", 10.);
  rad_E_floor = pin->GetOrAddReal("M1", "rad_E_floor", 1e-15);
  rad_N_floor = pin->GetOrAddReal("M1", "rad_N_floor", 1e-15);
  source_limiter = pin->GetOrAddReal("M1", "source_limiter", 0.5);
  backreact = pin->GetOrAddBoolean("M1", "backreact",true); 
  rad_eps = pin->GetOrAddReal("M1", "rad_eps", 1e-5);
  set_to_equilibrium = pin->GetOrAddBoolean("M1", "set_to_equilibrium",false); 
  reset_to_equilibrium = pin->GetOrAddBoolean("M1", "set_to_equilibrium",false); 
  equilibrium_rho_min =  pin->GetOrAddReal("M1", "fiducial_velocity_rho_fluid",1e11) * CGS_GCC;
  source_thick_limit = pin->GetOrAddReal("M1","source_thick_limit", -1);
  source_scat_limit = pin->GetOrAddReal("M1","source_scat_limit", -1);
  source_epsabs = pin->GetOrAddReal("M1","source_epsabs",1e-3);
  source_epsrel = pin->GetOrAddReal("M1","source_epsrel",1e-3);
  source_maxiter = pin->GetOrAddInteger("M1","source_maxiter", 100);

  // Problem-specific parameters
  m1_test = pin->GetOrAddString("M1", "m1_test","none");  
  beam_position = pin->GetOrAddReal("M1", "beam_position", 0.0);
  beam_width = pin->GetOrAddReal("M1", "beam_width", 1.0);
  equil_nudens_0[0] = pin->GetOrAddReal("M1", "equil_nudens_0_0", 0.0);
  equil_nudens_0[1] = pin->GetOrAddReal("M1", "equil_nudens_0_1", 0.0);
  equil_nudens_0[2] = pin->GetOrAddReal("M1", "equil_nudens_0_2", 0.0);
  equil_nudens_1[0] = pin->GetOrAddReal("M1", "equil_nudens_1_0", 0.0);
  equil_nudens_1[1] = pin->GetOrAddReal("M1", "equil_nudens_1_1", 0.0);
  equil_nudens_1[2] = pin->GetOrAddReal("M1", "equil_nudens_1_2", 0.0);
  diff_profile = pin->GetOrAddString("M1", "diffusion_profile","step");  
  kerr_beam_position = pin->GetOrAddReal("M1", "kerr_beam_position", 3.25);
  kerr_beam_width = pin->GetOrAddReal("M1", "kerr_beam_width", 0.5);
  kerr_mask_radius = pin->GetOrAddReal("M1", "kerr_mask_radius", 2.0);
  
  //---------------------------------------------------------------------------

  AverageBaryonMass = nullptr;

  fakerates = nullptr;
  if (pin->GetOrAddBoolean("FakeRates","use",false)) {
    fakerates = new FakeRates(pin, ngroups, nspecies);
    //TODO set AverageBaryonMass, etc
  }
  
  // Set aliases
  SetLabVarsAliases(storage.u, lab);
  SetRadVarsAliases(storage.u_rad, rad);
  SetRadMatVarsAliases(storage.radmat, rmat);
  SetDiagnoVarsAliases(storage.diagno, rdia);
  SetFiduVarsAliases(storage.intern, fidu);
  SetNetVarsAliases(storage.intern, net);
  
  // Allocate memory for auxiliary vars
  //...
  
  //Intergrid communication
  int N[] = {pmb->block_size.nx1, pmb->block_size.nx2, pmb->block_size.nx3};
  Real rdx[] = {
    1./pmb->pcoord->dx1f(0), 1./pmb->pcoord->dx2f(0), 1./pmb->pcoord->dx3f(0)
  };

  // TODO: CHECK THIS
  if(pmb->pmy_mesh->multilevel){
    int N_coarse[] = {pmb->block_size.nx1/2, pmb->block_size.nx2/2, pmb->block_size.nx3/2};
    Real rdx_coarse[] = {
      1./pmb->pmr->pcoarsec->dx1f(0), 1./pmb->pmr->pcoarsec->dx2f(0), 1./pmb->pmr->pcoarsec->dx3f(0)
    };
    
  }
  
  //---------------------------------------------------------------------------
  
  // Initialize excision mask (no excision)
  rad.mask.ZeroClear();
  
}

// destructor
M1::~M1()
{
  storage.u.DeleteAthenaArray();
  storage.u1.DeleteAthenaArray();
  storage.u_rhs.DeleteAthenaArray();
  storage.u_rad.DeleteAthenaArray();
  storage.radmat.DeleteAthenaArray();
  storage.diagno.DeleteAthenaArray();
  storage.intern.DeleteAthenaArray();

  dt1_.DeleteAthenaArray();
  dt2_.DeleteAthenaArray();
  dt3_.DeleteAthenaArray();

  if (fakerates != nullptr)
    delete fakerates;
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetLabVarsAliases(AthenaArray<Real> & u, Lab_vars & lab)
// \brief Set Lab radiation variables aliases

void M1::SetLabVarsAliases(AthenaArray<Real> & u, M1::Lab_vars & lab)
{
  lab.N.InitWithShallowSlice(u, I_Lab_N);
  lab.E.InitWithShallowSlice(u, I_Lab_E);
  lab.F_d.InitWithShallowSlice(u, I_Lab_Fx);
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetRadVarsAliases(AthenaArray<Real> & u, Rad_vars & rad)
// \brief Set fluid frame radiation variables et al aliases

void M1::SetRadVarsAliases(AthenaArray<Real> & u, M1::Rad_vars & rad)
{
  rad.nnu.InitWithShallowSlice(u, I_Rad_nnu);
  rad.J.InitWithShallowSlice(u, I_Rad_J);
  rad.Ht.InitWithShallowSlice(u, I_Rad_Ht);
  rad.H.InitWithShallowSlice(u, I_Rad_Hx);
  rad.P_dd.InitWithShallowSlice(u, I_Rad_Pxx);
  rad.chi.InitWithShallowSlice(u, I_Rad_chi);
  rad.mask.InitWithShallowSlice(u, I_Rad_mask);
  rad.ynu.InitWithShallowSlice(u, I_Rad_ynu);
  rad.znu.InitWithShallowSlice(u, I_Rad_znu);  
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetRadMatVarsAliases(AthenaArray<Real> & u, RadMat_vars & rmat)
// \brief Set radiation-matter variables aliases

void M1::SetRadMatVarsAliases(AthenaArray<Real> & u, M1::RadMat_vars & rmat)
{
  rmat.abs_0.InitWithShallowSlice(u, I_RadMat_abs_0);
  rmat.abs_1.InitWithShallowSlice(u, I_RadMat_abs_1);
  rmat.eta_0.InitWithShallowSlice(u, I_RadMat_eta_0);
  rmat.eta_1.InitWithShallowSlice(u, I_RadMat_eta_1);
  rmat.scat_1.InitWithShallowSlice(u, I_RadMat_scat_1);
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetDiagnoVarsAliases(AthenaArray<Real> & u, Diagno_vars & dia)
// \brief Set analyses variables aliases

void M1::SetDiagnoVarsAliases(AthenaArray<Real> & u, M1::Diagno_vars & rdia)
{
  rdia.radflux_0.InitWithShallowSlice(u, I_Diagno_radflux_0);
  rdia.radflux_1.InitWithShallowSlice(u, I_Diagno_radflux_1); 
  rdia.ynu.InitWithShallowSlice(u, I_Diagno_ynu);
  rdia.znu.InitWithShallowSlice(u, I_Diagno_znu);
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetFiduVarsAliases(AthenaArray<Real> & u, Fidu_vars & fidu)
// \brief Set fiducial velocity variables aliases

void M1::SetFiduVarsAliases(AthenaArray<Real> & u, M1::Fidu_vars & fidu)
{
  fidu.vel_u.InitWithShallowSlice(u, I_Intern_fidu_vx);
  fidu.Wlorentz.InitWithShallowSlice(u, I_Intern_fidu_Wlorentz);
}

//----------------------------------------------------------------------------------------
// \!fn void M1::SetNetVarsAliases(AthenaArray<Real> & u, Net_vars & net)
// \brief Set net abs/heat variables aliases

void M1::SetNetVarsAliases(AthenaArray<Real> & u, M1::Net_vars & net)
{
  net.abs.InitWithShallowSlice(u, I_Intern_netabs);
  net.heat.InitWithShallowSlice(u, I_Intern_netheat);
}
