//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

// C headers

// C++ headers
#include <algorithm>
#include <string>
#include <vector>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"
#include "srcterms/hydro_srcterms.hpp"
#include "../utils/linear_algebra.hpp"


// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb),
    u(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    u1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    w1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    derived_ms(NDRV_HYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    derived_int(NIDRV_HYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    // C++11: nested brace-init-list in Hydro member initializer list = aggregate init. of
    // flux[3] array --> direct list init. of each array element --> direct init. via
    // constructor overload resolution of non-aggregate class type AthenaArray<Real>
    flux{ {NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NHYDRO, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {NHYDRO, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}
    },
#if EFL_ENABLED // EFL
    flux_LO{ {NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NHYDRO, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {NHYDRO, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}},
    flux_HO{ {NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NHYDRO, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {NHYDRO, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}},
    ef_limiter{ {pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
           (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
          {pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
           (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}},
#endif // End EFL
    coarse_cons_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    coarse_prim_(NHYDRO, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                 (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                  AthenaArray<Real>::DataStatus::empty)),
    hbvar(pmb, &u, &coarse_cons_, flux, HydroBoundaryQuantity::cons),
    hsrc(this, pin),
    hdif(this, pin)
{
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;

  pmb->RegisterMeshBlockDataCC(u);

  floor_both_states = pin->GetOrAddBoolean("time", "floor_both_states", false);
  flux_reconstruction = pin->GetOrAddBoolean(
    "hydro", "flux_reconstruction", false);
  split_lr_fallback = pin->GetOrAddBoolean(
    "hydro", "split_lr_fallback", false);

  flux_table_limiter = pin->GetOrAddBoolean(
    "hydro", "flux_table_limiter", false);

  opt_excision.alpha_threshold =
      pin->GetOrAddReal("excision", "alpha_threshold", -1.0);
  opt_excision.horizon_based =
      pin->GetOrAddBoolean("excision", "horizon_based", false);
  opt_excision.horizon_factor =
      pin->GetOrAddReal("excision", "horizon_factor", 1.0);

  opt_excision.hybrid_hydro =
      pin->GetOrAddBoolean("excision", "hybrid_hydro", false);

  opt_excision.hybrid_fac_min_alpha =
      pin->GetOrAddReal("excision", "hybrid_fac_min_alpha", 1.5);

  opt_excision.use_taper =
      pin->GetOrAddBoolean("excision", "use_taper", false);

  opt_excision.excise_hydro_damping =
      pin->GetOrAddBoolean("excision", "excise_hydro_damping", false);

  opt_excision.hydro_damping_factor =
      pin->GetOrAddReal("excision", "hydro_damping_factor", 0.69);

  opt_excision.excise_flux =
      pin->GetOrAddBoolean("excision", "excise_flux", true);

  opt_excision.excise_c2p =
      pin->GetOrAddBoolean("excision", "excise_c2p", true);

  if (opt_excision.use_taper || opt_excision.excise_hydro_damping)
  {
    excision_mask.NewAthenaArray(nc3,nc2,nc1);
    excision_mask.Fill(1);
  }

  opt_excision.taper_pow =
      pin->GetOrAddReal("excision", "taper_pow", 1.0);

  opt_excision.excise_hydro_freeze_evo =
      pin->GetOrAddBoolean("excision", "excise_hydro_freeze_evo", false);

  opt_excision.excise_hydro_taper =
      pin->GetOrAddBoolean("excision", "excise_hydro_taper", false);


  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED) {
    // future extension may add "int nregister" to Hydro class
    u2.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
  }

  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinementCC(&u, &coarse_cons_);
  }

  // enroll HydroBoundaryVariable object
  hbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&hbvar);
  pmb->pbval->bvars_main_int.push_back(&hbvar);

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  dxw_.NewAthenaArray(nc1);
#if EFL_ENABLED
  dyw_.NewAthenaArray(nc1);
  dzw_.NewAthenaArray(nc1);
#endif
  wl_.NewAthenaArray(NWAVE, nc1);
  wr_.NewAthenaArray(NWAVE, nc1);
  wlb_.NewAthenaArray(NWAVE, nc1);

  if (pmy_block->precon->xorder_use_auxiliaries)
  {
    al_.NewAthenaArray(NDRV_HYDRO, nc1);
    ar_.NewAthenaArray(NDRV_HYDRO, nc1);
    alb_.NewAthenaArray(NDRV_HYDRO, nc1);
  }

  if (pmy_block->precon->xorder_use_fb)
  {
    r_wl_.NewAthenaArray(NWAVE, nc1);
    r_wr_.NewAthenaArray(NWAVE, nc1);
    r_wlb_.NewAthenaArray(NWAVE, nc1);

    if (pmy_block->precon->xorder_use_auxiliaries)
    {
      r_al_.NewAthenaArray(NDRV_HYDRO, nc1);
      r_ar_.NewAthenaArray(NDRV_HYDRO, nc1);
      r_alb_.NewAthenaArray(NDRV_HYDRO, nc1);
    }

    if (pmy_block->precon->xorder_use_fb_mask)
    {
      mask_l_.NewAthenaArray(nc1);
      mask_lb_.NewAthenaArray(nc1);
      mask_r_.NewAthenaArray(nc1);
    }
  }
  else
  {
#if USETM
    // Needed for PrimitiveSolver floors
    rl_.NewAthenaArray(NSCALARS, nc1);
    rr_.NewAthenaArray(NSCALARS, nc1);
    rlb_.NewAthenaArray(NSCALARS, nc1);
#endif
  }

  dflx_.NewAthenaArray(NHYDRO, nc1);

  UserTimeStep_ = pmb->pmy_mesh->UserTimeStep_;

// EFL ARRAYS
#if EFL_ENABLED
    entropy_R.NewAthenaArray(nc3 , nc2, nc1);
    ent.NewAthenaArray(nc3 , nc2, nc1);
    ent1.NewAthenaArray(nc3 , nc2, nc1);
    ent2.NewAthenaArray(nc3 , nc2, nc1);
    ent3.NewAthenaArray(nc3 , nc2, nc1);
#endif// EFL ARRAYS

  // scratches for rsolver ----------------------------------------------------
#if Z4C_ENABLED
  const int nn1 = pmy_block->nverts1;

  sqrt_detgamma_.NewAthenaTensor(nn1);
  detgamma_.NewAthenaTensor(     nn1);
  oo_detgamma_.NewAthenaTensor(  nn1);

  alpha_.NewAthenaTensor(   nn1);
  oo_alpha_.NewAthenaTensor(nn1);
  beta_u_.NewAthenaTensor(  nn1);
  gamma_dd_.NewAthenaTensor(nn1);
  gamma_uu_.NewAthenaTensor(nn1);

  chi_.NewAthenaTensor(nn1);

  w_v_u_l_.NewAthenaTensor(nn1);
  w_v_u_r_.NewAthenaTensor(nn1);

  w_norm2_v_l_.NewAthenaTensor(nn1);
  w_norm2_v_r_.NewAthenaTensor(nn1);

  lambda_p_l.NewAthenaTensor(nn1);
  lambda_m_l.NewAthenaTensor(nn1);
  lambda_p_r.NewAthenaTensor(nn1);
  lambda_m_r.NewAthenaTensor(nn1);
  lambda.NewAthenaTensor(nn1);

  w_util_d_l_.NewAthenaTensor(nn1);
  w_util_d_r_.NewAthenaTensor(nn1);

  W_l_.NewAthenaTensor(nn1);
  W_r_.NewAthenaTensor(nn1);

  w_hrho_l_.NewAthenaTensor(nn1);
  w_hrho_r_.NewAthenaTensor(nn1);

  cons_l_.NewAthenaTensor(nn1);
  cons_r_.NewAthenaTensor(nn1);

  flux_l_.NewAthenaTensor(nn1);
  flux_r_.NewAthenaTensor(nn1);

#if MAGNETIC_FIELDS_ENABLED
  oo_sqrt_detgamma_.NewAthenaTensor(nn1);

  oo_W_l_.NewAthenaTensor(nn1);
  oo_W_r_.NewAthenaTensor(nn1);

  w_v_d_l_.NewAthenaTensor(nn1);
  w_v_d_r_.NewAthenaTensor(nn1);

  alpha_w_vtil_u_l_.NewAthenaTensor(nn1);
  alpha_w_vtil_u_r_.NewAthenaTensor(nn1);

  beta_d_.NewAthenaTensor(nn1);

  q_scB_u_l_.NewAthenaTensor(nn1);
  q_scB_u_r_.NewAthenaTensor(nn1);

  b0_l_.NewAthenaTensor(nn1);
  b0_r_.NewAthenaTensor(nn1);

  b2_l_.NewAthenaTensor(nn1);
  b2_r_.NewAthenaTensor(nn1);

  bi_u_l_.NewAthenaTensor(nn1);
  bi_u_r_.NewAthenaTensor(nn1);

  bi_d_l_.NewAthenaTensor(nn1);
  bi_d_r_.NewAthenaTensor(nn1);
#endif // MAGNETIC_FIELDS_ENABLED

#endif

}

//----------------------------------------------------------------------------------------
//! \fn Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt)
//  \brief Calculate the weighting factor for the constrained transport method

Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt) {
  Real v_over_c = (1024.0)* dt * dflx / (dx * (rhol + rhor));
  Real tmp_min = std::min(static_cast<Real>(0.5), v_over_c);
  return 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
}


//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateEntropy(w,entropy)
//  \brief Calculate the Entropy per cell in a meshblock

void Hydro::CalculateEntropy(AthenaArray<Real> &w, AthenaArray<Real> &entropy)
{
  MeshBlock *pmb = pmy_block;
  int il, iu, jl, ju, kl, ku;
  // setting loop limits
  il = 0  ;
  iu = pmb->ncells1;
  // 2D
  jl = 0  ; 
  ju = pmb->ncells2 ;
  // 3D
  kl = 0  ; 
  ku = pmb->ncells3 ;
  
  Real g=pmb->peos->GetGamma();
  Real g_ = g - 1.;

#if USETM
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
  Real Y[MAX_SPECIES] = {0.0};
#endif

  for (int k=kl; k<ku ;++k){
    for(int j=jl ; j<ju ; ++j){
      for (int i=il; i<iu ; ++i){
#if USETM
        for(int n=0; n<NSCALARS;n++){
          Y[n] = pmb->pscalars->r(n,k,j,i);
        }
        const Real n = w(IDN,k,j,i)/mb ;
        const Real T = pmb->peos->GetEOS().GetTemperatureFromP(
          n,w(IPR,k,j,i),Y);
        //Real epsl = pmb->peos->GetEOS().GetSpecificInternalEnergy(n,T,Y);
        //Real p = pmb->peos->GetEOS().GetPressure(n,T,Y);
        //entropy(k,j,i)=std::log( w(IPR,k,j,i)/std::pow(w(IDN,k,j,i),g) );
        Real epsl=w(IPR,k,j,i)/(g_*w(IDN,k,j,i)) ;
        entropy(k,j,i) = std::log(std::abs(epsl/std::pow(w(IDN,k,j,i),g_)) );
#else
        entropy(k,j,i)=std::log( w(IPR,k,j,i)/std::pow(w(IDN,k,j,i),g) );
#endif
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateEFL(w,ent,ent1,ent2,ent3 )
//  \brief Calculate the Entropy Flux Limiter in §-dimensions

void Hydro::CalculateEFL(AthenaArray<Real> &w, AthenaArray<Real> &ent,
  AthenaArray<Real> &ent1, AthenaArray<Real> &ent2,
  AthenaArray<Real> &ent3 ){
  using namespace gra::aliases;
  using namespace LinearAlgebra;
  MeshBlock *pmb = pmy_block; // pointer to MeshBlock containg this Calculate EFL
  Mesh * pm = pmb->pmy_mesh;
  Z4c * pz4c = pmy_block->pz4c;
  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);
  AT_N_vec w_util_u(w, IVX); // fluid 3-velocity(contavariant)

  Real lim;
  Real dt =pmb->pmy_mesh->dt;// time step
  Real tc1=11.0/6.0, tc2=3.0,tc3=3.0/2.0, tc4=1.0/3.0;
  Real cons = 1.0;
  Real sc1=3.0/4.0, sc2=3.0/20.0, sc3=1.0/60.0;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;

  il=is-1;
  iu=ie+1;
  jl = js - (pm->f2 || pm->f3); 
  ju = je + (pm->f2 || pm->f3);
  kl = ks - (pm->f3);
  ku = ke + (pm->f3);

  int ndim=1 + pm->f2 + pm->f3;
  
  // First Calculate Entropy residual
  for (int k=kl ; k<=ku ;++k){
    for (int j=jl ; j<= ju ; ++j){
      pmb->pcoord->CenterWidth1(k, j, il, iu, dxw_);
      if(pmb->pmy_mesh->f2){
        pmb->pcoord->CenterWidth2(k, j, il, iu, dyw_);
      }
      if(pmb->pmy_mesh->f3){
        pmb->pcoord->CenterWidth3(k, j, il, iu, dzw_);
      }
      AT_N_sca W_(iu+1);
      AT_N_vec w_v_u(iu+1);
      // Lorentz factors
      for (int i=il; i<=iu; ++i)
      {
        W_(i) = std::sqrt(1. + InnerProductVecMetric(
          w_util_u, gamma_dd, k,j,i));
      }
      // Eulerian velocity centred contravariant componenets
      for (int a=0; a<ndim; ++a)
      {
        #pragma omp simd
        for (int i=il; i<= iu; ++i)
        {
          w_v_u(a,i) = w_util_u(a,k,j,i) / W_(i);
        }
      }

      for (int i=il; i<=iu; ++i){
        Real R; // entropy residual
        //3d 
        if (pmb->pmy_mesh->f3 && pmb->pmy_mesh->f2){
          R=( tc1*ent(k,j,i) -tc2*ent1(k,j,i) + tc3*ent2(k,j,i) - tc4*ent3(k,j,i ) )/dt 
          + (alpha(k,j,i)*w_v_u(IVX-1,i)-beta_u(IVX-1,k,j,i))*( (sc1*( ent(k,j,i+1) -ent(k,j,i-1) ) - sc2*( ent(k,j,i+2)-ent(k,j,i-2) ) + sc3*( ent(k,j,i+3)-ent(k,j,i-3) ) ) / dxw_(i) )
          + (alpha(k,j,i)*w_v_u(IVY-1,i)-beta_u(IVY-1,k,j,i))*( (sc1*( ent(k,j+1,i) -ent(k,j-1,i) ) - sc2*( ent(k,j+2,i)-ent(k,j-2,i) ) + sc3*( ent(k,j+3,i)-ent(k,j-3,i) ) ) / dyw_(i) )
          + (alpha(k,j,i)*w_v_u(IVZ-1,i)-beta_u(IVZ-1,k,j,i))*((sc1*( ent(k+1,j,i) -ent(k-1,j,i) ) - sc2*( ent(k+2,j,i)-ent(k-2,j,i) ) + sc3*( ent(k+3,j,i)-ent(k-3,j,i) ) ) / dzw_(i) ) ;
        }
        // 2D
        else if (pmb->pmy_mesh->f2 && !pmb->pmy_mesh->f3 ){
          R=( tc1*ent(k,j,i) -tc2*ent1(k,j,i) + tc3*ent2(k,j,i) - tc4*ent3(k,j,i ) )/dt 
          + (alpha(k,j,i)*w_v_u(IVX-1,i)-beta_u(IVX-1,k,j,i))*( (sc1*( ent(k,j,i+1) -ent(k,j,i-1) ) - sc2*( ent(k,j,i+2)-ent(k,j,i-2) ) + sc3*( ent(k,j,i+3)-ent(k,j,i-3) ) ) / dxw_(i) )
          + (alpha(k,j,i)*w_v_u(IVY-1,i)-beta_u(IVY-1,k,j,i))*( (sc1*( ent(k,j+1,i) -ent(k,j-1,i) ) - sc2*( ent(k,j+2,i)-ent(k,j-2,i) ) + sc3*( ent(k,j+3,i)-ent(k,j-3,i) ) ) / dyw_(i) ) ;
        }
        // 1D
        else {
          R=( tc1*ent(k,j,i) -tc2*ent1(k,j,i) + tc3*ent2(k,j,i) - tc4*ent3(k,j,i ) )/dt 
          + (alpha(k,j,i)*w_v_u(IVX-1,i) - beta_u(IVX-1,k,j,i))*( (sc1*( ent(k,j,i+1) -ent(k,j,i-1) ) - sc2*( ent(k,j,i+2)-ent(k,j,i-2) ) + sc3*( ent(k,j,i+3)-ent(k,j,i-3) ) ) / dxw_(i) ) ;
        }
        R=std::abs(R);
        entropy_R(k,j,i) = std::min(cons,R);
      }
    }
  }

  // Calculate Flux Limiter
  //--------------------------------------------------------------------------------------
  // i-direction
  AthenaArray<Real> &x1ef_limiter = ef_limiter[X1DIR];
  // setting the loop limits
  pmb->precon->SetIndicialLimitsCalculateFluxes(1,il,iu,jl,ju,kl,ku);

  for (int k=kl ; k<=ku ;++k)
  {
    for (int j=jl ; j<= ju ; ++j)
    {
      for (int i=il; i<=iu; ++i)
      {
       x1ef_limiter(k,j,i)= 1.0 -0.5*(entropy_R(k,j,i-1) + entropy_R(k,j,i));
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->f2){
    // set the loop limits
    pmb->precon->SetIndicialLimitsCalculateFluxes(2,il,iu,jl,ju,kl,ku);
    AthenaArray<Real> &x2ef_limiter = ef_limiter[X2DIR];
    for (int k=kl ; k<=ku ;++k){
      for (int j=jl ; j<= ju ; ++j){
        for (int i=il; i<=iu; ++i){
          x2ef_limiter(k,j,i) = 1.0 -0.5*(entropy_R(k,j-1,i) + entropy_R(k,j,i));
        }
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->f3){
    // set the loop limits
    pmb->precon->SetIndicialLimitsCalculateFluxes(3,il,iu,jl,ju,kl,ku);
    AthenaArray<Real> &x2ef_limiter = ef_limiter[X2DIR];
    for (int j=jl; j<= ju; ++j){
      for (int k=kl ; k<=ku ;++k){
        for (int i=il; i<=iu; ++i){
          x2ef_limiter(k,j,i) = 1.0 -0.5*(entropy_R(k-1,j,i) + entropy_R(k,j,i));
        }
      }
    }
  }

  
  return;
}
