//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro_gr.cpp
//  \brief Implements functions for going between primitive and conserved variables in
//  general-relativistic hydrodynamics, as well as for computing wavespeeds.

// C++ headers
#include <algorithm>  // max()
#include <cfloat>     // FLT_MIN
#include <cmath>      // NAN, sqrt(), abs(), isfinite(), isnan(), pow()

// Athena++ headers
#include "eos.hpp"
#include "../athena.hpp"                   // enums, macros
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../field/field.hpp"              // FaceField
#include "../mesh/mesh.hpp"                // MeshBlock
#include "../z4c/z4c.hpp"                // MeshBlock

// Reprimand headers

#include "reprimand/con2prim_imhd.h"
#include "reprimand/c2p_report.h"
#include "reprimand/eos_idealgas.h"


// Declarations
static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, Real gamma_adi,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco);
Real fthr, fatm, rhoc;
Real k_adi, gamma_adi;
EOS_Toolkit::real_t atmo_rho;
EOS_Toolkit::real_t rho_strict;
bool  ye_lenient;
int max_iter;
EOS_Toolkit::real_t c2p_acc;
EOS_Toolkit::real_t max_b;
EOS_Toolkit::real_t max_z;
EOS_Toolkit::real_t atmo_eps;
EOS_Toolkit::real_t atmo_ye;
EOS_Toolkit::real_t atmo_cut;
EOS_Toolkit::real_t atmo_p;
using namespace EOS_Toolkit;
EOS_Toolkit::eos_thermal eos;
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
}

//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  //density_floor_ = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*(FLT_MIN)) );
  //pressure_floor_ = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*(FLT_MIN)) );
  rho_min_ = pin->GetOrAddReal("hydro", "rho_min", density_floor_);
  rho_pow_ = pin->GetOrAddReal("hydro", "rho_pow", 0.0);
  pgas_min_ = pin->GetOrAddReal("hydro", "pgas_min", pressure_floor_);
  pgas_pow_ = pin->GetOrAddReal("hydro", "pgas_pow", 0.0);
  gamma_max_ = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);
  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  g_.NewAthenaArray(NMETRIC, ncells1);
  g_inv_.NewAthenaArray(NMETRIC, ncells1);
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fixed_.NewAthenaArray(ncells3, ncells2, ncells1);
  fthr = pin -> GetReal("problem","fthr");
  fatm = pin -> GetReal("problem","fatm");
  rhoc = pin -> GetReal("problem","rhoc");
  k_adi = pin -> GetReal("hydro","k_adi");
  gamma_adi = pin -> GetReal("hydro","gamma");
using namespace EOS_Toolkit;
 //Get some EOS
  real_t max_eps = 10000.;
  real_t max_rho = 1e6;
  real_t adiab_ind = 1.0/(gamma_-1.0);
  eos = make_eos_idealgas(adiab_ind, max_eps, max_rho);

  //Set up atmosphere
  atmo_rho = fatm*rhoc;
  atmo_eps = atmo_rho*k_adi;
  atmo_ye = 0.0;
  atmo_cut = atmo_rho * fthr;
  atmo_p = eos.at_rho_eps_ye(atmo_rho, atmo_eps, atmo_ye).press();

  density_floor_ = atmo_rho;
  pressure_floor_ = atmo_p;

  //Primitive recovery parameters 
  rho_strict = 1e-20;
  ye_lenient = false;
  max_iter = 10000;
  c2p_acc = 1e-10;
  max_b = 10.;
  max_z = 1e5;




}


//----------------------------------------------------------------------------------------
// Variable inverter
// Inputs:
//   cons: conserved quantities
//   prim_old: primitive quantities from previous half timestep
//   bb: face-centered magnetic field
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field
// Notes:
//   follows Noble et al. 2006, ApJ 641 626 (N)
//       writing wgas_rel for W = \gamma^2 w
//       writing d for D
//       writing q for Q
//       writing qq for \tilde{Q}
//       writing uu for \tilde{u}
//       writing vv for v
//   implements formulas assuming no magnetic field

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
    const AthenaArray<Real> &prim_old, const FaceField &bb, AthenaArray<Real> &prim,
    AthenaArray<Real> &bb_cc, Coordinates *pco, int il, int iu, int jl, int ju, int kl,
    int ku, int coarse_flag) {
  // Parameters
using namespace EOS_Toolkit;
  const Real max_wgas_rel = 1.0e8;
  const Real initial_guess_multiplier = 10.0;
  const int initial_guess_multiplications = 10;
  int nn1 = iu+1;
  // Extract ratio of specific heats
  const Real &gamma_adi = gamma_;
  atmosphere atmo{atmo_rho, atmo_eps, atmo_ye, atmo_p, atmo_cut};
  con2prim_mhd cv2pv(eos, rho_strict, ye_lenient, max_z, max_b,
                     atmo, c2p_acc, max_iter);
      AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
      AthenaArray<Real> vcgamma_yz,vcgamma_zz,vcbeta_x,vcbeta_y;
      AthenaArray<Real> vcbeta_z, vcalpha;
      AthenaArray<Real> vcgammat_xx,vcgammat_xy,vcgammat_xz,vcgammat_yy;
      AthenaArray<Real> vcgammat_yz,vcgammat_zz;
      AthenaArray<Real> vcchi;
      AthenaArray<Real> order_flag;

      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha; //lapse
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi; //lapse
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u; //lapse
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd; //lapse
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gammat_dd; //lapse
      
      order_flag.NewAthenaArray(nn1);
if(coarse_flag==0){

      alpha.NewAthenaTensor(nn1);
      beta_u.NewAthenaTensor(nn1);
      gamma_dd.NewAthenaTensor(nn1);
      vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
      vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
      vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
      vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
      vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
      vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
      vcbeta_x.InitWithShallowSlice(pmy_block_->pz4c->storage.u,Z4c::I_Z4c_betax,1);
      vcbeta_y.InitWithShallowSlice(pmy_block_->pz4c->storage.u,Z4c::I_Z4c_betay,1);
      vcbeta_z.InitWithShallowSlice(pmy_block_->pz4c->storage.u,Z4c::I_Z4c_betaz,1);
      vcalpha.InitWithShallowSlice(pmy_block_->pz4c->storage.u,Z4c::I_Z4c_alpha,1);

} else{

      alpha.NewAthenaTensor(nn1);
      chi.NewAthenaTensor(nn1);
      beta_u.NewAthenaTensor(nn1);
      gamma_dd.NewAthenaTensor(nn1);
      gammat_dd.NewAthenaTensor(nn1);
      vcgammat_xx.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxx,1);
      vcgammat_xy.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxy,1);
      vcgammat_xz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gxz,1);
      vcgammat_yy.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gyy,1);
      vcgammat_yz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gyz,1);
      vcgammat_zz.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_gzz,1);
      vcbeta_x.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_betax,1);
      vcbeta_y.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_betay,1);
      vcbeta_z.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_betaz,1);
      vcalpha.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_alpha,1);
      vcchi.InitWithShallowSlice(pmy_block_->pz4c->coarse_u_,Z4c::I_Z4c_chi,1);
}

  // Go through cells
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
// TODO here call for Cell metric needs to be replaced with local calculation
// of cell centred metric from VC metric returning 1D array in x1 direction.
// NB func should return gamma(a,b,i), beta(a,i), alpha(i)
//      pco->CellMetric(k, j, il, iu, g_, g_inv_);
/*          gamma_dd(0,0,i) = g_(I11,i);      
          gamma_dd(0,1,i) = g_(I12,i);      
          gamma_dd(0,2,i) = g_(I13,i);      
          gamma_dd(1,1,i) = g_(I22,i);      
          gamma_dd(1,2,i) = g_(I23,i);      
          gamma_dd(2,2,i) = g_(I33,i);      
          alpha(i) = std::sqrt(-1.0/g_inv_(I00,i));
          beta_u(0,i) = SQR(alpha(i))*g_inv_(I01,i);
          beta_u(1,i) = SQR(alpha(i))*g_inv_(I02,i);
          beta_u(2,i) = SQR(alpha(i))*g_inv_(I03,i);
*/
if(coarse_flag==0){
      #pragma omp simd
      for (int i=il; i<=iu; ++i) {
          gamma_dd(0,0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
          gamma_dd(0,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
          gamma_dd(0,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
          gamma_dd(1,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
          gamma_dd(1,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
          gamma_dd(2,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
          alpha(i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcalpha(k,j,i));
          beta_u(0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcbeta_x(k,j,i));
          beta_u(1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcbeta_y(k,j,i));
          beta_u(2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcbeta_z(k,j,i));
}
} else{
      #pragma omp simd
      for (int i=il; i<=iu; ++i) {
          gammat_dd(0,0,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_xx(k,j,i));
          gammat_dd(0,1,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_xy(k,j,i));
          gammat_dd(0,2,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_xz(k,j,i));
          gammat_dd(1,1,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_yy(k,j,i));
          gammat_dd(1,2,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_yz(k,j,i));
          gammat_dd(2,2,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcgammat_zz(k,j,i));
          alpha(i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcalpha(k,j,i));
          chi(i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcchi(k,j,i));
          beta_u(0,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcbeta_x(k,j,i));
          beta_u(1,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcbeta_y(k,j,i));
          beta_u(2,i) = pmy_block_->pz4c->ig_coarse->map3d_VC2CC(vcbeta_z(k,j,i));
}
          for(int a=0;a<3;a++){
          for(int b=0;b<3;b++){
      #pragma omp simd
      for (int i=il; i<=iu; ++i) {
          gamma_dd(a,b,i) = gammat_dd(a,b,i)/chi(i);
}
}
}
}

//      }
      
      #pragma omp simd
      for (int i=il; i<=iu; ++i) {
//TODO not needed
        // Extract metric
/*
        const Real
            // unused:
            //&g_00 = g_(I00,i), &g_01 = g_(I01,i), &g_02 = g_(I02,i), &g_03 = g_(I03,i),
            // &g_10 = g_(I01,i);
            // &g_20 = g_(I02,i), &g_21 = g_(I12,i);
            // &g_30 = g_(I03,i), &g_31 = g_(I13,i), &g_32 = g_(I23,i);
            &g_11 = g_(I11,i), &g_12 = g_(I12,i), &g_13 = g_(I13,i),
            &g_22 = g_(I22,i), &g_23 = g_(I23,i),
            &g_33 = g_(I33,i);
        const Real &g00 = g_inv_(I00,i), &g01 = g_inv_(I01,i), &g02 = g_inv_(I02,i),
                   &g03 = g_inv_(I03,i), &g10 = g_inv_(I01,i), &g11 = g_inv_(I11,i),
                   &g12 = g_inv_(I12,i), &g13 = g_inv_(I13,i), &g20 = g_inv_(I02,i),
                   &g21 = g_inv_(I12,i), &g22 = g_inv_(I22,i), &g23 = g_inv_(I23,i),
                   &g30 = g_inv_(I03,i), &g31 = g_inv_(I13,i), &g32 = g_inv_(I23,i),
                   &g33 = g_inv_(I33,i);
        Real alpha = 1.0/std::sqrt(-g00);

// TODO modify to calc detgamma from new metric array
        Real detgamma  = g_11*(g_22*g_33 - SQR(g_23)) - g_12*(g_12*g_33 - g_23*g_13) + g_13*(g_12*g_23 - g_22*g_13);
        Real sqrtdetgamma = std::sqrt(detgamma);
        Real sqrtdetg = alpha*sqrtdetgamma;

*/

        // Extract conserved quantities
        const Real &Dg = cons(IDN,k,j,i);
        const Real &taug = cons(IEN,k,j,i);
        const Real &S_1g = cons(IVX,k,j,i);
        const Real &S_2g = cons(IVY,k,j,i);
        const Real &S_3g = cons(IVZ,k,j,i);

        //Extract prims
        Real &rho = prim(IDN,k,j,i);
        Real &pgas = prim(IPR,k,j,i);
        Real &uu1 = prim(IVX,k,j,i);
        Real &uu2 = prim(IVY,k,j,i);
        Real &uu3 = prim(IVZ,k,j,i);
        Real eps=0.0;
        Real w_lor = 1.0;
        Real dummy = 0.0;

        cons_vars_mhd evolved{Dg, taug, 0.0,
                          {S_1g,S_2g,S_3g}, {0.0,0.0,0.0}};
// TODO feed new metric to reprimand here
        sm_tensor2_sym<real_t, 3, false, false> gtens(gamma_dd(0,0,i),gamma_dd(0,1,i),gamma_dd(1,1,i),gamma_dd(0,2,i),gamma_dd(1,2,i),gamma_dd(2,2,i));
        sm_metric3 g_eos(gtens);
        prim_vars_mhd primitives;
        con2prim_mhd::report rep;
  bool collapse = true;
   //recover
    cv2pv(primitives, evolved, g_eos, rep);
    //check
    if (rep.failed())
    {
//      printf("coarse flag = %d\n",coarse_flag);
//      std::cerr << rep.debug_message();
//      printf("i=%d, j=%d, k=%d\n",i,j,k);
      //abort simulation
  if(collapse){
  if(std::isnan(Dg) || std::isnan(taug) || std::isnan(S_1g) || std::isnan(S_2g) || std::isnan(S_3g)){
   uu1 = 0.0;
   uu2 = 0.0;
   uu3 = 0.0;
   rho = atmo_rho;
   pgas = k_adi*pow(atmo_rho,gamma_adi);
   PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd, k, j, i, cons, pco);
  }
  } 
    }
    else {
      primitives.scatter(rho, eps, dummy, pgas, uu1, uu2, uu3, w_lor,dummy,dummy,dummy,dummy,dummy,dummy);
      uu1 *= w_lor;
      uu2 *= w_lor;
      uu3 *= w_lor;
      bool pgasfix = false;
//this shouldnt be triggered...
  if(collapse){
  if(std::isnan(Dg) || std::isnan(taug) || std::isnan(S_1g) || std::isnan(S_2g) || std::isnan(S_3g)){
   uu1 = 0.0;
   uu2 = 0.0;
   uu3 = 0.0;
   rho = atmo_rho;
   pgas = k_adi*pow(atmo_rho,gamma_adi);
   PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd, k, j, i, cons, pco);
  }
  } 
      if(rho < fthr*atmo_rho ){
      uu1 = 0.0;
      uu2 = 0.0;
      uu3 = 0.0;
      }
      if(pgas < k_adi*pow(atmo_rho,gamma_adi)){
      pgas = k_adi*pow(atmo_rho,gamma_adi);
      rho = atmo_rho;
      uu1 = 0.0;
      uu2 = 0.0;
      uu3 = 0.0;
      pgasfix = true;
      }
      if (rep.adjust_cons || rep.set_atmo || pgasfix) {
          PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd, k, j, i, cons, pco);



      }
    }

}
}
}
  return;
}

//----------------------------------------------------------------------------------------
// Function for converting all primitives to conserved variables
// Inputs:
//   prim: primitives
//   bb_cc: cell-centered magnetic field (unused)
//   pco: pointer to Coordinates
//   il,iu,jl,ju,kl,ku: index bounds of region to be updated
// Outputs:
//   cons: conserved variables
// Notes:
//   single-cell function exists for other purposes; call made to that function rather
//       than having duplicate code

void EquationOfState::PrimitiveToConserved(AthenaArray<Real> &prim,
     AthenaArray<Real> &bb_cc, AthenaArray<Real> &cons, Coordinates *pco, int il,
     int iu, int jl, int ju, int kl, int ku) {
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd; //lapse
  int nn1 = iu+1;
      gamma_dd.NewAthenaTensor(nn1);
      AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
      AthenaArray<Real> vcgamma_yz,vcgamma_zz, order_flag;
      order_flag.NewAthenaArray(nn1);
      vcgamma_xx.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
      vcgamma_xy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
      vcgamma_xz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
      vcgamma_yy.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
      vcgamma_yz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
      vcgamma_zz.InitWithShallowSlice(pmy_block_->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
// TODO need a call to calculate CC metric locally here - we don't
// actually need alpha beta or the inverse metric to calculate the conservatives.
// Just return a 1D array in the x1 direction of gamma_{ij}
//      pco->CellMetric(k, j, il, iu, g_, g_inv_);
      for (int i=il; i<=iu; ++i) {
          gamma_dd(0,0,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xx(k,j,i));
          gamma_dd(0,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xy(k,j,i));
          gamma_dd(0,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_xz(k,j,i));
          gamma_dd(1,1,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yy(k,j,i));
          gamma_dd(1,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_yz(k,j,i));
          gamma_dd(2,2,i) = pmy_block_->pz4c->ig->map3d_VC2CC(vcgamma_zz(k,j,i));
}
      //#pragma omp simd // fn is too long to inline
      for (int i=il; i<=iu; ++i) {
        PrimitiveToConservedSingle(prim, gamma_, gamma_dd, k, j, i, cons, pco);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for converting primitives to conserved variables in a single cell
// Inputs:
//   prim: 3D array of primitives
//   gamma_adi: ratio of specific heats
//   g,gi: 1D arrays of metric covariant and contravariant coefficients
//   k,j,i: indices of cell
//   pco: pointer to Coordinates
// Outputs:
//   cons: conserved variables set in desired cell

// TODO change arguments so instead of taking g(n,i), gi(n,i) it just takes gamma(a,b,i)
static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, Real gamma_adi,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco) {

    AthenaArray<Real> utilde_u;  // primitive gamma^i_a u^a
    AthenaArray<Real> utilde_d;  // primitive gamma^i_a u^a
    AthenaArray<Real> v_d;  // primitive gamma^i_a u^a

  utilde_u.NewAthenaArray(3);
  utilde_d.NewAthenaArray(3);
  v_d.NewAthenaArray(3);

  // Apply floor to primitive variables. This should be
  // identical to what RePrimAnd does.
  if (prim(IDN, k, j, i) < atmo_cut) {
    prim(IDN, k, j, i) = atmo_rho;
    prim(IVX, k, j, i) = 0.0;
    prim(IVY, k, j, i) = 0.0;
    prim(IVZ, k, j, i) = 0.0;
    prim(IPR, k, j, i) = atmo_p;
  }

  const Real &rho = prim(IDN,k,j,i);
  const Real &pgas = prim(IPR,k,j,i);
//  const Real &uu1 = prim(IVX,k,j,i);
//  const Real &uu2 = prim(IVY,k,j,i);
//  const Real &uu3 = prim(IVZ,k,j,i);
     for(int a=0;a<NDIM;++a){
              utilde_u(a) = prim(a+IVX,k,j,i);
      }

  // Calculate 4-velocity
//  Real alpha = std::sqrt(-1.0/gi(I00,i));
  Real detgamma = std::sqrt(Det3Metric(gamma_dd,i));
  if(std::isnan(detgamma)){
//  printf("detgamma is nan\n");
//  printf("x = %.16e, y = %.16e, z = %.16e, g_xx = %.16e\n g_xy = %.16e, g_xz = %.16e, g_yy = %.16e, g_yz = %.16e, g_zz = %.16e\n",pco->x1v(i), pco->x2v(j), pco->x3v(k), gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i), gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i));
  detgamma = 1;
 } 
  Real Wlor = 0.0;
  for(int a=0;a<NDIM;++a){
          for(int b=0;b<NDIM;++b){
                  Wlor += utilde_u(a)*utilde_u(b)*gamma_dd(a,b,i);
           }
       }
   Wlor = std::sqrt(1.0+Wlor);
   if(std::isnan(Wlor)){ 
//   printf("Wlor is nan\n");
//   printf("x = %.16e, y = %.16e, z = %.16e\n",pco->x1v(i), pco->x2v(j), pco->x3v(k));
   Wlor = 1.0;
   }
// NB definitions have changed slightly here - a different velocity is being used . Double check me!
  
  utilde_d.ZeroClear();

        for(int a=0;a<NDIM;++a){
          for(int b=0;b<NDIM;++b){
                  utilde_d(a) += utilde_u(b)*gamma_dd(a,b,i);
              }
      }

      for(int a=0;a<NDIM;++a){
             v_d(a) = utilde_d(a)/Wlor;
      }

//  Real utilde_d_1 = g(I11,i)*uu1 + g(I12,i)*uu2 + g(I13,i)*uu3;
//  Real utilde_d_2 = g(I12,i)*uu1 + g(I22,i)*uu2 + g(I23,i)*uu3;
//  Real utilde_d_3 = g(I13,i)*uu1 + g(I23,i)*uu2 + g(I33,i)*uu3;

//  Real v_1 = u_1/gamma; 
//  Real v_2 = u_2/gamma;  
//  Real v_3 = u_3/gamma;  
//  Real v_1 = utilde_d_1/gamma; 
//  Real v_2 = utilde_d_2/gamma;  
//  Real v_3 = utilde_d_3/gamma;  
  // Extract conserved quantities
  Real &Ddg = cons(IDN,k,j,i);
  Real &taudg = cons(IEN,k,j,i);
  Real &S_1dg = cons(IM1,k,j,i);
  Real &S_2dg = cons(IM2,k,j,i);
  Real &S_3dg = cons(IM3,k,j,i);

  // Set conserved quantities
  Real wgas = rho + gamma_adi/(gamma_adi-1.0) * pgas;
  Ddg = rho*Wlor*detgamma;
  taudg = wgas*SQR(Wlor)*detgamma - rho*Wlor*detgamma - pgas*detgamma;
  S_1dg = wgas*SQR(Wlor) * v_d(0)*detgamma  ;
  S_2dg = wgas*SQR(Wlor) * v_d(1)*detgamma  ;
  S_3dg = wgas*SQR(Wlor) * v_d(2)*detgamma  ;
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating relativistic sound speeds
// Inputs:
//   rho_h: enthalpy per unit volume
//   pgas: gas pressure
//   vx: 3-velocity component v^x
//   gamma_lorentz_sq: Lorentz factor \gamma^2
// Outputs:
//   plambda_plus: value set to most positive wavespeed
//   plambda_minus: value set to most negative wavespeed
// Notes:
//   same function as in adiabatic_hydro_sr.cpp
//     uses SR formula (should be called in locally flat coordinates)
//   references Mignone & Bodo 2005, MNRAS 364 126 (MB)

void EquationOfState::SoundSpeedsSR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq,
    Real *plambda_plus, Real *plambda_minus) {
  const Real gamma_adi = gamma_;
  Real cs_sq = gamma_adi * pgas / rho_h;                                 // (MB 4)
  Real sigma_s = cs_sq / (gamma_lorentz_sq * (1.0-cs_sq));
  Real relative_speed = std::sqrt(sigma_s * (1.0 + sigma_s - SQR(vx)));
  *plambda_plus = 1.0/(1.0+sigma_s) * (vx + relative_speed);             // (MB 23)
  *plambda_minus = 1.0/(1.0+sigma_s) * (vx - relative_speed);            // (MB 23)
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating relativistic sound speeds in arbitrary coordinates
// Inputs:
//   rho_h: enthalpy per unit volume
//   pgas: gas pressure
//   u0,u1: 4-velocity components u^0, u^1
//   g00,g01,g11: metric components g^00, g^01, g^11
// Outputs:
//   plambda_plus: value set to most positive wavespeed
//   plambda_minus: value set to most negative wavespeed
// Notes:
//   follows same general procedure as vchar() in phys.c in Harm
//   variables are named as though 1 is normal direction
/*
void EquationOfState::SoundSpeedsGR(Real rho_h, Real pgas, Real u0, Real u1, Real g00,
    Real g01, Real g11, Real *plambda_plus, Real *plambda_minus) {
  // Parameters and constants
  const Real discriminant_tol = -1.0e-1;  // values between this and 0 are considered 0
  const Real gamma_adi = gamma_;

  // Calculate comoving sound speed
  Real cs_sq = gamma_adi * pgas / rho_h;

  // Set sound speeds in appropriate coordinates
  Real a = SQR(u0) - (g00 + SQR(u0)) * cs_sq;
  Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cs_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cs_sq;
  Real d = SQR(b) - 4.0*a*c;
//  if (d < 0.0 and d > discriminant_tol) {
  if (d < 0.0) {
    d = 0.0;
    b = 0.0;
  }
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0*a);
  Real root_2 = (-b - d_sqrt) / (2.0*a);
  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
  return;
}
*/
void EquationOfState::SoundSpeedsGR(Real rho_h, Real pgas, Real vi, Real v2, Real alpha,
    Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus) {
  // Parameters and constants

  // Calculate comoving sound speed
  Real cs_sq = gamma_adi * pgas / rho_h;
  Real cs = std::sqrt(cs_sq);

  Real  root_1 = alpha*(vi*(1.0-cs_sq) + cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1-cs_sq)  )     )    )/(1-v2*cs_sq) - betai;

  Real  root_2 = alpha*(vi*(1.0-cs_sq) - cs*std::sqrt( (1-v2)*(gammaii*(1.0-v2*cs_sq) - vi*vi*(1-cs_sq)  )     )    )/(1-v2*cs_sq) - betai;
  bool collapse = true;
  if(collapse){ 
  if(std::isnan(root_1) || std::isnan(root_2)){
  root_1 = 1.0;
  root_2 = 1.0;
  }
  }
  if (root_1 > root_2) {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } else {
    *plambda_plus = root_2;
    *plambda_minus = root_1;
  }
  return;
}


//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);
  // Not applying position-dependent floors here in GR, nor using rho_min
  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33) {
  Real det = a11 * Determinant(a22, a23, a32, a33)
             - a12 * Determinant(a21, a23, a31, a33)
             + a13 * Determinant(a21, a22, a31, a32);
  return det;
}
Real Determinant(Real a11, Real a12, Real a21, Real a22) {
  return a11 * a22 - a12 * a21;
}
Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i)
{
  return - SQR(gamma(0,2,i))*gamma(1,1,i) + 
          2*gamma(0,1,i)*gamma(0,2,i)*gamma(1,2,i) - 
          gamma(0,0,i)*SQR(gamma(1,2,i)) - SQR(gamma(0,1,i))*gamma(2,2,i) +
          gamma(0,0,i)*gamma(1,1,i)*gamma(2,2,i);
}

}
