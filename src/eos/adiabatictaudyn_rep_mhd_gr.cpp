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
#include "../utils/linear_algebra.hpp"     // Det. & friends

//#ifdef Z4C_AHF
#include "../z4c/ahf.hpp"                // MeshBlock
//#endif


// Reprimand headers

#include "reprimand/con2prim_imhd.h"
#include "reprimand/c2p_report.h"
#include "reprimand/eos_idealgas.h"


// Declarations
static void PrimitiveToConservedSingle(AthenaArray<Real> &prim, Real gamma_adi,
const AthenaArray<Real> &bb_cc,    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd, 
    AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> const & beta_u,   AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> const & alpha, 
    int k, int j, int i,
    AthenaArray<Real> &cons, Coordinates *pco, const Real detg_ceil );
Real fthr, fatm, rhoc;
Real k_adi, gamma_adi;
EOS_Toolkit::real_t atmo_rho;
EOS_Toolkit::real_t rho_strict;
bool  ye_lenient;
bool eos_debug;
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
/*
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
}
*/
//----------------------------------------------------------------------------------------
// Constructor
// Inputs:
//   pmb: pointer to MeshBlock
//   pin: pointer to runtime inputs

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) 
{
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  rho_min_ = pin->GetReal("hydro", "rho_min");
  rho_pow_ = pin->GetOrAddReal("hydro", "rho_pow", 0.0);
  pgas_min_ = pin->GetReal("hydro", "pgas_min");
  pgas_pow_ = pin->GetOrAddReal("hydro", "pgas_pow", 0.0);
  gamma_max_ = pin->GetOrAddReal("hydro", "gamma_max", 1000.0);
  int ncells1 = pmb->block_size.nx1 + 2*NGHOST;
  int ncells2 = (pmb->block_size.nx2 > 1) ? pmb->block_size.nx2 + 2*NGHOST : 1;
  int ncells3 = (pmb->block_size.nx3 > 1) ? pmb->block_size.nx3 + 2*NGHOST : 1;
  fthr = pin -> GetReal("problem","fthr");
  fatm = pin -> GetReal("problem","fatm");
  rhoc = pin -> GetReal("problem","rhoc");
  k_adi = pin -> GetReal("hydro","k_adi");
  gamma_adi = pin -> GetReal("hydro","gamma");
  alpha_excision = pin->GetOrAddReal("hydro", "alpha_excision", 0.0);
  b_excision = pin->GetOrAddBoolean("hydro", "b_excision", false);
  horizon_excision = pin->GetOrAddBoolean("hydro", "horizon_excision", false);
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
  max_b = pin -> GetOrAddReal("problem","max_b",10.0);
  max_z = pin -> GetOrAddReal("problem","max_z",20.0);

/*
eos_debug = pin->GetOrAddBoolean("problem","eos_debug", false);
if(eos_debug){
  std::snprintf(ofname, BUFSIZ, "eos_fail_mb.%d.txt", pmb->gid);
  ofile = fopen(ofname, "a");
  fprintf(ofile, "EOS failures in MB %d.\n", pmb->gid);
  fprintf(ofile, "#0 time #1: i  #2: j  #3: k  #4: x  #5: y  #6: z  #7: D  #8: tau  #9: S_1  #10: S_2  #11: S_3  #12: B^1  #13: B^2  #14: B^3  #15: g_11  #16: g_12  #17: g_13  #18: g_22  #19: g_23  #20: g_33  #21: coarseflag      .\n");
  fclose(ofile);
}
*/
}


//----------------------------------------------------------------------------------------
// Variable inverter
void EquationOfState::ConservedToPrimitive(
  AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, 
  const FaceField &bb, 
  AthenaArray<Real> &prim,
  AthenaArray<Real> &bb_cc, 
  Coordinates *pco, 
  int il, int iu, 
  int jl, int ju, 
  int kl, int ku, 
  int coarse_flag) 
{
  MeshBlock* pmb = pmy_block_;
  GRDynamical* pco_gr;
  int nn1;

  if (coarse_flag)
  {
    pco_gr = static_cast<GRDynamical*>(pmb->pmr->pcoarsec);
    nn1 = pmb->ncv1;
  }
  else
  {
    pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
    nn1 = pmb->nverts1;
  }

  // Parameters
  using namespace EOS_Toolkit;
  // Extract ratio of specific heats
  const Real &gamma_adi = gamma_;
  atmosphere atmo{atmo_rho, atmo_eps, atmo_ye, atmo_p, atmo_cut};
  con2prim_mhd cv2pv(eos, rho_strict, ye_lenient, max_z, max_b,
                     atmo, c2p_acc, max_iter);

  Real detg_ceil = pow((pmy_block_->pz4c->opt.chi_div_floor),-1.5);
  // Prepare variables for conversion -----------------------------------------
  Z4c * pz4c = pmy_block_->pz4c;

  // quantities we have (sliced below)
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_gamma_dd;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> z4c_gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> z4c_alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> z4c_beta_u;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> z4c_chi;

  // quantities we will construct (on CC)
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> chi;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> rchi;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;

  alpha.NewAthenaTensor(   nn1);
  beta_u.NewAthenaTensor(  nn1);
  gamma_dd.NewAthenaTensor(nn1);

  // need to pull quantities from appropriate storage
  if (!coarse_flag)
  {
    adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    z4c_alpha.InitWithShallowSlice(   pz4c->storage.u,   Z4c::I_Z4c_alpha);
    z4c_beta_u.InitWithShallowSlice(  pz4c->storage.u,   Z4c::I_Z4c_betax);
  }
  else
  {
    chi.NewAthenaTensor( nn1);  // additionally needed on coarse rep.
    rchi.NewAthenaTensor(nn1);

    z4c_gamma_dd.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_gxx);
    z4c_alpha.InitWithShallowSlice(   pz4c->coarse_u_, Z4c::I_Z4c_alpha);
    z4c_beta_u.InitWithShallowSlice(  pz4c->coarse_u_, Z4c::I_Z4c_betax);
    z4c_chi.InitWithShallowSlice(     pz4c->coarse_u_, Z4c::I_Z4c_chi);
  }

  // sanitize loop-limits (coarse / fine auto-switched)
  int IL, IU, JL, JU, KL, KU;
  pco_gr->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  // Restrict further the ranges to the input argument
  IL = std::max(il, IL);
  IU = std::min(iu, IU);

  JL = std::max(jl, JL);
  JU = std::min(ju, JU);

  KL = std::max(kl, KL);
  KU = std::min(ku, KU);

  // iterate over cells
  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    if (!coarse_flag)
    {
      pco_gr->GetGeometricFieldCC(gamma_dd, adm_gamma_dd, k, j);
      pco_gr->GetGeometricFieldCC(alpha,    z4c_alpha,    k, j);
      pco_gr->GetGeometricFieldCC(beta_u,   z4c_beta_u,   k, j);
    }
    else
    {
      // coarse variables require also further oper.
      pco_gr->GetGeometricFieldCC(gamma_dd, z4c_gamma_dd, k, j);
      pco_gr->GetGeometricFieldCC(alpha,    z4c_alpha,    k, j);
      pco_gr->GetGeometricFieldCC(beta_u,   z4c_beta_u,   k, j);
      pco_gr->GetGeometricFieldCC(chi,      z4c_chi,      k, j);

      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        rchi(i) = 1. / chi(i);
      }

      for(int a=0; a<NDIM; ++a)
      for(int b=a; b<NDIM; ++b)
      {
        #pragma omp simd
        for (int i=IL; i<=IU; ++i)
        {
          gamma_dd(a,b,i) = gamma_dd(a,b,i) * rchi(i);
        }
      }

    }
    pmy_block_->pfield->CalculateCellCenteredField(bb, bb_cc, pco, il, iu, jl, ju, kl, ku);


      #pragma omp simd
    for (int i=IL; i<=IU; ++i) 
    {
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
        Real &bb1g = bb_cc(IB1,k,j,i);
        Real &bb2g = bb_cc(IB2,k,j,i);
        Real &bb3g = bb_cc(IB3,k,j,i);
        Real eps=0.0;
        Real w_lor = 1.0;
        Real dummy = 0.0;

//AHF hydro excision
//      
//
        bool in_horizon = false;
        if(horizon_excision)
        {
          Real horizon_radius;
          for (auto pah_f : pmy_block_->pmy_mesh->pah_finder) 
          {
            horizon_radius = pah_f->GetHorizonRadius();
            if(((SQR(pco->x1v(i)) + SQR(pco->x2v(j)) + SQR(pco->x3v(k))) < SQR(horizon_radius)) || (alpha(i) < alpha_excision)     )  
            {
              in_horizon = true;
            }
          }
        } 
        else 
        {
          if(alpha(i) < alpha_excision)
          {
            in_horizon = true;
          }
        }

//        if(alpha(i) > alpha_excision){
        if(!in_horizon)
        {

          cons_vars_mhd evolved{Dg, taug, 0.0,
                              {S_1g, S_2g, S_3g}, 
                              {bb1g,bb2g,bb3g}};
          sm_tensor2_sym<real_t, 3, false, false> gtens(gamma_dd(0,0,i),
                                                      gamma_dd(0,1,i),
                                                      gamma_dd(1,1,i),
                                                      gamma_dd(0,2,i),
                                                      gamma_dd(1,2,i),
                                                      gamma_dd(2,2,i));
          sm_metric3 g_eos(gtens);
          prim_vars_mhd primitives;
          con2prim_mhd::report rep;
   //recover
          cv2pv(primitives, evolved, g_eos, rep);
    //check
        if (rep.failed())
        {
//      printf("coarse flag = %d\n",coarse_flag);
/*
if(eos_debug){
     std::cerr << rep.debug_message();
     std::cerr << " i = " << i << ", j = " << j << ", k = " << k << ", x = " << pco->x1v(i) << ", y = " << pco->x2v(j) << ", z = " << pco->x3v(k) << ", t = " << pmy_block_->pmy_mesh->time  << ", MB = " << pmy_block_->gid  << "\n";
     ofile = fopen(ofname, "a");
     fprintf(ofile, "%.16g  %d  %d  %d  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %d      .\n",pmy_block_->pmy_mesh->time,i,j,k,pco->x1v(i),pco->x2v(j),pco->x3v(k),Dg,taug,S_1g,S_2g,S_3g,bb1g,bb2g,bb3g,gamma_dd(0,0,i),gamma_dd(0,1,i),gamma_dd(0,2,i),gamma_dd(1,1,i),gamma_dd(1,2,i),gamma_dd(2,2,i),coarse_flag);
     fclose(ofile);   
}
*/
//     std::cerr << " " << Dg << " " << taug << " " << S_1g << " " << S_2g << " " << S_3g << " " << bb1g << " " << bb2g  << " " << bb3g  << " " << gamma_dd(0,0,i) << " " << gamma_dd(0,1,i) << " " << gamma_dd(0,2,i) << " " << gamma_dd(1,1,i) << " " << gamma_dd(1,2,i) << " " << gamma_dd(2,2,i) << " "  << "\n";
//     std::cerr << "Err code = " << rep.status << "\n";

//     if(rep.status == rep.ROOT_FAIL_CONV){
//     std::cerr << "ROOT_FAIL_CONV. Not adjusting Primitives.\n"; 
//     return;
//     }
 
//    printf("i=%d, j=%d, k=%d, x= %.16g, y = %.16g, z = %1.6g\n",i,j,k, pco->x1v(i), pco->x2v(j), pco->x3v(k));
      //abort simulation
//  if(collapse){
//  if(std::isnan(Dg) || std::isnan(taug) || std::isnan(S_1g) || std::isnan(S_2g) || std::isnan(S_3g)){



          uu1 = 0.0;
          uu2 = 0.0;
          uu3 = 0.0;
          rho = atmo_rho;
          pgas = k_adi*pow(atmo_rho,gamma_adi);
          PrimitiveToConservedSingle(prim, gamma_adi,bb_cc, 
                                     gamma_dd, beta_u, alpha, 
                                     k, j, i, cons, pco, detg_ceil);
//  }
//  } 
        }
        else 
        {
          primitives.scatter(rho, eps, dummy, pgas, 
                             uu1, uu2, uu3, w_lor,dummy,
                             dummy,dummy,dummy,dummy,dummy);
          uu1 *= w_lor;
          uu2 *= w_lor;
          uu3 *= w_lor;
          bool pgasfix = false;
//this shouldnt be triggered...
//  if(collapse){
          if(std::isnan(Dg) || std::isnan(taug) || 
             std::isnan(S_1g) || std::isnan(S_2g) || 
             std::isnan(S_3g))
          {
            uu1 = 0.0;
            uu2 = 0.0;
            uu3 = 0.0;
            rho = atmo_rho;
            pgas = k_adi*pow(atmo_rho,gamma_adi);
            PrimitiveToConservedSingle(prim, gamma_adi, bb_cc, 
                                       gamma_dd, beta_u, alpha, 
                                       k, j, i, cons, pco, detg_ceil);
            printf("NAN after success");
          } 
//  } 
          if(rho < fthr*atmo_rho )
          {
            uu1 = 0.0;
            uu2 = 0.0;
            uu3 = 0.0;
            pgas = k_adi*pow(atmo_rho,gamma_adi);
            rho = atmo_rho;
            pgasfix = true;
          }
          if (rep.adjust_cons || rep.set_atmo || pgasfix) 
          {
            PrimitiveToConservedSingle(prim, gamma_adi, bb_cc, gamma_dd, 
                                       beta_u, alpha, k, j, i, cons, pco, 
                                       detg_ceil);



          }
        }
      } 
      else
      { // hydro excision triggered if lapse < 0.3
        pgas = k_adi*pow(atmo_rho,gamma_adi);
        rho = atmo_rho;
        uu1 = 0.0;
        uu2 = 0.0;
        uu3 = 0.0;
// Experiment with exicising B field - probably dont want to use
/*      
        if(b_excision)
        {
          bb_cc(0,k,j,i) = 0.0;
          bb_cc(1,k,j,i) = 0.0;
          bb_cc(2,k,j,i) = 0.0;
          pmy_block_->pfield->b1.x1f(k,j,i) = pmy_block_->pfield->b1.x2f(k,j,i) = 
          pmy_block_->pfield->b1.x3f(k,j,i) = pmy_block_->pfield->b1.x1f(k,j,i+1) = 
          pmy_block_->pfield->b1.x2f(k,j+1,i) = pmy_block_->pfield->b1.x3f(k+1,j,i) = 0.0;
        }
*/
        PrimitiveToConservedSingle(prim, gamma_adi, bb_cc, 
                                   gamma_dd, beta_u, alpha, 
                                   k, j, i, cons, pco, detg_ceil);

      }   
    }
}
  alpha.DeleteAthenaTensor();
  beta_u.DeleteAthenaTensor();
  gamma_dd.DeleteAthenaTensor();

  if (coarse_flag)
  {
    chi.DeleteAthenaTensor();
    rchi.DeleteAthenaTensor();
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

void EquationOfState::PrimitiveToConserved(
  AthenaArray<Real> &prim,
  AthenaArray<Real> &bb_cc, 
  AthenaArray<Real> &cons,
  Coordinates *pco,
  int il, int iu,
  int jl, int ju,
  int kl, int ku
)
{


  MeshBlock* pmb = pmy_block_;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  Z4c * pz4c = pmb->pz4c;
  Real detg_ceil = pow((pz4c->opt.chi_div_floor),-1.5);

  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> z4c_alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> z4c_beta_u;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;

  adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
  z4c_alpha.InitWithShallowSlice(   pz4c->storage.u,   Z4c::I_Z4c_alpha);
  z4c_beta_u.InitWithShallowSlice(  pz4c->storage.u,   Z4c::I_Z4c_betax);
  gamma_dd.NewAthenaTensor(pmb->nverts1);
  alpha.NewAthenaTensor(pmb->nverts1);
  beta_u.NewAthenaTensor(pmb->nverts1);


  // sanitize loop-limits
  int IL, IU, JL, JU, KL, KU;
  pco_gr->GetGeometricFieldCCIdxRanges(
    IL, IU,
    JL, JU,
    KL, KU);

  // Restrict further the ranges to the input argument
  IL = std::max(il, IL);
  IU = std::min(iu, IU);

  JL = std::max(jl, JL);
  JU = std::min(ju, JU);

  KL = std::max(kl, KL);
  KU = std::min(ku, KU);

  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    pco_gr->GetGeometricFieldCC(gamma_dd, adm_gamma_dd, k, j);
    pco_gr->GetGeometricFieldCC(alpha,    z4c_alpha,    k, j);
    pco_gr->GetGeometricFieldCC(beta_u,   z4c_beta_u,   k, j);
      for (int i=IL; i<=IU; ++i) {
        PrimitiveToConservedSingle(prim, gamma_, bb_cc, gamma_dd, beta_u, alpha, k, j, i, cons, pco, detg_ceil);
      }
    }
  
  gamma_dd.DeleteAthenaTensor();
  alpha.DeleteAthenaTensor();
  beta_u.DeleteAthenaTensor();
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

static void PrimitiveToConservedSingle(
  AthenaArray<Real> &prim, Real gamma_adi,const AthenaArray<Real> &bb_cc,
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd,
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> const & beta_u, 
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> const & alpha, 
  int k, int j, int i, AthenaArray<Real> &cons, Coordinates *pco, 
  const Real detg_ceil) 
{
  using namespace LinearAlgebra;

    AthenaArray<Real> utilde_u;  // primitive gamma^i_a u^a
    AthenaArray<Real> utilde_d;  // primitive gamma^i_a u^a
    AthenaArray<Real> v_d, v_u, bb_u, bi_d, bi_u, beta_d;  // primitive gamma^i_a u^a

  utilde_u.NewAthenaArray(3);
  utilde_d.NewAthenaArray(3);
  v_d.NewAthenaArray(3);
  v_u.NewAthenaArray(3);
  bb_u.NewAthenaArray(3);
  bi_u.NewAthenaArray(3);
  bi_d.NewAthenaArray(3);
  beta_d.NewAthenaArray(3);


  // Apply floor to primitive variables. This should be
  // identical to what RePrimAnd does.
  if (prim(IDN, k, j, i) < atmo_cut) 
  {
    prim(IDN, k, j, i) = atmo_rho;
    prim(IVX, k, j, i) = 0.0;
    prim(IVY, k, j, i) = 0.0;
    prim(IVZ, k, j, i) = 0.0;
    prim(IPR, k, j, i) = atmo_p;
  }
  const Real &rho = prim(IDN,k,j,i);
  const Real &pgas = prim(IPR,k,j,i);
  for(int a=0;a<NDIM;++a)
  {
    utilde_u(a) = prim(a+IVX,k,j,i);
  }

  // Calculate 4-velocity
  Real detgamma = std::sqrt(Det3Metric(gamma_dd,i));

  if(std::isinf(detgamma))
  {
    detgamma = 1.0;
  }
  if(std::isnan(detgamma))
  {
    if(0)
    {
      printf("detgamma is nan\n");
      printf("x = %.16e, y = %.16e, z = %.16e, g_xx = %.16e\n g_xy = %.16e," 
              "g_xz = %.16e, g_yy = %.16e, g_yz = %.16e, g_zz = %.16e\n",
              pco->x1v(i), pco->x2v(j), pco->x3v(k), gamma_dd(0,0,i), gamma_dd(0,1,i), 
              gamma_dd(0,2,i), gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i));
     }
   detgamma = 1.0;
  } 
  if(detgamma > detg_ceil)
  {
    detgamma = detg_ceil;
  }
  Real Wlor = 0.0;
  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      Wlor += utilde_u(a)*utilde_u(b)*gamma_dd(a,b,i);
    }
  }
  Wlor = std::sqrt(1.0+Wlor);
  if(std::isnan(Wlor))
  { 
    if(0)
    {
       printf("Wlor is nan\n");
       printf("x = %.16e, y = %.16e, z = %.16e\n",pco->x1v(i), pco->x2v(j), pco->x3v(k));
     }
   Wlor = 1.0;
  }
  
  utilde_d.ZeroClear();

  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      utilde_d(a) += utilde_u(b)*gamma_dd(a,b,i);
    }
  }

  for(int a=0;a<NDIM;++a)
  {
    v_d(a) = utilde_d(a)/Wlor;
  }
  for(int a=0;a<NDIM;++a)
  {
    v_u(a) = utilde_u(a)/Wlor;
  }

  beta_d.ZeroClear();
  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      beta_d(a) += beta_u(b,i)*gamma_dd(a,b,i);
    }
  }
      


  // Extract conserved quantities
  Real &Ddg = cons(IDN,k,j,i);
  Real &taudg = cons(IEN,k,j,i);
  Real &S_1dg = cons(IM1,k,j,i);
  Real &S_2dg = cons(IM2,k,j,i);
  Real &S_3dg = cons(IM3,k,j,i);
  bb_u(0) = bb_cc(0,k,j,i)/detgamma;
  bb_u(1) = bb_cc(1,k,j,i)/detgamma;
  bb_u(2) = bb_cc(2,k,j,i)/detgamma;
  Real b0_u = 0.0;
  for(int a=0;a<NDIM;++a)
  {
    b0_u += Wlor*bb_u(a)*v_d(a)/alpha(i);
  }
  for(int a=0;a<NDIM;++a)
  {
    bi_u(a) = (bb_u(a) + alpha(i)*b0_u*Wlor*(v_u(a) - beta_u(a,i)/alpha(i)))/Wlor;
  }

  for(int a=0;a<NDIM;++a)
  {
    bi_d(a) = beta_d(a) * b0_u;
    for(int b=0;b<NDIM;++b)
    {
      bi_d(a) += gamma_dd(a,b,i)*bi_u(b);
    }
  }
 
 
 
  Real bsq = alpha(i)*alpha(i)*b0_u*b0_u/(Wlor*Wlor);
  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      bsq += bb_u(a)*bb_u(b)*gamma_dd(a,b,i)/(Wlor*Wlor);
    }
  }
   
 
   // Set conserved quantities
  Real wgas = rho + gamma_adi/(gamma_adi-1.0) * pgas;
  Ddg = rho*Wlor*detgamma;
  taudg = (wgas+bsq)*SQR(Wlor)*detgamma - rho*Wlor*detgamma - (pgas + bsq/2.0)*detgamma - 
          alpha(i)*alpha(i)*b0_u*b0_u*detgamma;
  S_1dg = (wgas+bsq)*SQR(Wlor) * v_d(0)*detgamma - alpha(i)*b0_u*bi_d(0)*detgamma ;
  S_2dg = (wgas+bsq)*SQR(Wlor) * v_d(1)*detgamma - alpha(i)*b0_u*bi_d(1)*detgamma  ;
  S_3dg = (wgas+bsq)*SQR(Wlor) * v_d(2)*detgamma - alpha(i)*b0_u*bi_d(2)*detgamma  ;

  utilde_u.DeleteAthenaArray();
  utilde_d.DeleteAthenaArray();
  v_d.DeleteAthenaArray();
  v_u.DeleteAthenaArray();
  bb_u.DeleteAthenaArray();
  bi_u.DeleteAthenaArray();
  bi_d.DeleteAthenaArray();
  beta_d.DeleteAthenaArray();
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
/*
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
*/
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

//TODO introduce FastMagnetoSonic function here Take care of b field densitisation.
/*void EquationOfState::SoundSpeedsGR(Real rho_h, Real pgas, Real vi, Real v2, Real alpha,
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

*/
void EquationOfState::FastMagnetosonicSpeedsGR(Real rho_h, Real pgas, Real b_sq, Real vi, Real v2, Real alpha,
      Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus) {
  // Parameters and constants
  const Real gamma_adi = gamma_;
  Real Wlor = std::sqrt(1.0-v2);
  Wlor = 1.0/Wlor;
  Real u0 = Wlor/alpha;
  Real g00 = -1.0/(alpha*alpha);
  Real g01 = betai/(alpha*alpha);
  Real u1 = (vi-betai/alpha)*Wlor;
  Real g11 = gammaii - betai*betai/(alpha*alpha);
  // Calculate comoving fast magnetosonic speed
  Real cs_sq = gamma_adi * pgas / rho_h;
  Real va_sq = b_sq / (b_sq + rho_h);
  Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

  // Set fast magnetosonic speeds in appropriate coordinates
  Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
  Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cms_sq);
  Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
  Real d = std::max(SQR(b) - 4.0*a*c, 0.0);
  Real d_sqrt = std::sqrt(d);
  Real root_1 = (-b + d_sqrt) / (2.0*a);
  Real root_2 = (-b - d_sqrt) / (2.0*a);
  if (root_1 > root_2) 
  {
    *plambda_plus = root_1;
    *plambda_minus = root_2;
  } 
  else 
  {
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


  if(prim.GetDim4()==1){    
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);
  // Not applying position-dependent floors here in GR, nor using rho_min
  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
  } else if(prim.GetDim4()==5){
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_p  = prim(IPR,k,j,i); 
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
  }
  else{
  printf("prim.GetDim4() = %d ApplyPrimitiveFloors only works with 1 or 5",prim.GetDim4());
  }
  // Not applying position-dependent floors here in GR, nor using rho_min
  // apply density floor
//  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
//  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}
