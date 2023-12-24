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
#include "../z4c/z4c.hpp"                  // MeshBlock
#include "../utils/linear_algebra.hpp"     // Det. & friends

// Reprimand headers

#include "reprimand/con2prim_imhd.h"
#include "reprimand/c2p_report.h"
#include "reprimand/eos_idealgas.h"

// debug:
#include "../hydro/hydro.hpp"



namespace {

  // Declarations
  inline static void PrimitiveToConservedSingle(
    AthenaArray<Real> &prim,
    Real gamma_adi,
    AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd,
    int k, int j, int i,
    AthenaArray<Real> &cons,
    Coordinates *pco);

  Real fthr, fatm, rhoc;
  Real epsatm;
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

}

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
  fthr = pin -> GetReal("problem","fthr");
  fatm = pin -> GetReal("problem","fatm");
  epsatm = pin -> GetOrAddReal("problem","epsatm",1.0e-8);
  rhoc = pin -> GetReal("problem","rhoc");
  k_adi = pin -> GetReal("hydro","k_adi");
  gamma_adi = pin -> GetReal("hydro","gamma");
  alpha_excision = pin->GetOrAddReal("hydro", "alpha_excision", 0.0);

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

  // Debug:
  if (false) {
    pmb->phydro->Hydro_IdealEoS_Cons2Prim(gamma_adi, cons, prim,
                                          const_cast<AthenaArray<Real> &>(prim_old),
                                          il, iu, jl, ju, kl, ku);
    return;
  }


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

    // do actual variable conversion ------------------------------------------
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
      Real eps=0.0;
      Real w_lor = 1.0;
      Real dummy = 0.0;

      if(alpha(i) > alpha_excision)
      {
        cons_vars_mhd evolved{Dg, taug, 0.0,
                              {S_1g, S_2g, S_3g},
                              {0.0, 0.0, 0.0}};
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
          // std::cout << "RF" << std::endl;
          /*
          if(eos_debug)
          {
            std::cerr << rep.debug_message();
            std::cerr << " i = " << i << ", j = " << j << ", k = " << k << ", x = " << pco->x1v(i) << ", y = " << pco->x2v(j) << ", z = " << pco->x3v(k) << ", t = " << pmy_block_->pmy_mesh->time  << ", MB = " << pmy_block_->gid  << "\n";
            ofile = fopen(ofname, "a");
            fprintf(ofile, "%.16g  %d  %d  %d  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  0  0  0  %.16g  %.16g  %.16g  %.16g  %.16g  %.16g  %d      .\n",pmy_block_->pmy_mesh->time,i,j,k,pco->x1v(i),pco->x2v(j),pco->x3v(k),Dg,taug,S_1g,S_2g,S_3g,gamma_dd(0,0,i),gamma_dd(0,1,i),gamma_dd(0,2,i),gamma_dd(1,1,i),gamma_dd(1,2,i),gamma_dd(2,2,i),coarse_flag);
            fclose(ofile);
          }
          */
          uu1 = 0.0;
          uu2 = 0.0;
          uu3 = 0.0;
          rho = atmo_rho;
          pgas = k_adi*pow(atmo_rho,gamma_adi);
          PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd,
                                     k, j, i, cons, pco);
        }
        else
        {
          primitives.scatter(rho, eps, dummy, pgas,
                             uu1, uu2, uu3, w_lor,
                             dummy, dummy, dummy,
                             dummy, dummy, dummy);
          uu1 *= w_lor;
          uu2 *= w_lor;
          uu3 *= w_lor;
          bool pgasfix = false;

          if(std::isnan(Dg)   || std::isnan(taug) ||
             std::isnan(S_1g) || std::isnan(S_2g) ||
             std::isnan(S_3g))
          {
            uu1 = 0.0;
            uu2 = 0.0;
            uu3 = 0.0;
            rho = atmo_rho;
            pgas = k_adi*pow(atmo_rho,gamma_adi);
            PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd,
                                       k, j, i, cons, pco);
            printf("NAN after success");
          }

          if(rho < fthr*atmo_rho)
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
            PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd,
                                       k, j, i, cons, pco);
          }
        }

      }
      else
      {
        // hydro excision triggered if lapse < 0.3
        pgas = k_adi*pow(atmo_rho,gamma_adi);
        rho = atmo_rho;
        uu1 = 0.0;
        uu2 = 0.0;
        uu3 = 0.0;
        PrimitiveToConservedSingle(prim, gamma_adi, gamma_dd,
                                   k, j, i, cons, pco);
      }
    }
  }

  // clean-up
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
  // Make this more readable
  MeshBlock* pmb = pmy_block_;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  Z4c * pz4c = pmb->pz4c;

  // Debug:
  if (false) {
    pmb->phydro->Hydro_IdealEoS_Prim2Cons(gamma_adi, prim, cons,
                                          il, iu, jl, ju, kl, ku);
    return;
  }

  // Require only ADM 3-metric and no gauge.
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_gamma_dd;
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;

  adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
  gamma_dd.NewAthenaTensor(pmb->nverts1);

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
    for (int i=IL; i<=IU; ++i)
    {
      PrimitiveToConservedSingle(prim, gamma_, gamma_dd,
                                 k, j, i, cons, pco);
    }
  }

  gamma_dd.DeleteAthenaTensor();
  return;
}


namespace {
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

inline static void PrimitiveToConservedSingle(
  AthenaArray<Real> &prim, Real gamma_adi,
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma_dd,
  int k, int j, int i,
  AthenaArray<Real> &cons, Coordinates *pco)
{
  using namespace LinearAlgebra;

  AthenaArray<Real> utilde_u;  // primitive gamma^i_a u^a
  AthenaArray<Real> utilde_d;
  AthenaArray<Real> v_d;

  utilde_u.NewAthenaArray(3);
  utilde_d.NewAthenaArray(3);
  v_d.NewAthenaArray(3);

  // Apply floor to primitive variables. This should be
  // identical to what RePrimAnd does.
  // coutBoldBlue("cut, gamma_adi: ");
  // std::cout << atmo_cut << ", ";
  // std::cout << gamma_adi << std::endl;

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

  // const Real &uu1 = prim(IVX,k,j,i);
  // const Real &uu2 = prim(IVY,k,j,i);
  // const Real &uu3 = prim(IVZ,k,j,i);

  for(int a=0;a<NDIM;++a)
  {
    utilde_u(a) = prim(a+IVX,k,j,i);
  }

  // Calculate 4-velocity
  // Real alpha = std::sqrt(-1.0/gi(I00,i));
  Real detgamma = std::sqrt(Det3Metric(gamma_dd, i));

  if(std::isnan(detgamma))
  {
    if(0)
    {
      printf("detgamma is nan\n");
      printf("x = %.16e, y = %.16e, z = %.16e, \n "
              "g_xx = %.16e g_xy = %.16e, g_xz = %.16e, "
              "g_yy = %.16e, g_yz = %.16e, g_zz = %.16e\n",
              pco->x1v(i), pco->x2v(j), pco->x3v(k),
              gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
              gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i));
    }
    detgamma = 1;
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
  // NB definitions have changed slightly here - a different velocity is being used . Double check me!

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

  // Real utilde_d_1 = g(I11,i)*uu1 + g(I12,i)*uu2 + g(I13,i)*uu3;
  // Real utilde_d_2 = g(I12,i)*uu1 + g(I22,i)*uu2 + g(I23,i)*uu3;
  // Real utilde_d_3 = g(I13,i)*uu1 + g(I23,i)*uu2 + g(I33,i)*uu3;

  // Real v_1 = u_1/gamma;
  // Real v_2 = u_2/gamma;
  // Real v_3 = u_3/gamma;
  // Real v_1 = utilde_d_1/gamma;
  // Real v_2 = utilde_d_2/gamma;
  // Real v_3 = utilde_d_3/gamma;

  // Extract conserved quantities
  Real &Ddg = cons(IDN,k,j,i);
  Real &taudg = cons(IEN,k,j,i);
  Real &S_1dg = cons(IM1,k,j,i);
  Real &S_2dg = cons(IM2,k,j,i);
  Real &S_3dg = cons(IM3,k,j,i);

  // Set conserved quantities
  // if (std::abs(detgamma - 1) > 1e-12)
  // {
  //   std::cout << "detgamma: " << detgamma;
  //   std::cout << "(i,j,k) = ";
  //   std::cout << i << "," << j << "," << k << std::endl;
  // }

  Real wgas = rho + gamma_adi/(gamma_adi-1.0) * pgas;
  Ddg = rho*Wlor*detgamma;
  taudg = wgas*SQR(Wlor)*detgamma - rho*Wlor*detgamma - pgas*detgamma;
  S_1dg = wgas*SQR(Wlor) * v_d(0)*detgamma;
  S_2dg = wgas*SQR(Wlor) * v_d(1)*detgamma;
  S_3dg = wgas*SQR(Wlor) * v_d(2)*detgamma;

  utilde_u.DeleteAthenaArray();
  utilde_d.DeleteAthenaArray();
  v_d.DeleteAthenaArray();
  return;

}

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


  if(prim.GetDim4()==1){
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);
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

