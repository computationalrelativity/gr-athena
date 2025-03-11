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
#include "../athena_aliases.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../utils/linear_algebra.hpp"

//#ifdef Z4C_AHF
#include "../z4c/ahf.hpp"
//#endif


// Reprimand headers
#include "reprimand/con2prim_imhd.h"
#include "reprimand/c2p_report.h"
#include "reprimand/eos_idealgas.h"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

namespace {

// Declarations
inline static void PrimitiveToConservedSingle(
  MeshBlock * pmb,
  AthenaArray<Real> &prim,
  Real gamma_adi,
  AthenaArray<Real> &bb_cc,
  const AT_N_sca & alpha_,
  const AT_N_vec & beta_u_,
  const AT_N_sym & adm_gamma_dd_,
  int k, int j, int i,
  AthenaArray<Real> &cons,
  Coordinates *pco,
  const Real detg_ceil);

using namespace EOS_Toolkit;

Real fthr, fatm;
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
EOS_Toolkit::real_t atmo_cut_p;
EOS_Toolkit::real_t atmo_p;
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

  // These variables (referred to elsewhere) are not meant to do anything
  rho_min_ = NAN;
  rho_pow_ = NAN;
  pgas_min_ = NAN;
  pgas_pow_ = NAN;
  gamma_max_ = NAN;
  epsatm = NAN;

  // Needed?
  density_floor_ = NAN;
  pressure_floor_ = NAN;

  // Active
  gamma_ = pin->GetReal("hydro", "gamma");
  fthr = pin -> GetReal("problem","fthr");
  fatm = pin -> GetReal("problem","fatm");
  k_adi = pin -> GetReal("hydro","k_adi");
  gamma_adi = pin -> GetReal("hydro","gamma");
  restrict_cs2 = pin->GetOrAddBoolean("hydro", "restrict_cs2", false);
  warn_unrestricted_cs2 = pin->GetOrAddBoolean(
    "hydro", "warn_unrestricted_cs2", false
  );

  // Reprimand
  using namespace EOS_Toolkit;
 //Get some EOS
  real_t max_eps = 10000.;
  real_t max_rho = 1e6;
  real_t adiab_ind = 1.0/(gamma_adi-1.0);
  eos = make_eos_idealgas(adiab_ind, max_eps, max_rho);

  //Set up atmosphere
  atmo_rho = fatm;
  atmo_eps = atmo_rho*k_adi;
  atmo_ye = 0.0;
  atmo_cut = atmo_rho * fthr;
  atmo_p = eos.at_rho_eps_ye(atmo_rho, atmo_eps, atmo_ye).press();
  atmo_cut_p = 0 * eos.at_rho_eps_ye(atmo_cut, atmo_eps, atmo_ye).press();

  //Primitive recovery parameters
  rho_strict = pin->GetOrAddReal("hydro", "rho_strict", atmo_rho);

  ye_lenient = false;
  max_iter = pin->GetOrAddReal("hydro", "max_iter", 10000);
  c2p_acc = pin->GetOrAddReal("hydro", "c2p_acc", 1e-12);
  max_b = 10.;
  max_z = pin -> GetOrAddReal("problem", "max_z", 100.0);
}


//----------------------------------------------------------------------------------------
// Variable inverter
void EquationOfState::ConservedToPrimitive(
  AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old,
  AthenaArray<Real> &prim,
  AthenaArray<Real> &bb_cc,
  Coordinates *pco,
  int il, int iu,
  int jl, int ju,
  int kl, int ku,
  int coarse_flag)
{
  MeshBlock * pmb = pmy_block_;
  Field * pf      = pmb->pfield;
  Hydro * ph      = pmb->phydro;

  GRDynamical * pco_gr;
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

  // Prepare variables for conversion -----------------------------------------
  Z4c * pz4c = pmy_block_->pz4c;

  Real detg_ceil = pow((pz4c->opt.chi_div_floor),-1.5);

  // quantities we have (sliced below)
  AT_N_sym adm_gamma_dd;
  AT_N_sym z4c_gamma_dd;
  AT_N_sca z4c_alpha;
  AT_N_vec z4c_beta_u;
  AT_N_sca z4c_chi;

  // quantities we will construct (on CC)
  AT_N_sca alpha_(   nn1);
  AT_N_sca chi_;  // only when required (se below)
  AT_N_sca rchi_;
  AT_N_vec beta_u_(  nn1);
  AT_N_sym gamma_dd_(nn1);

  // need to pull quantities from appropriate storage
  if (!coarse_flag)
  {
    adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    z4c_alpha.InitWithShallowSlice(   pz4c->storage.adm, Z4c::I_ADM_alpha);
    z4c_beta_u.InitWithShallowSlice(  pz4c->storage.adm, Z4c::I_ADM_betax);
  }
  else
  {
    chi_.NewAthenaTensor( nn1);  // additionally needed on coarse rep.
    rchi_.NewAthenaTensor(nn1);

    z4c_gamma_dd.InitWithShallowSlice(pz4c->coarse_u_, Z4c::I_Z4c_gxx);
    z4c_alpha.InitWithShallowSlice(   pz4c->coarse_u_, Z4c::I_Z4c_alpha);
    z4c_beta_u.InitWithShallowSlice(  pz4c->coarse_u_, Z4c::I_Z4c_betax);
    z4c_chi.InitWithShallowSlice(     pz4c->coarse_u_, Z4c::I_Z4c_chi);
  }

  // sanitize loop limits (coarse / fine auto-switched)
  const bool coarse_flag = false;
  int IL = il; int IU = iu;
  int JL = jl; int JU = ju;
  int KL = kl; int KU = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  // iterate over cells
  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    if (!coarse_flag)
    {
      pco_gr->GetGeometricFieldCC(gamma_dd_, adm_gamma_dd, k, j);
      pco_gr->GetGeometricFieldCC(alpha_,    z4c_alpha,    k, j);
      pco_gr->GetGeometricFieldCC(beta_u_,   z4c_beta_u,   k, j);
    }
    else
    {
      // coarse variables require also further oper.
      pco_gr->GetGeometricFieldCC(gamma_dd_, z4c_gamma_dd, k, j);
      pco_gr->GetGeometricFieldCC(alpha_,    z4c_alpha,    k, j);
      pco_gr->GetGeometricFieldCC(beta_u_,   z4c_beta_u,   k, j);
      pco_gr->GetGeometricFieldCC(chi_,      z4c_chi,      k, j);

      #pragma omp simd
      for (int i=IL; i<=IU; ++i)
      {
        rchi_(i) = 1. / chi_(i);
      }

      for(int a=0; a<NDIM; ++a)
      for(int b=a; b<NDIM; ++b)
      {
        #pragma omp simd
        for (int i=IL; i<=IU; ++i)
        {
          gamma_dd_(a,b,i) = gamma_dd_(a,b,i) * rchi_(i);
        }
      }

    }

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
      Real &rho  = prim(IDN,k,j,i);
      Real &pgas = prim(IPR,k,j,i);
      Real &uu1  = prim(IVX,k,j,i);
      Real &uu2  = prim(IVY,k,j,i);
      Real &uu3  = prim(IVZ,k,j,i);
      Real &bb1g = bb_cc(IB1,k,j,i);
      Real &bb2g = bb_cc(IB2,k,j,i);
      Real &bb3g = bb_cc(IB3,k,j,i);
      Real eps   = 0.0;
      Real w_lor = 1.0;
      Real dummy = 0.0;

      //AHF hydro excision --------------------------------------------------
      bool in_horizon = false;

      if(ph->opt_excision.horizon_based)
      {
        Real horizon_radius;
        for (auto pah_f : pmy_block_->pmy_mesh->pah_finder)
        {
          horizon_radius = pah_f->GetHorizonRadius();
          const Real r_2 = SQR(pco->x1v(i)) +
                            SQR(pco->x2v(j)) +
                            SQR(pco->x3v(k));
          if((r_2 < SQR(horizon_radius)) ||
             (alpha_(i) < ph->opt_excision.alpha_threshold))
          {
            in_horizon = true;
          }
        }
      }
      else
      {
        if(alpha_(i) < ph->opt_excision.alpha_threshold)
        {
          in_horizon = true;
        }
      }

      if(!in_horizon)
      {
        cons_vars_mhd evolved{Dg, taug, 0.0,
                              {S_1g, S_2g, S_3g}, 
                              {bb1g,bb2g,bb3g}};
        sm_tensor2_sym<real_t, 3, false, false> gtens(gamma_dd_(0,0,i),
                                                      gamma_dd_(0,1,i),
                                                      gamma_dd_(1,1,i),
                                                      gamma_dd_(0,2,i),
                                                      gamma_dd_(1,2,i),
                                                      gamma_dd_(2,2,i));
        sm_metric3 g_eos(gtens);
        prim_vars_mhd primitives;
        con2prim_mhd::report rep;
        //recover
        cv2pv(primitives, evolved, g_eos, rep);

        //check
        if (rep.failed())
        {
          uu1 = 0.0;
          uu2 = 0.0;
          uu3 = 0.0;
          rho = atmo_rho;
          pgas = k_adi*pow(atmo_rho,gamma_adi);
          PrimitiveToConservedSingle(pmb, prim,
                                      gamma_adi,
                                      bb_cc,
                                      alpha_,
                                      beta_u_,
                                      gamma_dd_,
                                      k, j, i,
                                      cons,
                                      pco,
                                      detg_ceil);

          if (pmb->phydro->c2p_status(k,j,i) == 0)
          {
            pmb->phydro->c2p_status(k,j,i) = 1;
          }
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
          if(std::isnan(Dg)   || std::isnan(taug) ||
             std::isnan(S_1g) || std::isnan(S_2g) ||
             std::isnan(S_3g))
          {
            uu1 = 0.0;
            uu2 = 0.0;
            uu3 = 0.0;
            rho = atmo_rho;
            pgas = k_adi*pow(atmo_rho,gamma_adi);
            PrimitiveToConservedSingle(pmb, prim,
                                      gamma_adi,
                                      bb_cc,
                                      alpha_,
                                      beta_u_,
                                      gamma_dd_,
                                      k, j, i,
                                      cons,
                                      pco,
                                      detg_ceil);
            printf("NAN after success");

            if (pmb->phydro->c2p_status(k,j,i) == 0)
            {
              pmb->phydro->c2p_status(k,j,i) = 2;
            }
          }

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
            PrimitiveToConservedSingle(pmb, prim,
                                      gamma_adi,
                                      bb_cc,
                                      alpha_,
                                      beta_u_,
                                      gamma_dd_,
                                      k, j, i,
                                      cons,
                                      pco,
                                      detg_ceil);

            if (pmb->phydro->c2p_status(k,j,i) == 0)
            {
              pmb->phydro->c2p_status(k,j,i) = 3;
            }

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
        // if(b_excision)
        // {
        //   bb_cc(0,k,j,i) = 0.0;
        //   bb_cc(1,k,j,i) = 0.0;
        //   bb_cc(2,k,j,i) = 0.0;
        //   pmy_block_->pfield->b1.x1f(k,j,i) = pmy_block_->pfield->b1.x2f(k,j,i) = 
        //   pmy_block_->pfield->b1.x3f(k,j,i) = pmy_block_->pfield->b1.x1f(k,j,i+1) = 
        //   pmy_block_->pfield->b1.x2f(k,j+1,i) = pmy_block_->pfield->b1.x3f(k+1,j,i) = 0.0;
        // }

        PrimitiveToConservedSingle(pmb,
                                   prim,
                                   gamma_adi,
                                   bb_cc,
                                   alpha_,
                                   beta_u_,
                                   gamma_dd_,
                                   k, j, i,
                                   cons,
                                   pco,
                                   detg_ceil);

        if (pmb->phydro->c2p_status(k,j,i) == 0)
        {
          pmb->phydro->c2p_status(k,j,i) = 4;
        }

      }

      ph->derived_ms(IX_LOR,k,j,i) = w_lor;
    }
  }

  // BD: TODO - probably better to move outside this
  AA derived_gs;
  DerivedQuantities(
    ph->derived_ms, derived_gs,
    cons, cons_scalar,
    prim, prim_scalar,
    bb_cc,
    pmb->pcoord,
    IL, IU, JL, JU, KL, KU,
    coarse_flag, skip_physical);
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
  int kl, int ku)
{

  MeshBlock* pmb = pmy_block_;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  Z4c * pz4c = pmb->pz4c;
  Real detg_ceil = pow((pz4c->opt.chi_div_floor),-1.5);

  AT_N_sym adm_gamma_dd;
  AT_N_sca adm_alpha;
  AT_N_vec adm_beta_u;
  AT_N_sym gamma_dd;
  AT_N_sca alpha;
  AT_N_vec beta_u;

  const int nn1 = pmb->nverts1;

  adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
  adm_alpha.InitWithShallowSlice(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  adm_beta_u.InitWithShallowSlice(  pz4c->storage.adm, Z4c::I_ADM_betax);
  gamma_dd.NewAthenaTensor(nn1);
  alpha.NewAthenaTensor(   nn1);
  beta_u.NewAthenaTensor(  nn1);

  // sanitize loop limits (coarse / fine auto-switched)
  const bool coarse_flag = false;
  int IL = il; int IU = iu;
  int JL = jl; int JU = ju;
  int KL = kl; int KU = ku;
  SanitizeLoopLimits(IL, IU, JL, JU, KL, KU, coarse_flag, pco);

  for (int k=KL; k<=KU; ++k)
  for (int j=JL; j<=JU; ++j)
  {
    pco_gr->GetGeometricFieldCC(gamma_dd, adm_gamma_dd, k, j);
    pco_gr->GetGeometricFieldCC(alpha,    adm_alpha,    k, j);
    pco_gr->GetGeometricFieldCC(beta_u,   adm_beta_u,   k, j);

    for (int i=IL; i<=IU; ++i)
    {
      PrimitiveToConservedSingle(pmb,
                                 prim,
                                 gamma_adi,
                                 bb_cc,
                                 alpha,
                                 beta_u,
                                 gamma_dd,
                                 k, j, i,
                                 cons,
                                 pco,
                                 detg_ceil);
    }
  }
}


// BD: TODO - eigenvalues, _not_ the speed; should be refactored / renamed
void EquationOfState::FastMagnetosonicSpeedsGR(
  Real rho_h, Real pgas, Real b_sq, Real vi, Real v2, Real alpha,
  Real betai, Real gammaii, Real *plambda_plus, Real *plambda_minus)
{
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

  if ((cs_sq > 1.0) && warn_unrestricted_cs2)
  {
    cs_sq = std::min(std::max(cs_sq, 0.0), 1.0);
  }

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

  // BD: TODO - should we use this or enforce zero?
  if (std::isnan(root_1) || std::isnan(root_2))
  {
    root_1 = 1.0;
    root_2 = 1.0;
  }

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

void EquationOfState::ApplyPrimitiveFloors(const int dir,
                                           AthenaArray<Real> &prim_l,
                                           AthenaArray<Real> &prim_r, int i)
{
  const int ix_o = dir == 1;
  if ((prim_l(IDN,i+ix_o) < atmo_cut)   ||
      (prim_l(IPR,i+ix_o) < atmo_cut_p) ||
      (prim_r(IDN,i) < atmo_cut)   ||
      (prim_r(IPR,i) < atmo_cut_p))
  {
    prim_l(IDN,i+ix_o) = atmo_rho;
    prim_l(IVX,i+ix_o) = 0.0;
    prim_l(IVY,i+ix_o) = 0.0;
    prim_l(IVZ,i+ix_o) = 0.0;
    prim_l(IPR,i+ix_o) = atmo_p;

    prim_r(IDN,i) = atmo_rho;
    prim_r(IVX,i) = 0.0;
    prim_r(IVY,i) = 0.0;
    prim_r(IVZ,i) = 0.0;
    prim_r(IPR,i) = atmo_p;
  }

  return;
}

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i)
{

  if(prim.GetDim4()==1)
  {
    if ((prim(IDN,i) < atmo_cut) ||
        (prim(IPR,i) < atmo_cut_p))
    {
      prim(IDN,i) = atmo_rho;
      prim(IVX,i) = 0.0;
      prim(IVY,i) = 0.0;
      prim(IVZ,i) = 0.0;
      prim(IPR,i) = atmo_p;
    }
  }
  else if(prim.GetDim4()==5)
  {
    if ((prim(IDN,k,j,i) < atmo_cut) ||
        (prim(IPR,k,j,i) < atmo_cut_p))
    {
      prim(IDN,k,j,i) = atmo_rho;
      prim(IVX,k,j,i) = 0.0;
      prim(IVY,k,j,i) = 0.0;
      prim(IVZ,k,j,i) = 0.0;
      prim(IPR,k,j,i) = atmo_p;
    }
  }

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
void PrimitiveToConservedSingle(
  MeshBlock * pmb,
  AthenaArray<Real> &prim,
  Real gamma_adi,
  AthenaArray<Real> &bb_cc,
  const AT_N_sca & alpha_,
  const AT_N_vec & beta_u_,
  const AT_N_sym & adm_gamma_dd_,
  int k, int j, int i,
  AthenaArray<Real> &cons,
  Coordinates *pco,
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

  // Apply floor to primitive variables as required.
  pmb->peos->ApplyPrimitiveFloors(prim, k, j, i);

  const Real &w_rho = prim(IDN,k,j,i);
  const Real &w_p   = prim(IPR,k,j,i);

  for(int a=0;a<NDIM;++a)
  {
    utilde_u(a) = prim(a+IVX,k,j,i);
  }

  // robust calc. of sqrt det
  const Real detgamma = Det3Metric(adm_gamma_dd_, i);
  Real sqrt_detgamma = (
    detgamma > 0
  ) ? std::sqrt(detgamma) : 1.;

  if (!std::isfinite(sqrt_detgamma))
  {
    sqrt_detgamma = 1.0;
  }
  else if (sqrt_detgamma > detg_ceil)
  {
    sqrt_detgamma = detg_ceil;
  }

  // robust calc. of Lorentz factor
  Real util_norm2 = 0;
  for(int a=0;a<NDIM;++a)
  for(int b=0;b<NDIM;++b)
  {
    util_norm2 += utilde_u(a)*utilde_u(b)*adm_gamma_dd_(a,b,i);
  }

  const Real W = (util_norm2 > -1.) ? std::sqrt(1.0 + util_norm2) : 1.;

  utilde_d.ZeroClear();

  for(int a=0;a<NDIM;++a)
  for(int b=0;b<NDIM;++b)
  {
    utilde_d(a) += utilde_u(b)*adm_gamma_dd_(a,b,i);
  }

  for(int a=0;a<NDIM;++a)
  {
    v_d(a) = utilde_d(a)/W;
  }
  for(int a=0;a<NDIM;++a)
  {
    v_u(a) = utilde_u(a)/W;
  }

  beta_d.ZeroClear();
  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      beta_d(a) += beta_u_(b,i)*adm_gamma_dd_(a,b,i);
    }
  }

  // Extract conserved quantities
  bb_u(0) = bb_cc(0,k,j,i)/sqrt_detgamma;
  bb_u(1) = bb_cc(1,k,j,i)/sqrt_detgamma;
  bb_u(2) = bb_cc(2,k,j,i)/sqrt_detgamma;
  Real b0_u = 0.0;

  for(int a=0;a<NDIM;++a)
  {
    b0_u += W*bb_u(a)*v_d(a)/alpha_(i);
  }
  for(int a=0;a<NDIM;++a)
  {
    bi_u(a) = (bb_u(a) + alpha_(i)*b0_u*W*(v_u(a) - beta_u_(a,i)/alpha_(i)))/W;
  }

  for(int a=0;a<NDIM;++a)
  {
    bi_d(a) = beta_d(a) * b0_u;
    for(int b=0;b<NDIM;++b)
    {
      bi_d(a) += adm_gamma_dd_(a,b,i)*bi_u(b);
    }
  }

  Real bsq = alpha_(i)*alpha_(i)*b0_u*b0_u/(W*W);
  for(int a=0;a<NDIM;++a)
  {
    for(int b=0;b<NDIM;++b)
    {
      bsq += bb_u(a)*bb_u(b)*adm_gamma_dd_(a,b,i)/(W*W);
    }
  }

  // Set conserved quantities
  const Real w_hrho = w_rho + gamma_adi/(gamma_adi-1.0) * w_p;

  cons(IDN,k,j,i) = sqrt_detgamma * w_rho*W;
  cons(IM1,k,j,i) = sqrt_detgamma * ((w_hrho+bsq)*SQR(W) * v_d(0) -
                                      alpha_(i)*b0_u*bi_d(0));
  cons(IM2,k,j,i) = sqrt_detgamma * ((w_hrho+bsq)*SQR(W) * v_d(1) -
                                      alpha_(i)*b0_u*bi_d(1));
  cons(IM3,k,j,i) = sqrt_detgamma * ((w_hrho+bsq)*SQR(W) * v_d(2) -
                                      alpha_(i)*b0_u*bi_d(2));
  cons(IEN,k,j,i) = sqrt_detgamma * (
    (w_hrho+bsq)*SQR(W) -
    w_rho*W -
    (w_p + bsq/2.0) -
    alpha_(i)*alpha_(i)*b0_u*b0_u
  );

}

}
