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
#include "reprimand/hydro_atmo.h"
#include "reprimand/hydro_cons.h"
#include "reprimand/hydro_prim.h"
#include "reprimand/smtensor.h"
#include "reprimand/unitconv.h"



namespace {

// for readability
const int D = NDIM + 1;
const int N = NDIM;

typedef AthenaTensor<Real, TensorSymm::NONE, N, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, N, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, N, 2> AT_N_sym;

// Declarations
inline static void PrimitiveToConservedSingle(
  MeshBlock * pmb,
  AthenaArray<Real> &prim,
  Real gamma_adi,
  AT_N_sym const & gamma_dd,
  int k, int j, int i,
  AthenaArray<Real> &cons,
  Coordinates *pco);

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
  alpha_excision = pin->GetOrAddReal("hydro", "alpha_excision", 0.0);

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
  Hydro* phydro = pmb->phydro;

  int nn1;

  // Debug:
  if (false) {
    pmb->pz4c->Z4cToADM(pmb->pz4c->storage.u, pmb->pz4c->storage.adm);
    pmb->phydro->Hydro_IdealEoS_Cons2Prim(gamma_adi, cons, prim,
                                          const_cast<AthenaArray<Real> &>(prim_old),
                                          il, iu, jl, ju, kl, ku);
    // const_cast<AthenaArray<Real> &>(prim_old) = prim;
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
  AT_N_sym adm_gamma_dd;
  AT_N_sym z4c_gamma_dd;
  AT_N_sca z4c_alpha;
  AT_N_vec z4c_beta_u;
  AT_N_sca z4c_chi;

  // quantities we will construct (on CC)
  AT_N_sca alpha;
  AT_N_sca chi;
  AT_N_sca rchi;
  AT_N_vec beta_u;
  AT_N_sym gamma_dd;

  alpha.NewAthenaTensor(   nn1);
  beta_u.NewAthenaTensor(  nn1);
  gamma_dd.NewAthenaTensor(nn1);

  // Prepare inverse + det outside Reprimand
  AT_N_sym gamma_uu_(    nn1);
  AT_N_sca det_gamma_(   nn1);
  AT_N_sca oo_det_gamma_(nn1);

  // need to pull quantities from appropriate storage
  if (!coarse_flag)
  {
    adm_gamma_dd.InitWithShallowSlice(pz4c->storage.adm, Z4c::I_ADM_gxx);
    // Not great with the notation...
    z4c_alpha.InitWithShallowSlice(   pz4c->storage.adm, Z4c::I_ADM_alpha);
    z4c_beta_u.InitWithShallowSlice(  pz4c->storage.adm, Z4c::I_ADM_betax);
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

    #pragma omp simd
    for (int i=IL; i<=IU; ++i)
    {
      det_gamma_(i) = LinearAlgebra::Det3Metric(gamma_dd, i);
      oo_det_gamma_(i) = 1. / det_gamma_(i);
      LinearAlgebra::Inv3Metric(oo_det_gamma_, gamma_dd, gamma_uu_, i);
    }

    // do actual variable conversion ------------------------------------------
    #pragma omp simd
    for (int i=IL; i<=IU; ++i)
    {
      // Extract conserved quantities
      Real &Dg =   cons(IDN,k,j,i);
      Real &taug = cons(IEN,k,j,i);
      Real &S_1g = cons(IVX,k,j,i);
      Real &S_2g = cons(IVY,k,j,i);
      Real &S_3g = cons(IVZ,k,j,i);

      //Extract prims
      Real &w_rho = prim(IDN,k,j,i);
      Real &w_p   = prim(IPR,k,j,i);
      Real &uu1   = prim(IVX,k,j,i);
      Real &uu2   = prim(IVY,k,j,i);
      Real &uu3   = prim(IVZ,k,j,i);
      Real eps    = 0.0;
      Real W      = 1.0;
      Real dummy  = 0.0;

      // cons->prim requires atmosphere reset
      phydro->q_reset_mask(k,j,i) = (
        (alpha(i) <= alpha_excision) ||
        std::isnan(Dg)   ||
        std::isnan(taug) ||
        std::isnan(S_1g) ||
        std::isnan(S_2g) ||
        std::isnan(S_3g) ||
        (det_gamma_(i) < 0.) ||
        (Dg < 0.)
      );

      if (~phydro->q_reset_mask(k,j,i))
      {
        sm_tensor1<real_t, 3, false>  S_dg(S_1g,
                                           S_2g,
                                           S_3g);

        cons_vars_mhd evolved{Dg, taug, 0.0, S_dg, {0., 0., 0.}};

        sm_tensor2_sym<real_t, 3, false, false> rpgamma_dd(
          gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(1,1,i),
          gamma_dd(0,2,i), gamma_dd(1,2,i), gamma_dd(2,2,i));

        sm_tensor2_sym<real_t, 3, true, true> rpgamma_uu(
          gamma_uu_(0,0,i), gamma_uu_(0,1,i), gamma_uu_(1,1,i),
          gamma_uu_(0,2,i), gamma_uu_(1,2,i), gamma_uu_(2,2,i));


        sm_metric3 g_eos(rpgamma_dd, rpgamma_uu, det_gamma_(i));
        prim_vars_mhd primitives;
        con2prim_mhd::report rep;
        //recover
        cv2pv(primitives, evolved, g_eos, rep);

        phydro->q_reset_mask(k,j,i) = (
          phydro->q_reset_mask(k,j,i)  || rep.failed()
        );

        if (~phydro->q_reset_mask(k,j,i))
        {
          primitives.scatter(w_rho, eps, dummy, w_p,
                             uu1, uu2, uu3, W,
                             dummy, dummy, dummy,
                             dummy, dummy, dummy);

          // AT_N_sca W_(        iu+1);
          // AT_N_vec V_(        iu+1);
          // V_(0,i) = uu1;
          // V_(1,i) = uu2;
          // V_(2,i) = uu3;

          // const Real norm2_V = LinearAlgebra::InnerProductSlicedVec3Metric(
          //   V_, gamma_uu_, i
          // );
          // const Real W_c = 1. / std::sqrt(1.-norm2_V);

          uu1 *= W;
          uu2 *= W;
          uu3 *= W;

          // check recovered do not need cut
          phydro->q_reset_mask(k,j,i) = (
            phydro->q_reset_mask(k,j,i)  ||
            (w_rho < atmo_cut) ||
            // // Below are additional checks (Use with PTCS disabled below)
            (w_p < atmo_cut_p)   ||
            (W < 1.)
          );
        }
      }

      // check if atmo. reset required
      if (phydro->q_reset_mask(k,j,i))
      {
        w_p   = atmo_p;
        w_rho = atmo_rho;
        uu1 = 0.0;
        uu2 = 0.0;
        uu3 = 0.0;

        // PTCS: disabled
        PrimitiveToConservedSingle(pmb, prim, gamma_adi, gamma_dd,
                                   k, j, i, cons, pco);

        /*
        const Real Eos_Gamma_ratio = gamma_adi / (gamma_adi - 1.0);
        const Real w_hrho = w_rho + Eos_Gamma_ratio * w_p;

        // conservatives
        const Real W_ = 1.;
        Dg   = det_gamma_(i) * w_rho * W_;
        S_1g = 0;
        S_2g = 0;
        S_3g = 0;
        taug = det_gamma_(i) * (w_hrho * SQR(W_) -
                                w_rho  * W_ -
                                w_p);
        */
      }

    }
  }

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
  // Make this more readable
  MeshBlock* pmb = pmy_block_;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  Z4c * pz4c = pmb->pz4c;

  // Debug:
  if (false)
  {
    pmb->phydro->Hydro_IdealEoS_Prim2Cons(gamma_adi, prim, cons,
                                          il, iu, jl, ju, kl, ku);
    return;
  }

  // Require only ADM 3-metric and no gauge.
  AT_N_sym sl_g_dd( pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sym gamma_dd(pmb->nverts1);

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
    pco_gr->GetGeometricFieldCC(gamma_dd, sl_g_dd, k, j);
    for (int i=IL; i<=IU; ++i)
    {
      PrimitiveToConservedSingle(pmb, prim, gamma_adi, gamma_dd,
                                 k, j, i, cons, pco);
    }
  }
}

void EquationOfState::SoundSpeedsGR(
  Real rho_h, Real pgas,
  Real vi, Real v2, Real alpha,
  Real betai, Real gammaii,
  Real *plambda_plus, Real *plambda_minus)
{

  // Calculate comoving sound speed
  const Real cs_sq = (pgas > 0) ? gamma_adi * pgas / rho_h : 0;
  const Real cs = std::sqrt(cs_sq);

  const Real fac_sqrt = cs * std::sqrt(
    (1.0-v2)*(gammaii*(1.0-v2*cs_sq)-vi*vi*(1.0-cs_sq))
  );

  const Real fac_alpha = alpha / (1.0-v2*cs_sq);

  Real root_1 = fac_alpha * (vi*(1.0-cs_sq) + fac_sqrt) - betai;
  Real root_2 = fac_alpha * (vi*(1.0-cs_sq) - fac_sqrt) - betai;

  // collapse correction
  if (std::isnan(root_1) || std::isnan(root_2))
  {
    root_1 = 1.0;
    root_2 = 1.0;
  }

  if (root_1 > root_2) {
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

void EquationOfState::ForcePrimitiveFloor(AthenaArray<Real> &prim, int k, int j, int i)
{


  if(prim.GetDim4()==1)
  {
    prim(IDN,i) = atmo_rho;
    prim(IVX,i) = 0.0;
    prim(IVY,i) = 0.0;
    prim(IVZ,i) = 0.0;
    prim(IPR,i) = atmo_p;
  }
  else if(prim.GetDim4()==5)
  {
    prim(IDN,k,j,i) = atmo_rho;
    prim(IVX,k,j,i) = 0.0;
    prim(IVY,k,j,i) = 0.0;
    prim(IVZ,k,j,i) = 0.0;
    prim(IPR,k,j,i) = atmo_p;
  }

  return;
}

bool EquationOfState::RequirePrimitiveFloor(
  const AthenaArray<Real> &prim, int k, int j, int i)
{

  bool rpf = false;

  if(prim.GetDim4()==1)
  {
    rpf = ((prim(IDN,i) < atmo_cut) ||
           (prim(IPR,i) < atmo_cut_p));
  }
  else if(prim.GetDim4()==5)
  {
    rpf = ((prim(IDN,k,j,i) < atmo_cut) ||
           (prim(IPR,k,j,i) < atmo_cut_p));
  }

  return rpf;
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
  MeshBlock * pmb,
  AthenaArray<Real> &prim, Real gamma_adi,
  AT_N_sym const & adm_gamma_dd_,
  int k, int j, int i,
  AthenaArray<Real> &cons, Coordinates *pco)
{
  using namespace LinearAlgebra;

  AthenaArray<Real> utilde_u;  // primitive gamma^i_a u^a
  AthenaArray<Real> utilde_d;

  utilde_u.NewAthenaArray(3);
  utilde_d.NewAthenaArray(3);

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
  const Real sqrt_detgamma = (
    detgamma > 0
  ) ? std::sqrt(detgamma) : 1.;

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

  // Set conserved quantities
  const Real w_hrho = w_rho + gamma_adi/(gamma_adi-1.0) * w_p;

  // Ddg, taudg, S_1dg, S_2dg, S_3dg
  cons(IDN,k,j,i) = sqrt_detgamma * w_rho * W;
  cons(IEN,k,j,i) = sqrt_detgamma * (w_hrho * SQR(W) - w_rho*W - w_p);
  cons(IM1,k,j,i) = sqrt_detgamma * w_hrho * W * utilde_d(0);
  cons(IM2,k,j,i) = sqrt_detgamma * w_hrho * W * utilde_d(1);
  cons(IM3,k,j,i) = sqrt_detgamma * w_hrho * W * utilde_d(2);

  utilde_u.DeleteAthenaArray();
  utilde_d.DeleteAthenaArray();
}

}