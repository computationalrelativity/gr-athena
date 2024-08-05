//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file llf_rel_no_transform.cpp
//  \brief Implements local Lax-Friedrichs Riemann solver for relativistic hydrodynamics
//  in pure GR.

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/linear_algebra.hpp"
#include "../../../utils/interp_intergrid.hpp"
#include "../../../athena_aliases.hpp"
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock

#include "../../../z4c/ahf.hpp"

//----------------------------------------------------------------------------------------
using namespace gra::aliases;
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Riemann solver
// Inputs:
//   kl,ku,jl,ju,il,iu: lower and upper x1-, x2-, and x3-indices
//   ivx: type of interface (IVX for x1, IVY for x2, IVZ for x3)
//   bb: 3D array of normal magnetic fields (not used)
//   prim_l,prim_r: 3D arrays of left and right primitive states
// Outputs:
//   flux: 3D array of hydrodynamical fluxes across interfaces
//   ey,ez: 3D arrays of magnetic fluxes (electric fields) across interfaces (not used)
// Notes:
//   implements LLF algorithm similar to that of fluxcalc() in step_ch.c in Harm
//   cf. LLFNonTransforming() in llf_rel.cpp
// Here we use the D, S, tau variable choice for conservatives, and assume a dynamically evolving spacetime
// so a factor of sqrt(detgamma) is included

// BD: TODO - refactor this.

void Hydro::RiemannSolver(
  const int k, const int j,
  const int il, const int iu,
  const int ivx,
  const AthenaArray<Real> &bb,
  AthenaArray<Real> &prim_l,
  AthenaArray<Real> &prim_r,
  AthenaArray<Real> &flux,
  AthenaArray<Real> &ey,
  AthenaArray<Real> &ez,
  AthenaArray<Real> &wct,
  const AthenaArray<Real> &dxw)
{
  using namespace LinearAlgebra;

  MeshBlock * pmb = pmy_block;
  EquationOfState * peos = pmb->peos;

  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int a,b;
  const int nn1 = pmy_block->nverts1;  // utilize the verts

  // Extract ratio of specific heats
#if USETM
  const Real mb = pmb->peos->GetEOS().GetBaryonMass();
#else
  const Real Gamma = pmb->peos->GetGamma();
  const Real Eos_Gamma_ratio = Gamma / (Gamma - 1.0);
#endif

  // perform variable resampling when required
  Z4c * pz4c = pmb->pz4c;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym sl_adm_gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca sl_adm_alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec sl_adm_beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

  // 1d slices ----------------------------------------------------------------
  AT_N_sca w_rho_l_(prim_l, IDN);
  AT_N_sca w_rho_r_(prim_r, IDN);
  AT_N_sca w_p_l_(  prim_l, IPR);
  AT_N_sca w_p_r_(  prim_r, IPR);

  AT_N_vec w_util_u_l_(prim_l, IVX);
  AT_N_vec w_util_u_r_(prim_r, IVX);

  // Reconstruction to FC -----------------------------------------------------
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);
  pco_gr->GetGeometricFieldFC(gamma_dd_, sl_adm_gamma_dd, ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(alpha_,    sl_adm_alpha,    ivx-1, k, j);
  pco_gr->GetGeometricFieldFC(beta_u_,   sl_adm_beta_u,   ivx-1, k, j);

  // ==========================================================================

  Real dt = pmb->pmy_mesh->dt;
  Real alpha_excision = peos->alpha_excision;
  bool horizon_excision = peos->horizon_excision;

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Wlor_l, Wlor_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> u0_l, u0_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> v2_l, v2_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> b0_u_l, b0_u_r, bsq_l, bsq_r;

  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_u_l, v_u_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_d_l, v_d_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> ucon_l, ucon_r;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> bb_l, bb_r, bi_u_l, bi_u_r,
    bi_d_l, bi_d_r;

  Wlor_l.NewAthenaTensor(nn1);
  Wlor_r.NewAthenaTensor(nn1);
  u0_l.NewAthenaTensor(nn1);
  u0_r.NewAthenaTensor(nn1);
  v2_l.NewAthenaTensor(nn1);
  v2_r.NewAthenaTensor(nn1);
  v_u_l.NewAthenaTensor(nn1);
  v_u_r.NewAthenaTensor(nn1);
  v_d_l.NewAthenaTensor(nn1);
  v_d_r.NewAthenaTensor(nn1);
  ucon_l.NewAthenaTensor(nn1);
  ucon_r.NewAthenaTensor(nn1);
  b0_u_l.NewAthenaTensor(nn1);
  b0_u_r.NewAthenaTensor(nn1);
  bsq_l.NewAthenaTensor(nn1);
  bsq_r.NewAthenaTensor(nn1);
  bb_l.NewAthenaTensor(nn1);
  bb_r.NewAthenaTensor(nn1); 
  bi_u_l.NewAthenaTensor(nn1);
  bi_u_r.NewAthenaTensor(nn1); 
  bi_d_l.NewAthenaTensor(nn1);
  bi_d_r.NewAthenaTensor(nn1);


  // =============================================================
  // Prepare determinant-like
  #pragma omp simd
  for (int i = il; i <= iu; ++i)
  {
    detgamma_(i)      = Det3Metric(gamma_dd_, i);

    sqrt_detgamma_(i) = std::sqrt(detgamma_(i));
    oo_detgamma_(i)   = 1. / detgamma_(i);
  }

  Inv3Metric(oo_detgamma_, gamma_dd_, gamma_uu_, il, iu);
  // =============================================================


  // deal with excision -------------------------------------------------------
  auto excise = [&](const int i)
  {
    // set flat space if the interpolated det is negative and inside
    // horizon (either in ahf or lapse below excision value)
    std::cout << "Set flat space" << "\n";
    gamma_dd_(0, 0, i) = 1.0;
    gamma_dd_(1, 1, i) = 1.0;
    gamma_dd_(2, 2, i) = 1.0;
    gamma_dd_(0, 1, i) = 0.0;
    gamma_dd_(0, 2, i) = 0.0;
    gamma_dd_(1, 2, i) = 0.0;
    gamma_uu_(0, 0, i) = 1.0;
    gamma_uu_(1, 1, i) = 1.0;
    gamma_uu_(2, 2, i) = 1.0;
    gamma_uu_(0, 1, i) = 0.0;
    gamma_uu_(0, 2, i) = 0.0;
    gamma_uu_(1, 2, i) = 0.0;
    detgamma_(i) = 1.0;
  };

  if (horizon_excision)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      // if ahf enabled set flat space if in horizon
      // TODO: read in centre of each horizon and shift origin - not needed for
      // now if collapse is at origin
      Real horizon_radius;
      for (auto pah_f : pmy_block->pmy_mesh->pah_finder)
      {
        horizon_radius = pah_f->GetHorizonRadius();
        const Real R2 = (
          SQR(pco_gr->x1f(i)) + SQR(pco_gr->x2v(j)) + SQR(pco_gr->x3v(k))
        );

        if ((R2 < SQR(horizon_radius)) || (alpha_(i) < alpha_excision))
        {
          excise(i);
        }
      }
    }
  }
  else if (alpha_excision > 0)  // by default disabled (i.e. 0)
  {
    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      if (alpha_(i) < alpha_excision)
      {
        excise(i);
      }
    }
  }
  // --------------------------------------------------------------------------



    #pragma omp simd
    for (int i = il; i <= iu; ++i)
    {
      switch (ivx)
      {
        case IVX:
        {
          bb_l(0,i) = bb(k,j,i) * oo_detgamma_(i);
          bb_l(1,i) = prim_l(IBY,i) * oo_detgamma_(i);
          bb_l(2,i) = prim_l(IBZ,i) * oo_detgamma_(i);
          bb_r(0,i) = bb(k,j,i) * oo_detgamma_(i);
          bb_r(1,i) = prim_r(IBY,i) * oo_detgamma_(i);
          bb_r(2,i) = prim_r(IBZ,i) * oo_detgamma_(i);
          break;
        }
        case IVY:
        {
          bb_l(1,i) = bb(k,j,i) * oo_detgamma_(i);
          bb_l(2,i) = prim_l(IBY,i) * oo_detgamma_(i);
          bb_l(0,i) = prim_l(IBZ,i) * oo_detgamma_(i);
          bb_r(1,i) = bb(k,j,i) * oo_detgamma_(i);
          bb_r(2,i) = prim_r(IBY,i) * oo_detgamma_(i);
          bb_r(0,i) = prim_r(IBZ,i) * oo_detgamma_(i);
          break;
        }
        case IVZ:
        {
            bb_l(2,i) = bb(k,j,i) * oo_detgamma_(i);
            bb_l(0,i) = prim_l(IBY,i) * oo_detgamma_(i);
            bb_l(1,i) = prim_l(IBZ,i) * oo_detgamma_(i);
            bb_r(2,i) = bb(k,j,i) * oo_detgamma_(i);
            bb_r(0,i) = prim_r(IBY,i) * oo_detgamma_(i);
            bb_r(1,i) = prim_r(IBZ,i) * oo_detgamma_(i);
            break;

        }
        default:
        {
          assert(false);
        }
      }
    }

  // --------------------------------------------------------------------------

    beta_d_.ZeroClear();
    for(a=0;a<NDIM;++a)
    {
      for(b=0;b<NDIM;++b)
      {
        #pragma omp simd
        for(int i = il; i <= iu; ++i) 
        {
          beta_d_(a,i) += gamma_dd_(a,b,i)*beta_u_(b,i);
        }
      }
    }


// Everything here is undensitised.

    Wlor_l.ZeroClear();
    for(a=0;a<NDIM;++a)
    {
      for(b=0;b<NDIM;++b)
      {
        #pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          Wlor_l(i) += w_util_u_l_(a,i)*w_util_u_l_(b,i)*gamma_dd_(a,b,i);
        }
      }
     }
     #pragma omp simd
     for (int i = il; i <= iu; ++i)
     {
            Wlor_l(i) = std::sqrt(1.0+Wlor_l(i));
     }
      Wlor_r.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  Wlor_r(i) += w_util_u_r_(a,i)*w_util_u_r_(b,i)*gamma_dd_(a,b,i);
              }
           }
       }
        #pragma omp simd
      for (int i = il; i <= iu; ++i){
            Wlor_r(i) = std::sqrt(1.0+Wlor_r(i));
       }
      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
             v_u_l(a,i) = w_util_u_l_(a,i)/Wlor_l(i);
          }
      }
      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
             v_u_r(a,i) = w_util_u_r_(a,i)/Wlor_r(i);
          }
      }

      v2_l.ZeroClear();
      for(a=0;a<NDIM;++a){
            for(b=0;b<NDIM;++b){
               #pragma omp simd
                   for (int i = il; i <= iu; ++i){
                       v2_l(i) += gamma_dd_(a,b,i)*v_u_l(a,i)*v_u_l(b,i);
                   }
             }
       }
      v2_r.ZeroClear();
      for(a=0;a<NDIM;++a){
            for(b=0;b<NDIM;++b){
               #pragma omp simd
                   for (int i = il; i <= iu; ++i){
                       v2_r(i) += gamma_dd_(a,b,i)*v_u_r(a,i)*v_u_r(b,i);
                   }
             }
       }

     v_d_l.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
      #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  v_d_l(a,i) += v_u_l(b,i)*gamma_dd_(a,b,i);
              }
          }
      }
     v_d_r.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  v_d_r(a,i) += v_u_r(b,i)*gamma_dd_(a,b,i);
              }
          }
      }

    b0_u_l.ZeroClear();
    for(a=0;a<NDIM;++a){
       #pragma omp simd
       for (int i = il; i <= iu; ++i){
          b0_u_l(i) += Wlor_l(i)*bb_l(a,i)*v_d_l(a,i)/alpha_(i);  
       } 
     }
    b0_u_r.ZeroClear();
    for(a=0;a<NDIM;++a){
       #pragma omp simd
       for (int i = il; i <= iu; ++i){
          b0_u_r(i) += Wlor_r(i)*bb_r(a,i)*v_d_r(a,i)/alpha_(i);  
       } 
     }
    for(a=0;a<NDIM;++a){
       #pragma omp simd
       for (int i = il; i <= iu; ++i){
          bi_u_l(a,i) = (bb_l(a,i) + alpha_(i)*b0_u_l(i)*Wlor_l(i)*(v_u_l(a,i) - beta_u_(a,i)/alpha_(i)))/Wlor_l(i);
       }
     }
    for(a=0;a<NDIM;++a){
       #pragma omp simd
       for (int i = il; i <= iu; ++i){
          bi_u_r(a,i) = (bb_r(a,i) + alpha_(i)*b0_u_r(i)*Wlor_r(i)*(v_u_r(a,i) - beta_u_(a,i)/alpha_(i)))/Wlor_r(i);
       }
     }
     #pragma omp simd
     for (int i = il; i <= iu; ++i){
       bsq_l(i) = alpha_(i)*alpha_(i)*b0_u_l(i)*b0_u_l(i)/(Wlor_l(i)*Wlor_l(i));
     }
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
         bsq_l(i) += bb_l(a,i)*bb_l(b,i)*gamma_dd_(a,b,i)/(Wlor_l(i)*Wlor_l(i));
      }
      }
      } 
     #pragma omp simd
     for (int i = il; i <= iu; ++i){
       bsq_r(i) = alpha_(i)*alpha_(i)*b0_u_r(i)*b0_u_r(i)/(Wlor_r(i)*Wlor_r(i));
     }
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
         bsq_r(i) += bb_r(a,i)*bb_r(b,i)*gamma_dd_(a,b,i)/(Wlor_r(i)*Wlor_r(i));
      }
      }
      }
       for(a=0;a<NDIM;++a){
#pragma omp simd
        for (int i = il; i <= iu; ++i){
         bi_d_l(a,i) = beta_d_(a,i) * b0_u_l(i);
        }
        for(b=0;b<NDIM;++b){
#pragma omp simd
         for (int i = il; i <= iu; ++i){
          bi_d_l(a,i) += gamma_dd_(a,b,i)*bi_u_l(b,i);
         }
        }
       }
       for(a=0;a<NDIM;++a){
#pragma omp simd
        for (int i = il; i <= iu; ++i){
         bi_d_r(a,i) = beta_d_(a,i) * b0_u_r(i);
        }
        for(b=0;b<NDIM;++b){
#pragma omp simd
         for (int i = il; i <= iu; ++i){
          bi_d_r(a,i) += gamma_dd_(a,b,i)*bi_u_r(b,i);
         }
        }
       }
                     
     w_util_d_l_.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  w_util_d_l_(a,i) += w_util_u_l_(b,i)*gamma_dd_(a,b,i);
              }
          }
      }
     w_util_d_r_.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  w_util_d_r_(a,i) += w_util_u_r_(b,i)*gamma_dd_(a,b,i);
              }
          }
      }
       #pragma omp simd
      for (int i = il; i <= iu; ++i){
      u0_l(i) = Wlor_l(i)/alpha_(i);
      u0_r(i) = Wlor_r(i)/alpha_(i);
      }
/*      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
            ucon_u_l(a,i) = w_util_u_l_(a,i) + Wlor_l(i) * beta_u_(a,i)/alpha_(i);
            ucon_u_r(a,i) = w_util_u_r_(a,i) + Wlor_r(i) * beta_u_(a,i)/alpha_(i);
          }
       }
*/

      // copy from below!
      #pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
      // Calculate wavespeeds in left state NB EOS specific
#if USETM
      // If using the PrimitiveSolver framework, get the number density
      // and temperature to help calculate enthalpy.
      Real nl = w_rho_l_(i) / mb;
      Real nr = w_rho_r_(i) / mb;
      // FIXME: Generalize to work with EOSes accepting particle fractions.
      Real Yl[MAX_SPECIES] = { 0.0 };  // Should we worry about r vs l here?
      Real Yr[MAX_SPECIES] = { 0.0 };

      // PH TODO scalars should be passed in?
#ifdef DBG_COMBINED_HYDPA
      for (int n = 0; n < NSCALARS; n++)
      {
        Yl[n] = pmy_block->pscalars->rl_(n, i);
        Yr[n] = pmy_block->pscalars->rr_(n, i);
      }
#else
      for (int n = 0; n < NSCALARS; n++)
      {
        Yr[n] = pmy_block->pscalars->r(n, k, j, i);
      }
      switch (ivx)
      {
      case IVX:
              for (int n = 0; n < NSCALARS; n++)
              {
                  Yl[n] = pmy_block->pscalars->r(n, k, j, i - 1);
              }
              break;
      case IVY:
              for (int n = 0; n < NSCALARS; n++)
              {
                  Yl[n] = pmy_block->pscalars->r(n, k, j - 1, i);
              }
              break;
      case IVZ:
              for (int n = 0; n < NSCALARS; n++)
              {
                  Yl[n] = pmy_block->pscalars->r(n, k - 1, j, i);
              }
              break;
      }
#endif // DBG_COMBINED_HYDPA

      Real Tl =
        pmy_block->peos->GetEOS().GetTemperatureFromP(nl, w_p_l_(i), Yl);
      Real Tr =
        pmy_block->peos->GetEOS().GetTemperatureFromP(nr, w_p_r_(i), Yr);
      w_hrho_l_(i) =
        w_rho_l_(i) * pmy_block->peos->GetEOS().GetEnthalpy(nl, Tl, Yl);
      w_hrho_r_(i) =
        w_rho_r_(i) * pmy_block->peos->GetEOS().GetEnthalpy(nr, Tr, Yr);

      // Calculate the wave speeds
      pmy_block->peos->FastMagnetosonicSpeedsGR(
        nl, Tl, bsq_l(i), v_u_l(ivx - 1, i), v2_l(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_l(i),
        &lambda_m_l(i), Yl);
      pmy_block->peos->FastMagnetosonicSpeedsGR(
        nr, Tr, bsq_r(i), v_u_r(ivx - 1, i), v2_r(i), alpha_(i),
        beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i), &lambda_p_r(i),
        &lambda_m_r(i), Yr);
#else
      w_hrho_l_(i) = w_rho_l_(i) + Eos_Gamma_ratio * w_p_l_(i);
      w_hrho_r_(i) = w_rho_r_(i) + Eos_Gamma_ratio * w_p_r_(i);

      pmy_block->peos->FastMagnetosonicSpeedsGR(
        w_hrho_l_(i), w_p_l_(i), bsq_l(i), v_u_l(ivx - 1, i), v2_l(i),
        alpha_(i), beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i),
        &lambda_p_l(i), &lambda_m_l(i));
      pmy_block->peos->FastMagnetosonicSpeedsGR(
        w_hrho_r_(i), w_p_r_(i), bsq_r(i), v_u_r(ivx - 1, i), v2_r(i),
        alpha_(i), beta_u_(ivx - 1, i), gamma_uu_(ivx - 1, ivx - 1, i),
        &lambda_p_r(i), &lambda_m_r(i));
#endif
      }

      // Calculate extremal wavespeed
#pragma omp simd
      for (int i = il; i <= iu; ++i){
        Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
        Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
        lambda(i) = std::max(lambda_r, -lambda_l);
        }

        // Calculate conserved quantities in L region including factor of sqrt(detgamma)
         #pragma omp simd
      for (int i = il; i <= iu; ++i){

//TODO cons definitions change for Magfield
//TODO add in cons B field (for flux calc at end)
        // D = rho * gamma_lorentz
        cons_l_(IDN,i) = w_rho_l_(i)*Wlor_l(i)*sqrt_detgamma_(i);
        // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
        cons_l_(IEN,i) = ((w_hrho_l_(i)+bsq_l(i)) * SQR(Wlor_l(i)) - w_rho_l_(i)*Wlor_l(i) - w_p_l_(i) - bsq_l(i)/2.0 - alpha_(i)*alpha_(i)*b0_u_l(i)*b0_u_l(i))*sqrt_detgamma_(i);
        // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
//        cons_l_(IVX,i) = w_hrho_l_ * gamma_l * ucov_l[1]*std::sqrt(detgamma);
//        cons_l_(IVY,i) = w_hrho_l_ * gamma_l * ucov_l[2]*std::sqrt(detgamma);
//        cons_l_(IVZ,i) = w_hrho_l_ * gamma_l * ucov_l[3]*std::sqrt(detgamma);
//NB TODO double check velocity has chenged here (also in right state)
        cons_l_(IVX,i) = ((w_hrho_l_(i)+bsq_l(i)) * Wlor_l(i) * w_util_d_l_(0,i) - alpha_(i)*b0_u_l(i)*bi_d_l(0,i)  )*sqrt_detgamma_(i);
        cons_l_(IVY,i) = ((w_hrho_l_(i)+bsq_l(i)) * Wlor_l(i) * w_util_d_l_(1,i) - alpha_(i)*b0_u_l(i)*bi_d_l(1,i)  )*sqrt_detgamma_(i);
        cons_l_(IVZ,i) = ((w_hrho_l_(i)+bsq_l(i)) * Wlor_l(i) * w_util_d_l_(2,i) - alpha_(i)*b0_u_l(i)*bi_d_l(2,i)  )*sqrt_detgamma_(i);
        cons_l_(IBY,i) = sqrt_detgamma_(i)*bb_l(ivy-1,i);
        cons_l_(IBZ,i) = sqrt_detgamma_(i)*bb_l(ivz-1,i);
        // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
        // D flux: D(v^i - beta^i/alpha)
        //
         flux_l_(IDN,i) = cons_l_(IDN,i)*alpha_(i)*(v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i));

        // tau flux: alpha_(S^i - Dv^i) - beta^i tau
          flux_l_(IEN,i) = cons_l_(IEN,i) * alpha_(i) * (v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) + alpha_(i)*sqrt_detgamma_(i)*((w_p_l_(i)+ bsq_l(i)/2.0)*v_u_l(ivx-1,i) - alpha_(i)*b0_u_l(i)*bb_l(ivx-1,i)/Wlor_l(i));
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_l_(IVX,i) = cons_l_(IVX,i) * alpha_(i) * (v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_l(0,i)*bb_l(ivx-1,i)/Wlor_l(i);      
        flux_l_(IVY,i) = cons_l_(IVY,i) * alpha_(i) * (v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_l(1,i)*bb_l(ivx-1,i)/Wlor_l(i);      
        flux_l_(IVZ,i) = cons_l_(IVZ,i) * alpha_(i) * (v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_l(2,i)*bb_l(ivx-1,i)/Wlor_l(i);      
        flux_l_(ivx,i) += (w_p_l_(i)+bsq_l(i)/2.0)*alpha_(i)*sqrt_detgamma_(i);
        flux_l_(IBY,i) = alpha_(i)*(cons_l_(IBY,i)*(v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - bb_l(ivx-1,i)*sqrt_detgamma_(i)*(v_u_l(ivy-1,i) - beta_u_(ivy-1,i)/alpha_(i))  );
        flux_l_(IBZ,i) = alpha_(i)*(cons_l_(IBZ,i)*(v_u_l(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - bb_l(ivx-1,i)*sqrt_detgamma_(i)*(v_u_l(ivz-1,i) - beta_u_(ivz-1,i)/alpha_(i))  ); //check these indices

        cons_r_(IDN,i) = w_rho_r_(i)*Wlor_r(i)*sqrt_detgamma_(i);
        // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
        cons_r_(IEN,i) = ((w_hrho_r_(i)+bsq_r(i)) * SQR(Wlor_r(i)) - w_rho_r_(i)*Wlor_r(i) - w_p_r_(i) - bsq_r(i)/2.0 - alpha_(i)*alpha_(i)*b0_u_r(i)*b0_u_r(i))*sqrt_detgamma_(i);
        // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
//        cons_l_(IVX,i) = w_hrho_l_ * gamma_l * ucov_l[1]*std::sqrt(detgamma);
//        cons_l_(IVY,i) = w_hrho_l_ * gamma_l * ucov_l[2]*std::sqrt(detgamma);
//        cons_l_(IVZ,i) = w_hrho_l_ * gamma_l * ucov_l[3]*std::sqrt(detgamma);
        cons_r_(IVX,i) = ((w_hrho_r_(i)+bsq_r(i)) * Wlor_r(i) * w_util_d_r_(0,i) - alpha_(i)*b0_u_r(i)*bi_d_r(0,i)  )*sqrt_detgamma_(i);
        cons_r_(IVY,i) = ((w_hrho_r_(i)+bsq_r(i)) * Wlor_r(i) * w_util_d_r_(1,i) - alpha_(i)*b0_u_r(i)*bi_d_r(1,i)  )*sqrt_detgamma_(i);
        cons_r_(IVZ,i) = ((w_hrho_r_(i)+bsq_r(i)) * Wlor_r(i) * w_util_d_r_(2,i) - alpha_(i)*b0_u_r(i)*bi_d_r(2,i)  )*sqrt_detgamma_(i);
        cons_r_(IBY,i) = sqrt_detgamma_(i)*bb_r(ivy-1,i);
        cons_r_(IBZ,i) = sqrt_detgamma_(i)*bb_r(ivz-1,i);
        // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
        // D flux: D(v^i - beta^i/alpha)
        //
         flux_r_(IDN,i) = cons_r_(IDN,i)*alpha_(i)*(v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i));

        // tau flux: alpha_(S^i - Dv^i) - beta^i tau
          flux_r_(IEN,i) = cons_r_(IEN,i) * alpha_(i) * (v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) + alpha_(i)*sqrt_detgamma_(i)*((w_p_r_(i)+ bsq_r(i)/2.0)*v_u_r(ivx-1,i) - alpha_(i)*b0_u_r(i)*bb_r(ivx-1,i)/Wlor_r(i));
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_r_(IVX,i) = cons_r_(IVX,i) * alpha_(i) * (v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_r(0,i)*bb_r(ivx-1,i)/Wlor_r(i);      
        flux_r_(IVY,i) = cons_r_(IVY,i) * alpha_(i) * (v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_r(1,i)*bb_r(ivx-1,i)/Wlor_r(i);      
        flux_r_(IVZ,i) = cons_r_(IVZ,i) * alpha_(i) * (v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - alpha_(i)*sqrt_detgamma_(i)*bi_d_r(2,i)*bb_r(ivx-1,i)/Wlor_r(i);      
        flux_r_(ivx,i) += (w_p_r_(i)+bsq_r(i)/2.0)*alpha_(i)*sqrt_detgamma_(i);
        flux_r_(IBY,i) = alpha_(i)*(cons_r_(IBY,i)*(v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - bb_r(ivx-1,i)*sqrt_detgamma_(i)*(v_u_r(ivy-1,i) - beta_u_(ivy-1,i)/alpha_(i))  ); //check indices
        flux_r_(IBZ,i) = alpha_(i)*(cons_r_(IBZ,i)*(v_u_r(ivx-1,i) - beta_u_(ivx-1,i)/alpha_(i)) - bb_r(ivx-1, i)*sqrt_detgamma_(i)*(v_u_r(ivz-1,i) - beta_u_(ivz-1,i)/alpha_(i))  );
       } 
      // Set fluxes
        for (int n = 0; n < NHYDRO; ++n) {
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
          flux(n,k,j,i) =
              0.5 * (flux_l_(n,i) + flux_r_(n,i) - lambda(i) * (cons_r_(n,i) - cons_l_(n,i)));
          }
        }
        
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
    ey(k,j,i) =
            -0.5 * (flux_l_(IBY,i) + flux_r_(IBY,i) - lambda(i) * (cons_r_(IBY,i) - cons_l_(IBY,i)));
     ez(k,j,i) =
            0.5 * (flux_l_(IBZ,i) + flux_r_(IBZ,i) - lambda(i) * (cons_r_(IBZ,i) - cons_l_(IBZ,i)));

    wct(k,j,i) =
        GetWeightForCT(flux(IDN,k,j,i), prim_l(IDN,i), prim_r(IDN,i), dxw(i), dt);
}

  return;
}
