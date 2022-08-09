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
#include "../../../utils/interp_intergrid.hpp"
#include "../../../athena.hpp"                   // enums, macros
#include "../../../athena_arrays.hpp"            // AthenaArray
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock

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
// compare with modification to add_flux_divergence_dyn, where factors of face area, cell volume etc are missing
// since they are included here.
namespace{
Real Determinant(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23,
                 Real a31, Real a32, Real a33);
Real Determinant(Real a11, Real a12, Real a21, Real a22);
Real Det3Metric(AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> const & gamma,
                  int const i);
void Inverse3Metric(Real const detginv,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                Real * uyy, Real * uyz, Real * uzz);

}

void Hydro::RiemannSolver(const int k, const int j,
    const int il, const int iu, const int ivx,
    AthenaArray<Real> &prim_l, AthenaArray<Real> &prim_r, AthenaArray<Real> &flux, const AthenaArray<Real> &dxw) {
  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int a,b;
  const int nn1 = iu  +1;
  // Extract ratio of specific heats
#if USETM
  const Real mb = pmy_block->peos->GetEOS().GetBaryonMass();
#else
  const Real gamma_adi = pmy_block->peos->GetGamma();
#endif

  // Go through 1D arrays of interfaces
//  for (int k = kl; k <= ku; ++k) {
//    for (int j = jl; j <= ju; ++j) {

//    Returning alpha(i), beta_u(a,i) gamma_dd(a,b,i)
      // Get metric components in ``old style'' for testing
      switch (ivx) {
        case IVX:
          pmy_block->pcoord->Face1Metric(k, j, il, iu, g_, gi_);
          break;
        case IVY:
          pmy_block->pcoord->Face2Metric(k, j, il, iu, g_, gi_);
          break;
        case IVZ:
          pmy_block->pcoord->Face3Metric(k, j, il, iu, g_, gi_);
          break;
      }
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha, rho_l, rho_r, pgas_l, pgas_r, wgas_l, wgas_r,detgamma,detg;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> Wlor_l, Wlor_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> u0_l, u0_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> v2_l, v2_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> lambda_p_l, lambda_m_l;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> lambda_p_r, lambda_m_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> lambda;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> utilde_u_l, utilde_u_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> utilde_d_l, utilde_d_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> v_u_l, v_u_r;
      AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> ucon_l, ucon_r;
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_dd;
      AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gamma_uu;

     AthenaArray<Real> vcgamma_xx,vcgamma_xy,vcgamma_xz,vcgamma_yy;
      AthenaArray<Real> vcgamma_yz,vcgamma_zz,vcbeta_x,vcbeta_y;
      AthenaArray<Real> vcbeta_z, vcalpha;

    vcgamma_xx.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxx,1);
      vcgamma_xy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxy,1);
      vcgamma_xz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gxz,1);
      vcgamma_yy.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gyy,1);
      vcgamma_yz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gyz,1);
      vcgamma_zz.InitWithShallowSlice(pmy_block->pz4c->storage.adm,Z4c::I_ADM_gzz,1);
  vcbeta_x.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betax,1);
      vcbeta_y.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betay,1);
      vcbeta_z.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_betaz,1);
      vcalpha.InitWithShallowSlice(pmy_block->pz4c->storage.u,Z4c::I_Z4c_alpha,1);



      alpha.NewAthenaTensor(nn1);
      detg.NewAthenaTensor(nn1);
      detgamma.NewAthenaTensor(nn1);
      rho_l.NewAthenaTensor(nn1);
      rho_r.NewAthenaTensor(nn1);
      pgas_l.NewAthenaTensor(nn1);
      pgas_r.NewAthenaTensor(nn1);
      wgas_l.NewAthenaTensor(nn1);
      wgas_r.NewAthenaTensor(nn1);
      Wlor_l.NewAthenaTensor(nn1);
      Wlor_r.NewAthenaTensor(nn1);
      u0_l.NewAthenaTensor(nn1);
      u0_r.NewAthenaTensor(nn1);
      v2_l.NewAthenaTensor(nn1);
      v2_r.NewAthenaTensor(nn1);
      lambda.NewAthenaTensor(nn1);
      lambda_p_l.NewAthenaTensor(nn1);
      lambda_m_l.NewAthenaTensor(nn1);
      lambda_p_r.NewAthenaTensor(nn1);
      lambda_m_r.NewAthenaTensor(nn1);
      beta_u.NewAthenaTensor(nn1);
      utilde_u_l.NewAthenaTensor(nn1);
      utilde_u_r.NewAthenaTensor(nn1);
      v_u_l.NewAthenaTensor(nn1);
      v_u_r.NewAthenaTensor(nn1);
      utilde_d_l.NewAthenaTensor(nn1);
      utilde_d_r.NewAthenaTensor(nn1);
      ucon_l.NewAthenaTensor(nn1);
      ucon_r.NewAthenaTensor(nn1);
      gamma_dd.NewAthenaTensor(nn1);
      gamma_uu.NewAthenaTensor(nn1);
      // Go through each interface
       #pragma omp simd
      for (int i = il; i <= iu; ++i){
          gamma_dd(0,0,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_xx(k,j,i));
          gamma_dd(0,1,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_xy(k,j,i));
          gamma_dd(0,2,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_xz(k,j,i));
          gamma_dd(1,1,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_yy(k,j,i));
          gamma_dd(1,2,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_yz(k,j,i));
          gamma_dd(2,2,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcgamma_zz(k,j,i));
          alpha(i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcalpha(k,j,i));
          beta_u(0,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcbeta_x(k,j,i));
          beta_u(1,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcbeta_y(k,j,i));
          beta_u(2,i) = pmy_block->pz4c->ig->map3d_VC2FC(ivx-1,vcbeta_z(k,j,i));
      }



       #pragma omp simd
      for (int i = il; i <= iu; ++i) {
      detgamma(i) = Det3Metric(gamma_dd,i);
      detg(i) = SQR(alpha(i)) * detgamma(i);
      Inverse3Metric(1.0/detgamma(i),
          gamma_dd(0,0,i), gamma_dd(0,1,i), gamma_dd(0,2,i),
          gamma_dd(1,1,i), gamma_dd(1,2,i), gamma_dd(2,2,i),
          &gamma_uu(0,0,i), &gamma_uu(0,1,i), &gamma_uu(0,2,i),
          &gamma_uu(1,1,i), &gamma_uu(1,2,i), &gamma_uu(2,2,i));
 
      }

    #pragma omp simd
      for (int i = il; i <= iu; ++i){
              rho_l(i) = prim_l(IDN,i);
              pgas_l(i) = prim_l(IPR,i);
              rho_r(i) = prim_r(IDN,i);
              pgas_r(i) = prim_r(IPR,i);
          }



      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
              utilde_u_l(a,i) = prim_l(a+IVX,i);
          }
      }
      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
              utilde_u_r(a,i) = prim_r(a+IVX,i);
          }
      }
      Wlor_l.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  Wlor_l(i) += utilde_u_l(a,i)*utilde_u_l(b,i)*gamma_dd(a,b,i);
              }
           }
       }
        #pragma omp simd
      for (int i = il; i <= iu; ++i){
            Wlor_l(i) = std::sqrt(1.0+Wlor_l(i));
       }
      Wlor_r.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  Wlor_r(i) += utilde_u_r(a,i)*utilde_u_r(b,i)*gamma_dd(a,b,i);
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
             v_u_l(a,i) = utilde_u_l(a,i)/Wlor_l(i);
          }
      }
      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
             v_u_r(a,i) = utilde_u_r(a,i)/Wlor_r(i);
          }
      }

      v2_l.ZeroClear();
      for(a=0;a<NDIM;++a){
            for(b=0;b<NDIM;++b){
               #pragma omp simd
                   for (int i = il; i <= iu; ++i){
                       v2_l(i) += gamma_dd(a,b,i)*v_u_l(a,i)*v_u_l(b,i);
                   }
             }
       }
      v2_r.ZeroClear();
      for(a=0;a<NDIM;++a){
            for(b=0;b<NDIM;++b){
               #pragma omp simd
                   for (int i = il; i <= iu; ++i){
                       v2_r(i) += gamma_dd(a,b,i)*v_u_r(a,i)*v_u_r(b,i);
                   }
             }
       }

     utilde_d_l.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  utilde_d_l(a,i) += utilde_u_l(b,i)*gamma_dd(a,b,i);
              }
          }
      }
     utilde_d_r.ZeroClear();
      for(a=0;a<NDIM;++a){
          for(b=0;b<NDIM;++b){
               #pragma omp simd
      for (int i = il; i <= iu; ++i){
                  utilde_d_r(a,i) += utilde_u_r(b,i)*gamma_dd(a,b,i);
              }
          }
      }
       #pragma omp simd
      for (int i = il; i <= iu; ++i){
      u0_l(i) = Wlor_l(i)/alpha(i);
      u0_r(i) = Wlor_r(i)/alpha(i);
      }
/*      for(a=0;a<NDIM;++a){
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
            ucon_u_l(a,i) = utilde_u_l(a,i) + Wlor_l(i) * beta_u(a,i)/alpha(i);
            ucon_u_r(a,i) = utilde_u_r(a,i) + Wlor_r(i) * beta_u(a,i)/alpha(i);
          }
       }
*/
      
       #pragma omp simd
      for (int i = il; i <= iu; ++i){
        // Calculate wavespeeds in left state NB EOS specific
#if USETM
        // If using the PrimitiveSolver framework, get the number density
        // and temperature to help calculate enthalpy.
        Real nl = rho_l(i)/mb;
        Real nr = rho_r(i)/mb;
        // FIXME: Generalize to work with EOSes accepting particle fractions.
        Real Y[MAX_SPECIES] = {0.0};
        Real Tl = pmy_block->peos->GetEOS().GetTemperatureFromP(nl, pgas_l(i), Y);
        Real Tr = pmy_block->peos->GetEOS().GetTemperatureFromP(nr, pgas_r(i), Y);
        wgas_l(i) = rho_l(i)*pmy_block->peos->GetEOS().GetEnthalpy(nl, Tl, Y);
        wgas_r(i) = rho_r(i)*pmy_block->peos->GetEOS().GetEnthalpy(nr, Tr, Y);

        // Calculate the sound speeds
        pmy_block->peos->SoundSpeedsGR(nl, Tl, v_u_l(ivx-1,i), v2_l(i), alpha(i), beta_u(ivx-1,i), gamma_uu(ivx-1,ivx-1,i),
            &lambda_p_l(i), &lambda_m_l(i));
        pmy_block->peos->SoundSpeedsGR(nr, Tr, v_u_r(ivx-1,i), v2_r(i), alpha(i), beta_u(ivx-1,i), gamma_uu(ivx-1,ivx-1,i),
            &lambda_p_r(i), &lambda_m_r(i));
#else
        wgas_l(i) = rho_l(i) + gamma_adi/(gamma_adi-1.0) * pgas_l(i);
        wgas_r(i) = rho_r(i) + gamma_adi/(gamma_adi-1.0) * pgas_r(i);
       
        pmy_block->peos->SoundSpeedsGR(wgas_l(i), pgas_l(i), v_u_l(ivx-1,i), v2_l(i), alpha(i), beta_u(ivx-1,i), gamma_uu(ivx-1,ivx-1,i),  &lambda_p_l(i), &lambda_m_l(i));
        pmy_block->peos->SoundSpeedsGR(wgas_r(i), pgas_r(i), v_u_r(ivx-1,i), v2_r(i), alpha(i), beta_u(ivx-1,i), gamma_uu(ivx-1,ivx-1,i),  &lambda_p_r(i), &lambda_m_r(i));
#endif
}
        // Calculate extremal wavespeed
         #pragma omp simd
      for (int i = il; i <= iu; ++i){
        Real lambda_l = std::min(lambda_m_l(i), lambda_m_r(i));
        Real lambda_r = std::max(lambda_p_l(i), lambda_p_r(i));
        lambda(i) = std::max(lambda_r, -lambda_l);
        }
        AthenaArray<Real> cons_l, cons_r, flux_l, flux_r;
        cons_l.NewAthenaArray(NWAVE,nn1);
        cons_r.NewAthenaArray(NWAVE,nn1);
        flux_l.NewAthenaArray(NWAVE,nn1);
        flux_r.NewAthenaArray(NWAVE,nn1);
        // Calculate conserved quantities in L region including factor of sqrt(detgamma)
         #pragma omp simd
      for (int i = il; i <= iu; ++i){
        // D = rho * gamma_lorentz
        cons_l(IDN,i) = rho_l(i)*Wlor_l(i)*std::sqrt(detgamma(i));
        // tau = (rho * h = ) wgas * gamma_lorentz**2 - rho * gamma_lorentz - p
        cons_l(IEN,i) = (wgas_l(i) * SQR(Wlor_l(i)) - rho_l(i)*Wlor_l(i) - pgas_l(i))*std::sqrt(detgamma(i));
        // S_i = wgas * gamma_lorentz**2 * v_i = wgas * gamma_lorentz * u_i
        cons_l(IVX,i) = wgas_l(i) * Wlor_l(i) * utilde_d_l(0,i)*std::sqrt(detgamma(i));
        cons_l(IVY,i) = wgas_l(i) * Wlor_l(i) * utilde_d_l(1,i)*std::sqrt(detgamma(i));
        cons_l(IVZ,i) = wgas_l(i) * Wlor_l(i) * utilde_d_l(2,i)*std::sqrt(detgamma(i));
        // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
        // D flux: D(v^i - beta^i/alpha)
         flux_l(IDN,i) = cons_l(IDN,i)*alpha(i)*(v_u_l(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));

        // tau flux: alpha(S^i - Dv^i) - beta^i tau
          flux_l(IEN,i) = cons_l(IEN,i) * alpha(i) * (v_u_l(ivx-1,i) - beta_u(ivx-1,i)/alpha(i)) + std::sqrt(detg(i))*pgas_l(i)*v_u_l(ivx-1,i);
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_l(IVX,i) = cons_l(IVX,i) * alpha(i) * (v_u_l(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_l(IVY,i) = cons_l(IVY,i) * alpha(i) * (v_u_l(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_l(IVZ,i) = cons_l(IVZ,i) * alpha(i) * (v_u_l(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_l(ivx,i) += pgas_l(i)*std::sqrt(detg(i));

        // D = rho * gamma_rorentz
        cons_r(IDN,i) = rho_r(i)*Wlor_r(i)*std::sqrt(detgamma(i));
        // tau = (rho * h = ) wgas * gamma_rorentz**2 - rho * gamma_rorentz - p
        cons_r(IEN,i) = (wgas_r(i) * SQR(Wlor_r(i)) - rho_r(i)*Wlor_r(i) - pgas_r(i))*std::sqrt(detgamma(i));
        // S_i = wgas * gamma_rorentz**2 * v_i = wgas * gamma_rorentz * u_i
        cons_r(IVX,i) = wgas_r(i) * Wlor_r(i) * utilde_d_r(0,i)*std::sqrt(detgamma(i));
        cons_r(IVY,i) = wgas_r(i) * Wlor_r(i) * utilde_d_r(1,i)*std::sqrt(detgamma(i));
        cons_r(IVZ,i) = wgas_r(i) * Wlor_r(i) * utilde_d_r(2,i)*std::sqrt(detgamma(i));
        // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
        // D flux: D(v^i - beta^i/alpha)
         flux_r(IDN,i) = cons_r(IDN,i)*alpha(i)*(v_u_r(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));

        // tau flux: alpha(S^i - Dv^i) - beta^i tau
          flux_r(IEN,i) = cons_r(IEN,i) * alpha(i) * (v_u_r(ivx-1,i) - beta_u(ivx-1,i)/alpha(i)) + std::sqrt(detg(i))*pgas_r(i)*v_u_r(ivx-1,i);
 
        //S_i flux alpha S^j_i - beta^j S_i
        flux_r(IVX,i) = cons_r(IVX,i) * alpha(i) * (v_u_r(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_r(IVY,i) = cons_r(IVY,i) * alpha(i) * (v_u_r(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_r(IVZ,i) = cons_r(IVZ,i) * alpha(i) * (v_u_r(ivx-1,i) - beta_u(ivx-1,i)/alpha(i));      
        flux_r(ivx,i) += pgas_r(i)*std::sqrt(detg(i));
       } 
      // Set fluxes
        for (int n = 0; n < NHYDRO; ++n) {
           #pragma omp simd
      for (int i = il; i <= iu; ++i){
          flux(n,k,j,i) =
              0.5 * (flux_l(n,i) + flux_r(n,i) - lambda(i) * (cons_r(n,i) - cons_l(n,i)));
          }
        }



    
  
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

void Inverse3Metric(Real const detginv,
                     Real const gxx, Real const gxy, Real const gxz,
                     Real const gyy, Real const gyz, Real const gzz,
                     Real * uxx, Real * uxy, Real * uxz,
                     Real * uyy, Real * uyz, Real * uzz)
{
  *uxx = (-SQR(gyz) + gyy*gzz)*detginv;
  *uxy = (gxz*gyz  - gxy*gzz)*detginv;
  *uyy = (-SQR(gxz) + gxx*gzz)*detginv;
  *uxz = (-gxz*gyy + gxy*gyz)*detginv;
  *uyz = (gxy*gxz  - gxx*gyz)*detginv;
  *uzz = (-SQR(gxy) + gxx*gyy)*detginv;
  return;
}


}
