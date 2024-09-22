//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file Rusanov_rel_GR.cpp
//  \brief Implements local High order Rusanov-Lax Friederichs  Riemann solver for relativistic hydrodynamics
//  in pure GR.
// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()
#include <iomanip>

// Athena++ headers
#include "../../hydro.hpp"
#include "../../../z4c/z4c.hpp"
#include "../../../utils/linear_algebra.hpp"
#include "../../../utils/interp_intergrid.hpp"
#include "../../../athena_aliases.hpp"
#include "../../../coordinates/coordinates.hpp"  // Coordinates
#include "../../../eos/eos.hpp"                  // EquationOfState
#include "../../../mesh/mesh.hpp"                // MeshBlock
#include "../../Characterisitcfields.hpp"


void Hydro::RusanovFlux(
  AthenaArray<Real> &prim,
  AthenaArray<Real> &cons,
  AthenaArray<Real> &x1flux,
  AthenaArray<Real> &x2flux,
  AthenaArray<Real> &x3flux)
{
//----------------------------------------------------------------------------------------
  using namespace gra::aliases;
  using namespace LinearAlgebra;
  using namespace characterisiticfields;
  using namespace averages::grhd;
//----------------------------------------------------------------------------------------

  MeshBlock *pmb = pmy_block;
  const int nn1 = pmy_block->nverts1;  // number of interfaces
  const int nc1 = pmy_block->ncells1; // cell
  const int nc2 = pmy_block->ncells2;
  const int nc3 = pmy_block->ncells3;
  int il,iu,jl,ju,kl,ku;
  int ivx;
// perform variable resampling when required
  Z4c * pz4c = pmy_block->pz4c;
  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym gamma_dd(pz4c->storage.adm, Z4c::I_ADM_gxx);
  AT_N_sca alpha(   pz4c->storage.adm, Z4c::I_ADM_alpha);
  AT_N_vec beta_u(  pz4c->storage.adm, Z4c::I_ADM_betax);

// 1d slices of primitives ----------------------------------------------------------------
  AT_N_sca w_rho(prim, IDN); //density
  AT_N_sca w_p(  prim, IPR); //pressure
  AT_N_vec w_util_u(prim, IVX); // fluid 3-velocity(contavariant)

  // various scratches --------------------------------------------------------
  // BD: TODO - faster to pre-alloc in Hydro class, probably
  AT_N_sca sqrt_detgamma(nc3,nc2,nc1);
  AT_N_sca detgamma(nc3,nc2, nc1);  // spatial met det
  AT_N_sca oo_detgamma(nc1);  // spatial met det

  AT_N_sym gamma_uu(nc3,nc2,nc1); //inverse spatial slice metric

  AT_N_vec w_v_u(nc3,nc2,nc1);// Eulerian velocity
  AT_N_vec w_v_d(nc3,nc2,nc1);
  AT_N_sca Wcc(nc3,nc2,nc1);   // Lorentz factor
  AT_N_sca w_norm2_v(nc3,nc2,nc1);     // Normed Eulerian Velocity

  // primitive vel. (covariant.)
  AT_N_vec w_util_d(nc1);

  // cell centre fluxes and eigen values
  AT_H_vec fcc(nc3,nc2,nc1);
  AT_H_vec lambda(nc3,nc2,nc1);

  //...............................................................
  
  // Left/ Right eigen vectors
  //AT_H_eig L_eig(nc1);
  //AT_H_eig R_eig(nc1);
  Real L_eig[NHYDRO][NHYDRO];
  Real R_eig[NHYDRO][NHYDRO];
  // Max Wave speeds in stencil
  //AT_H_vec lambda_max(nc1);
  Real lambda_max[NHYDRO];
  // Characteristic flux
  //AT_H_vec char_flx(nc1);
  Real char_flx[NHYDRO];
  for (int k=0; k<nc3; ++k)
  for (int j=0; j<nc2; ++j)
  {
    // Determinant of 3 metric
    for (int i = 0; i < nc1; ++i)
    {
      detgamma(k,j,i)      = Det3Metric(gamma_dd, k,j,i);
      sqrt_detgamma(k,j,i) = std::sqrt(detgamma(k,j,i));
      oo_detgamma(i)   = 1. / detgamma(k,j,i);
    }

    // Inverse of the 3 metric at cell centre
    for (int i=0; i<nc1; ++i)
    {
      Inv3Metric(
        oo_detgamma(i),
        gamma_dd(0,0,k,j,i), gamma_dd(0,1,k,j,i), gamma_dd(0,2,k,j,i),
        gamma_dd(1,1,k,j,i), gamma_dd(1,2,k,j,i), gamma_dd(2,2,k,j,i),
        &gamma_uu(0,0,k,j,i), &gamma_uu(0,1,k,j,i), &gamma_uu(0,2,k,j,i),
        &gamma_uu(1,1,k,j,i), &gamma_uu(1,2,k,j,i), &gamma_uu(2,2,k,j,i));
    }

    // Get Euleiran Velocity
    GetEulerianVelocity(0,nc1-1,k,j, gamma_dd, w_util_u,Wcc, w_v_u );

    // Get Normed Eulerian Velocity
    for (int i=0; i<nc1; ++i){
      w_norm2_v(k,j,i)=InnerProductVecMetric(w_v_u,gamma_dd,k,j,i);
    }
  }


//---------------------------------------------------------------------------
// i-direction
  { 
    ivx=1;
    AT_H_vec fcc(nc3,nc2,nc1);
    for (int k=0; k<nc3; ++k)
    for (int j=0; j<nc2; ++j)
    {
    // Get fluxes And Eigen values at cell centres
      GetFluxesGRHD(k,j,ivx,0,nc1-1, sqrt_detgamma,alpha,beta_u,
      w_p,w_v_u,cons,fcc );

      GetEigenValues(pmb,k,j,0, nc1-1, ivx, gamma_dd, gamma_uu,
        alpha,w_rho,w_p,beta_u,w_v_u,w_norm2_v,lambda);
    }

    // NOW GET AVERAGE QUANTITIES
    pmb->precon->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for(int i =il;i<=iu;++i)
      {
        Real avg_field[12]={}; 
        Real avg_hq[13]={};
        Real avg_eig[6]={};
        GetAvgs(pmb,k,j,i,ivx,gamma_dd,gamma_uu,detgamma,beta_u,alpha,
                w_rho,w_p,w_v_u,w_v_d,w_norm2_v,Wcc,avg_field,avg_hq,avg_eig );
        GetEigenVectorGRHD(pmb,ivx,avg_field,avg_hq,avg_eig,
                          L_eig,R_eig );      
      // Maximum wave speed on interface depending on stensil
        GetMaximalWaveSpeed(k,j,i,2,ivx,lambda, lambda_max);
      // Get Characteristic fields
        ReconCharFields(k,j,i,3,ivx,fcc, lambda_max,cons,L_eig,
                    char_flx);
        ReconFlux(k,j,i,ivx, char_flx, R_eig, x1flux);
#if 1
        for(int a = 0;a<NHYDRO;++a){
          if (!std::isfinite(char_flx[a])){
            std::cout<<i<<","<<j<<","<<k<<std::endl;
            for(int a =0;a<12;++a){
              std::cout<<"Field="<<avg_field[a]<<std::endl;
            }
            for(int a =0;a<13;++a){
              std::cout<<"hq="<<avg_hq[a]<<std::endl;
            }
            for(int a =0;a<6;++a){
              std::cout<<"eig="<<avg_eig[a]<<std::endl;
            }
            for(int a =0;a<NHYDRO;++a){
              std::cout<<"max lambda"<<lambda_max[a]<<std::endl;
            }

            

            for(int a =0;a<NHYDRO;++a)
            for(int b=0;b<NHYDRO;++b){
              std::cout<<"L_EIG="<<a<<","<<b<<"="<<L_eig[a][b]<<std::endl;
            }

            for(int a =0;a<NHYDRO;++a)
            for(int b=0;b<NHYDRO;++b){
              std::cout<<"R_EIG="<<b<<","<<a<<"="<<R_eig[b][a]<<std::endl;
            }
            for(int a=0; a<NHYDRO;++a){
              std::cout<<"char_flux"<<char_flx[a]<<std::endl;
            }
            for(int a=0; a<NHYDRO;++a){
              std::cout<<"flux"<<x1flux(a,k,j,i)<<std::endl;
            }

            for(int a=0; a<NHYDRO;++a){
              std::cout<<"flux"<<fcc(a,k,j,i)<<std::endl;
            }

            exit(1);

          }
        }
#endif
      }
    }
  }
//---------------------------------------------------------------------------
// j-direction
  if (pmb->pmy_mesh->f2)
  {
    AT_H_vec fcc(nc3,nc2,nc1);
    ivx=2;
    for (int k=0; k<nc3; ++k)
    for (int j=0; j<nc2; ++j)
    {
    // Get fluxes And Eigen values at cell centres
      GetFluxesGRHD(k,j,ivx,0,nc1-1, sqrt_detgamma,alpha,beta_u,
      w_p,w_v_u,cons,fcc );

      GetEigenValues(pmb,k,j,0, nc1-1, ivx, gamma_dd, gamma_uu,
        alpha,w_rho,w_p,beta_u,w_v_u,w_norm2_v,lambda);
    }

  // NOW GET AVERAGE QUANTITIES
    pmb->precon->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    {
      for(int i =il;i<=iu;++i)
      {
        Real avg_field[12]={}; 
        Real avg_hq[13]={};
        Real avg_eig[6]={};
        GetAvgs(pmb,k,j,i,ivx,gamma_dd,gamma_uu,detgamma,beta_u,alpha,
                w_rho,w_p,w_v_u,w_v_d,w_norm2_v,Wcc, avg_field,avg_hq,avg_eig );
        GetEigenVectorGRHD(pmb,ivx,avg_field,avg_hq,avg_eig,
                          L_eig,R_eig );

        // Maximum wave speed on interface depending on stensil
        GetMaximalWaveSpeed(k,j,i,2,ivx,lambda, lambda_max); 
        // Get Characteristic fields
        ReconCharFields(k,j,i,3,ivx,fcc, lambda_max,cons,L_eig, 
                      char_flx);
        ReconFlux(k,j,i,ivx, char_flx, R_eig, x2flux); //TODO

      }
    }
  }


//---------------------------------------------------------------------------
// k-direction
  if (pmb->pmy_mesh->f3)
  {
    AT_H_vec fcc(nc3,nc2,nc1);
    ivx=3;
    for (int k=0; k<nc3; ++k)
    for (int j=0; j<nc2; ++j)
    {
    // Get fluxes And Eigen values at cell centres
      GetFluxesGRHD(k,j,ivx,0,nc1-1, sqrt_detgamma,alpha,beta_u,
      w_p,w_v_u,cons,fcc );

      GetEigenValues(pmb,k,j,0, nc1-1, ivx, gamma_dd, gamma_uu,
        alpha,w_rho,w_p,beta_u,w_v_u,w_norm2_v,lambda);
    }

    // NOW GET AVERAGE QUANTITIES
    pmb->precon->SetIndicialLimitsCalculateFluxes(ivx, il, iu, jl, ju, kl, ku);
    for (int j=jl; j<=ju; ++j)
    for (int k=kl; k<=ku; ++k)
    {
      for(int i =il;i<=iu;++i)
      {
        Real avg_field[12]={}; 
        Real avg_hq[13]={};
        Real avg_eig[6]={};
        GetAvgs(pmb,k,j,i,ivx,gamma_dd,gamma_uu,detgamma,beta_u,alpha,
                w_rho,w_p,w_v_u,w_v_d,w_norm2_v,Wcc, avg_field,avg_hq,avg_eig );
        GetEigenVectorGRHD(pmb,ivx,avg_field,avg_hq,avg_eig,
                          L_eig,R_eig );

        // Maximum wave speed on interface depending on stensil
        GetMaximalWaveSpeed(k,j,i,2,ivx,lambda, lambda_max);
        // Get Characteristic fields
        ReconCharFields(k,j,i,3,ivx,fcc, lambda_max,cons,L_eig,
                      char_flx);
        ReconFlux(k,j,i,ivx, char_flx, R_eig,x3flux);
      }
    }
  }

return;
  
}