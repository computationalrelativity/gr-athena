//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file m1_grsource.cpp
//  \brief calculate source with geometry terms and add to r.h.s.

// C++ standard headers
//#include <cmath> // pow

// Athena++ headers
#include "m1.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"

using namespace utils;

void M1::GRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  Lab_vars vec, vec_rhs;
  SetLabVarsAliases(u, vec);
  SetLabVarsAliases(u_rhs, vec_rhs); 
  
  // Metric on VC
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> vc_z4c_alpha;     // lapse
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> vc_z4c_beta_u;    // shift
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> vc_adm_g_dd;      // 3-metric 
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> vc_adm_K_dd;      // extr.curv.
  vc_z4c_alpha.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_alpha);
  vc_z4c_beta_u.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_betax);
  vc_adm_g_dd.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxx);
  vc_adm_K_dd.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kxx);

  // 1D metrc variables on CC
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gam_dd;   //gamma_{ij}
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> K_dd;     // K_{ij}
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> beta_u;   // beta^i
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> alpha;    // lapse
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> gam_uu;   // gamma^{ij}
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> dgam_ddd; // pd_i gamma_{jk}
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> dbeta_du; // pd_i beta^j
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> dalpha_d; // pd_i alpha
      
  int const nn1 = pmy_block->ncells1;

  gam_dd.NewAthenaTensor(nn1);
  K_dd.NewAthenaTensor(nn1);
  beta_u.NewAthenaTensor(nn1);
  alpha.NewAthenaTensor(nn1);
  gam_uu.NewAthenaTensor(nn1);
  dgam_ddd.NewAthenaTensor(nn1);
  dbeta_du.NewAthenaTensor(nn1);
  dalpha_d.NewAthenaTensor(nn1);
  
  // 3-tensors at (k,j,i) CC
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_uu;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> g_uu;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> K_dd__;
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dalp_d;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 3> dg_ddd;

  F_d.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  P_uu.NewTensorPointwise();
  K_dd__.NewTensorPointwise();
  dalp_d.NewTensorPointwise();
  dg_ddd.NewTensorPointwise();
  
  // Go through cells
  CLOOP2(k,j) {
    
    // Get ADM 3-metric, extr.curv., alpha and beta on CC
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {   
	      CLOOP1(i) {
          gam_dd(a,b,i) = VCInterpolation(vc_adm_g_dd(a,b),k,j,i);
	      }
      }
    }
    
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {   
	      CLOOP1(i) {
          K_dd(a,b,i) = VCInterpolation(vc_adm_K_dd(a,b),k,j,i);
	      }
      }
    }
    
    for(int a = 0; a < NDIM; ++a) {
      CLOOP1(i) {
        beta_u(a,i) = VCInterpolation(vc_z4c_beta_u(a),k,j,i);
      }
    }
    
    CLOOP1(i) {
      alpha(i) = VCInterpolation(vc_z4c_alpha(),k,j,i);
    }
    
    // Get metric drvts on CC
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
	      for(int c = 0; c < NDIM; ++c) {
	        CLOOP1(i) {
            dgam_ddd(c,a,b,i) = VCDiff(c,vc_adm_g_dd(a,b),k,j,i);
	        }
	      }
      }
    }
    
    for(int a = 0; a < NDIM; ++a) {
      for(int c = 0; c < NDIM; ++c) {
	      CLOOP1(i) {
          dbeta_du(c,a,i) = VCDiff(c,vc_z4c_beta_u(a),k,j,i);
	      }
      }
    }
    
    for(int a = 0; a < NDIM; ++a) {
      CLOOP1(i) {
        dalpha_d(a,i) = VCDiff(a,vc_z4c_alpha(),k,j,i);
      }
    }
    // Inverse metric on CC
    CLOOP1(i) {
      Real const detg = SpatialDet(gam_dd(0,0,i),gam_dd(0,1,i), gam_dd(0,2,i), 
				                           gam_dd(1,1,i), gam_dd(1,2,i), gam_dd(2,2,i));
      SpatialInv(1.0/detg,
		             gam_dd(0,0,i), gam_dd(0,1,i), gam_dd(0,2,i),
		             gam_dd(1,1,i), gam_dd(1,2,i), gam_dd(2,2,i),
		             &gam_uu(0,0,i), &gam_uu(0,1,i), &gam_uu(0,2,i),
		             &gam_uu(1,1,i), &gam_uu(1,2,i), &gam_uu(2,2,i));
    }
    
    // Now combine with the radiation fields
    CLOOP1(i) {
      
      // Convert metric to pointwise tensor variables at (k,j,i) CC
      // Can re-use the P_dd and F_d pack routines
      pack_P_dd(gam_uu(0,0,k,j,i), gam_uu(0,1,k,j,i), gam_uu(0,2,k,j,i),
		            gam_uu(1,1,k,j,i), gam_uu(1,2,k,j,i), gam_uu(2,2,k,j,i), g_uu);      
      
      pack_P_dd(K_dd(0,0,k,j,i),K_dd(0,1,k,j,i),K_dd(0,2,k,j,i),
		            K_dd(1,1,k,j,i),K_dd(1,2,k,j,i),K_dd(2,2,k,j,i), K_dd__);      
      
      pack_P_ddd(dgam_ddd(0,0,0,k,j,i), dgam_ddd(0,0,1,k,j,i), dgam_ddd(0,0,2,k,j,i),
                 dgam_ddd(0,1,1,k,j,i), dgam_ddd(0,1,2,k,j,i), dgam_ddd(0,2,2,k,j,i),
                 dgam_ddd(1,0,0,k,j,i), dgam_ddd(1,0,1,k,j,i), dgam_ddd(1,0,2,k,j,i),
                 dgam_ddd(1,1,1,k,j,i), dgam_ddd(1,1,2,k,j,i), dgam_ddd(1,2,2,k,j,i),
                 dgam_ddd(2,0,0,k,j,i), dgam_ddd(2,0,1,k,j,i), dgam_ddd(2,0,2,k,j,i),
                 dgam_ddd(2,1,1,k,j,i), dgam_ddd(2,1,2,k,j,i), dgam_ddd(2,2,2,k,j,i), dg_ddd);

      pack_F_d(dalpha_d(0,i), dalpha_d(1,i), dalpha_d(2,i), dalp_d);

      alpha(i) = VCInterpolation(vc_z4c_alpha(),k,j,i);
      Real const alpha_i = alpha(i);

      for (int ig = 0; ig < ngroups*nspecies; ++ig) {

        pack_F_d(vec.F_d(0,ig,k,j,i),vec.F_d(1,ig,k,j,i),vec.F_d(2,ig,k,j,i), F_d);
        pack_P_dd(rad.P_dd(0,0,ig,k,j,i),rad.P_dd(0,1,ig,k,j,i),rad.P_dd(0,2,ig,k,j,i),
                  rad.P_dd(1,1,ig,k,j,i),rad.P_dd(1,2,ig,k,j,i),rad.P_dd(2,2,ig,k,j,i), P_dd);

        // Contravariant radiation pressure
        tensor::contract2(g_uu, P_dd, &P_uu);

        // Compute radiation energy sources
        // Note that everything is already densitized
        Real const rhsE = alpha_i *tensor::dot(P_uu, K_dd__) - tensor::dot(g_uu, F_d, dalp_d);
        vec_rhs.E(ig,k,j,i) += rhsE;

        // Compute the radiation flux sources
        for (int a = 0; a < NDIM; ++a) {
          Real rhsFd_a = - rhsE * dalp_d(a);
          for (int b = 0; b < NDIM; ++b) {
            rhsFd_a += F_d(b) * dbeta_du(a,b,i);
          }
          for (int b = 0; b < NDIM; ++b) {
            for (int c = 0; c < NDIM; ++c) {
              rhsFd_a += 0.5*alpha_i * P_uu(b,c) * dg_ddd(a,b,c);
            }
          }
          vec_rhs.F_d(a,ig,k,j,i) += rhsFd_a;
        }
      }
    } // CLOOP1
  } // CLOOP2     
  
  gam_dd.DeleteAthenaTensor();
  K_dd.DeleteAthenaTensor();
  beta_u.DeleteAthenaTensor();
  alpha.DeleteAthenaTensor();
  gam_uu.DeleteAthenaTensor();
  dgam_ddd.DeleteAthenaTensor();
  dbeta_du.DeleteAthenaTensor();
  dalpha_d.DeleteAthenaTensor();

  F_d.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  P_uu.DeleteTensorPointwise();
  K_dd__.DeleteTensorPointwise();
  dalp_d.DeleteTensorPointwise();
  
}

	
