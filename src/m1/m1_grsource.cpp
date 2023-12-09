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
#include "../coordinates/coordinates.hpp"
#include "../z4c/z4c.hpp"

using namespace utils;

void M1::GRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  if (M1_GRSOURCES_OFF) return;
  M1_DEBUG_PR("in: GRSources");

  Lab_vars vec, vec_rhs;
  SetLabVarsAliases(u, vec);
  SetLabVarsAliases(u_rhs, vec_rhs); 

  const Real idx[3] = {
     1.0/pmy_block->pcoord->dx1v(0),
     1.0/pmy_block->pcoord->dx2v(0),
     1.0/pmy_block->pcoord->dx3v(0),
  };
  
  // Metric on VC
  AthenaArray<Real> vc_z4c_alpha;
  AthenaArray<Real> vc_z4c_beta_u_x, vc_z4c_beta_u_y, vc_z4c_beta_u_z;
  AthenaArray<Real> vc_gamma_xx, vc_gamma_xy, vc_gamma_xz, vc_gamma_yy,
    vc_gamma_yz, vc_gamma_zz, vc_K_xx, vc_K_xy, vc_K_xz, vc_K_yy, vc_K_yz, vc_K_zz;
  
  // 3-tensors at (k,j,i) CC
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gam_dd;   //gamma_{ij}
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> K_dd;     // K_{ij}
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> beta_u;   // beta^i
  TensorPointwise<Real, Symmetries::NONE, NDIM, 0> alpha;    // lapse
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gam_uu;   // gamma^{ij}
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 3> dgam_ddd; // pd_i gamma_{jk}
  TensorPointwise<Real, Symmetries::NONE, NDIM, 2> dbeta_du; // pd_i beta^j
  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dalpha_d; // pd_i alpha

  TensorPointwise<Real, Symmetries::NONE, NDIM, 1> F_d;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_dd;
  TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_uu;

  gam_dd.NewTensorPointwise();
  K_dd.NewTensorPointwise();
  beta_u.NewTensorPointwise();
  alpha.NewTensorPointwise();
  gam_uu.NewTensorPointwise();
  dgam_ddd.NewTensorPointwise();
  dbeta_du.NewTensorPointwise();
  dalpha_d.NewTensorPointwise();
  F_d.NewTensorPointwise();
  P_dd.NewTensorPointwise();
  P_uu.NewTensorPointwise();
  
  // Go through cells
  CLOOP3(k,j,i) {

    vc_z4c_alpha.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_alpha,1);
    vc_z4c_beta_u_x.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_betax,1);
    vc_z4c_beta_u_y.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_betay,1);
    vc_z4c_beta_u_z.InitWithShallowSlice(pmy_block->pz4c->storage.u, Z4c::I_Z4c_betaz,1);
    vc_gamma_xx.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxx,1);
    vc_gamma_xy.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxy,1);
    vc_gamma_xz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gxz,1);
    vc_gamma_yy.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gyy,1);
    vc_gamma_yz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gyz,1);
    vc_gamma_zz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_gzz,1);
    vc_K_xx.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kxx,1);
    vc_K_xy.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kxy,1);
    vc_K_xz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kxz,1);
    vc_K_yy.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kyy,1);
    vc_K_yz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kyz,1);
    vc_K_zz.InitWithShallowSlice(pmy_block->pz4c->storage.adm, Z4c::I_ADM_Kzz,1);

    alpha() = VCInterpolation(vc_z4c_alpha,k,j,i);
    beta_u(0) = VCInterpolation(vc_z4c_beta_u_x,k,j,i);
    beta_u(1) = VCInterpolation(vc_z4c_beta_u_y,k,j,i);
    beta_u(2) = VCInterpolation(vc_z4c_beta_u_z,k,j,i);
    gam_dd(0,0) = VCInterpolation(vc_gamma_xx,k,j,i);
    gam_dd(0,1) = VCInterpolation(vc_gamma_xy,k,j,i);
    gam_dd(0,2) = VCInterpolation(vc_gamma_xz,k,j,i);
    gam_dd(1,1) = VCInterpolation(vc_gamma_yy,k,j,i);
    gam_dd(1,2) = VCInterpolation(vc_gamma_yz,k,j,i);
    gam_dd(2,2) = VCInterpolation(vc_gamma_zz,k,j,i);
    K_dd(0,0) = VCInterpolation(vc_K_xx,k,j,i);
    K_dd(0,1) = VCInterpolation(vc_K_xy,k,j,i);
    K_dd(0,2) = VCInterpolation(vc_K_xz,k,j,i);
    K_dd(1,1) = VCInterpolation(vc_K_yy,k,j,i);
    K_dd(1,2) = VCInterpolation(vc_K_yz,k,j,i);
    K_dd(2,2) = VCInterpolation(vc_K_zz,k,j,i);
    for(int c = 0; c < NDIM; ++c) {
      dalpha_d(c) = idx[c]*VCDiff(c,vc_z4c_alpha,k,j,i);
      dbeta_du(c,0) = idx[c]*VCDiff(c,vc_z4c_beta_u_x,k,j,i);
      dbeta_du(c,1) = idx[c]*VCDiff(c,vc_z4c_beta_u_y,k,j,i);
      dbeta_du(c,2) = idx[c]*VCDiff(c,vc_z4c_beta_u_z,k,j,i);
      dgam_ddd(c,0,0) = idx[c]*VCDiff(c,vc_gamma_xx,k,j,i);
      dgam_ddd(c,0,1) = idx[c]*VCDiff(c,vc_gamma_xy,k,j,i);
      dgam_ddd(c,0,2) = idx[c]*VCDiff(c,vc_gamma_xz,k,j,i);
      dgam_ddd(c,1,1) = idx[c]*VCDiff(c,vc_gamma_yy,k,j,i);
      dgam_ddd(c,1,2) = idx[c]*VCDiff(c,vc_gamma_yz,k,j,i);
      dgam_ddd(c,2,2) = idx[c]*VCDiff(c,vc_gamma_zz,k,j,i);
    }

    // Inverse metric on CC
    SpatialInv(gam_dd, gam_uu);

    for (int ig = 0; ig < ngroups*nspecies; ++ig) {

      pack_F_d(vec.F_d(0,ig,k,j,i),vec.F_d(1,ig,k,j,i),vec.F_d(2,ig,k,j,i), F_d);
      pack_P_dd(rad.P_dd(0,0,ig,k,j,i),rad.P_dd(0,1,ig,k,j,i),rad.P_dd(0,2,ig,k,j,i),
                rad.P_dd(1,1,ig,k,j,i),rad.P_dd(1,2,ig,k,j,i),rad.P_dd(2,2,ig,k,j,i), P_dd);

      // Contravariant radiation pressure
      tensor::contract2(gam_uu, P_dd, P_uu);

      // Compute radiation energy sources
      // Note that everything is already densitized
      Real const rhsE = alpha() *tensor::dot(P_uu, K_dd) - tensor::dot(gam_uu, F_d, dalpha_d);
      vec_rhs.E(ig,k,j,i) += rhsE;

      // Compute the radiation flux sources
      for (int a = 0; a < NDIM; ++a) {
        Real rhsFd_a = -vec.E(ig,k,j,i) * dalpha_d(a);
        for (int b = 0; b < NDIM; ++b) {
          rhsFd_a += F_d(b) * dbeta_du(a,b);
          for (int c = 0; c < NDIM; ++c) {
            rhsFd_a += 0.5*alpha()* P_uu(b,c) * dgam_ddd(a,b,c);
          }
        }
        vec_rhs.F_d(a,ig,k,j,i) += rhsFd_a;
      }
    }
  } // CLOOP3
  gam_dd.DeleteTensorPointwise();
  K_dd.DeleteTensorPointwise();
  beta_u.DeleteTensorPointwise();
  alpha.DeleteTensorPointwise();
  gam_uu.DeleteTensorPointwise();
  dgam_ddd.DeleteTensorPointwise();
  dbeta_du.DeleteTensorPointwise();
  dalpha_d.DeleteTensorPointwise();
  F_d.DeleteTensorPointwise();
  P_dd.DeleteTensorPointwise();
  P_uu.DeleteTensorPointwise();
}




// void M1::GRSources(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
// {
//   if (M1_GRSOURCES_OFF) return;
//   M1_DEBUG_PR("in: GRSources");

//   MeshBlock *pmb = pmy_block;

//   Lab_vars vec, vec_rhs;
//   SetLabVarsAliases(u, vec);
//   SetLabVarsAliases(u_rhs, vec_rhs); 
  
//   // 3-tensors at (k,j,i) CC
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gam_dd;   //gamma_{ij}
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> K_dd;     // K_{ij}
//   TensorPointwise<Real, Symmetries::NONE, NDIM, 1> beta_u;   // beta^i
//   TensorPointwise<Real, Symmetries::NONE, NDIM, 0> alpha;    // lapse
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> gam_uu;   // gamma^{ij}
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 3> dgam_ddd; // pd_i gamma_{jk}
//   TensorPointwise<Real, Symmetries::NONE, NDIM, 2> dbeta_du; // pd_i beta^j
//   TensorPointwise<Real, Symmetries::NONE, NDIM, 1> dalpha_d; // pd_i alpha

//   TensorPointwise<Real, Symmetries::NONE, NDIM, 1> F_d;
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_dd;
//   TensorPointwise<Real, Symmetries::SYM2, NDIM, 2> P_uu;

//   gam_dd.NewTensorPointwise();
//   K_dd.NewTensorPointwise();
//   beta_u.NewTensorPointwise();
//   alpha.NewTensorPointwise();
//   gam_uu.NewTensorPointwise();
//   dgam_ddd.NewTensorPointwise();
//   dbeta_du.NewTensorPointwise();
//   dalpha_d.NewTensorPointwise();
//   F_d.NewTensorPointwise();
//   P_dd.NewTensorPointwise();
//   P_uu.NewTensorPointwise();
  
//   // Go through cells
//   CLOOP3(k,j,i) {

//     if (m1_mask(k,j,i)) {
//      continue;
//     }

//     Real const x = pmy_block->pcoord->x1v(i);
//     Real const y = pmy_block->pcoord->x2v(j);
//     Real const z = pmy_block->pcoord->x3v(k);
//     Real const xv[3] = {x, y, z};
//     Real const r = std::sqrt(x*x+y*y+z*z);
    
//     pmb->pz4c->ADMKerrSchild(x, y, z, alpha, beta_u, gam_dd, K_dd);
//     for (int a=0; a<3; ++a) {
//       dalpha_d(a) = POW3(alpha()/r) *xv[a];
//       for (int b=0; b<3; ++b) {
//         dbeta_du(b,a) = 4*SQR(alpha())*xv[a]*xv[b]/std::pow(r, 4) * (SQR(alpha())/std::pow(r,0.5) - 1) + 2*SQR(alpha()/r) * ( a == b ? 1. : 0.);
//         for (int c=0; c<3; ++c) {
//           dgam_ddd(c,a,b) = 2.0/POW3(r) * (xv[b]*(a==c?1:0)+xv[a]*(b==c?1:0)) - 6*xv[a]*xv[b]*xv[c]/std::pow(r, 5);
//         }
//       }
//     }

//     // Inverse metric on CC
//     Real const detg = SpatialDet(gam_dd);
//     SpatialInv(1.0/detg, gam_dd, gam_uu);

//     for (int ig = 0; ig < ngroups*nspecies; ++ig) {

//       pack_F_d(vec.F_d(0,ig,k,j,i),vec.F_d(1,ig,k,j,i),vec.F_d(2,ig,k,j,i), F_d);
//       pack_P_dd(rad.P_dd(0,0,ig,k,j,i),rad.P_dd(0,1,ig,k,j,i),rad.P_dd(0,2,ig,k,j,i),
//                 rad.P_dd(1,1,ig,k,j,i),rad.P_dd(1,2,ig,k,j,i),rad.P_dd(2,2,ig,k,j,i), P_dd);

//       // Contravariant radiation pressure
//       tensor::contract2(gam_uu, P_dd, P_uu);

//       // Compute radiation energy sources
//       // Note that everything is already densitized
//       Real const rhsE = alpha() *tensor::dot(P_uu, K_dd) - tensor::dot(gam_uu, F_d, dalpha_d);
//       vec_rhs.E(ig,k,j,i) += rhsE;

//       // Compute the radiation flux sources
//       for (int a = 0; a < NDIM; ++a) {
//         Real rhsFd_a = -vec.E(ig,k,j,i) * dalpha_d(a);
//         for (int b = 0; b < NDIM; ++b) {
//           rhsFd_a += F_d(b) * dbeta_du(a,b);
//           for (int c = 0; c < NDIM; ++c) {
//             rhsFd_a += 0.5*alpha()* P_uu(b,c) * dgam_ddd(a,b,c);
//           }
//         }
//         vec_rhs.F_d(a,ig,k,j,i) += rhsFd_a;
//       }
//     }
//   } // CLOOP3
//   gam_dd.DeleteTensorPointwise();
//   K_dd.DeleteTensorPointwise();
//   beta_u.DeleteTensorPointwise();
//   alpha.DeleteTensorPointwise();
//   gam_uu.DeleteTensorPointwise();
//   dgam_ddd.DeleteTensorPointwise();
//   dbeta_du.DeleteTensorPointwise();
//   dalpha_d.DeleteTensorPointwise();
//   F_d.DeleteTensorPointwise();
//   P_dd.DeleteTensorPointwise();
//   P_uu.DeleteTensorPointwise();
// }

	
