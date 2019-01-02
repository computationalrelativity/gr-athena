//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition

// C++ standard headers
#include <cmath> // exp, pow

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../mesh/mesh.hpp"

void Z4c::Z4cRHS(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs)
{
  Z4c_vars z4c, rhs;
  SetZ4cAliases(u, z4c);
  SetZ4cAliases(u_rhs, rhs);
  u_rhs.Zero();
  LOOP2(k,j) {
    // -----------------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      LOOP1(i) {
        dalpha_d(a,i) = FD.Dx(a, z4c.alpha(k,j,i));
        dchi_d(a,i) = FD.Dx(a, z4c.chi(k,j,i));
        dKhat_d(a,i) = FD.Dx(a, z4c.Khat(k,j,i));
        dTheta_d(a,i) = FD.Dx(a, z4c.Theta(k,j,i));
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      LOOP1(i) {
        dbeta_du(a,b,i) = FD.Dx(a, z4c.beta_u(b,k,j,i));
        dGam_du(a,b,i) = FD.Dx(a, z4c.Gam_u(b,k,j,i));
      }
    }
    // Tensors
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      LOOP1(i) {
        dg_ddd(c,a,b,i) = FD.Dx(c, z4c.g_dd(a,b,k,j,i));
        dA_ddd(c,a,b,i) = FD.Dx(c, z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // 2nd derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      LOOP1(i) {
        ddalpha_dd(a,a,i) = FD.Dxx(a, z4c.alpha(k,j,i));
        ddchi_dd(a,a,i) = FD.Dxx(a, z4c.chi(k,j,i));
      }
      for(int b = a + 1; b < NDIM; ++b) {
        LOOP1(i) {
          ddalpha_dd(a,b,i) = FD.Dxy(a, b, z4c.alpha(k,j,i));
          ddchi_dd(a,b,i) = FD.Dxy(a, b, z4c.chi(k,j,i));
        }
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      if(a == b) {
        LOOP1(i) {
          ddbeta_ddu(a,b,c,i) = FD.Dxx(a, z4c.beta_u(c,k,j,i));
        }
      }
      else {
        LOOP1(i) {
          ddbeta_ddu(a,b,c,i) = FD.Dxy(a, b, z4c.beta_u(c,k,j,i));
        }
      }
    }
    // Tensors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d) {
      if(a == b) {
        LOOP1(i) {
          ddg_dddd(a,b,c,d,i) = FD.Dxx(a, z4c.g_dd(c,d,k,j,i));
        }
      }
      else {
        LOOP1(i) {
          ddg_dddd(a,b,c,d,i) = FD.Dxy(a, b, z4c.g_dd(c,d,k,j,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Advective derivatives
    //
    // Scalars
    Lalpha.Zero();
    Lchi.Zero();
    LKhat.Zero();
    LTheta.Zero();
    for(int a = 0; a < NDIM; ++a) {
      LOOP1(i) {
        Lalpha(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.alpha(k,j,i));
        Lchi(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.chi(k,j,i));
        LKhat(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Khat(k,j,i));
        LTheta(i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Theta(k,j,i));
      }
    }
    // Vectors
    Lbeta_u.Zero();
    LGam_u.Zero();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      LOOP1(i) {
        Lbeta_u(b,i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.beta_u(b,k,j,i));
        LGam_u(b,i) += FD.Lx(a, z4c.beta_u(a,k,j,i), z4c.Gam_u(b,k,j,i));
      }
    }
    // Tensors
    Lg_dd.Zero();
    LA_dd.Zero();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      LOOP1(i) {
        Lg_dd(a,b,i) += FD.Lx(c, z4c.beta_u(c,k,j,i), z4c.g_dd(a,b,k,j,i));
        LA_dd(a,b,i) += FD.Lx(c, z4c.beta_u(c,k,j,i), z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // Get K from Khat
    //
    LOOP1(i) {
      K(i) = z4c.Khat(k,j,i) + 2*z4c.Theta(k,j,i);
    }
    for(int a = 0; a < NDIM; ++a) {
      LOOP1(i) {
        dK_d(a,i) = dKhat_d(a,i) + 2*dTheta_d(a,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Inverse metric
    //
    LOOP1(i) {
      detg(i) = SpatialDet(z4c.g_dd, k, j, i);
      SpatialInv(1.0/detg(i),
          z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
          z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
          &g_uu(0,0,k,j,i),    &g_uu(0,1,k,j,i),    &g_uu(0,2,k,j,i),
          &g_uu(1,1,k,j,i),    &g_uu(1,2,k,j,i),    &g_uu(2,2,k,j,i));
    }
    dg_duu.Zero();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d)
    for(int e = 0; e < NDIM; ++e) {
      LOOP1(i) {
        dg_duu(a,b,c,i) -= g_uu(b,d,i) * g_uu(c,e,i) * dg_ddd(a,d,e,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      LOOP1(i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
        Gamma_udd(c,a,b,i) = 0.0;
      }
    }
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      LOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }
    Gamma_u.Zero();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      LOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Curvature of conformal metric
    //
    R_dd.Zero();
#warning "TODO: finish to implement me!"
  }
}
