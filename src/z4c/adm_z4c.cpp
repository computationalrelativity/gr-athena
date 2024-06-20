//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition

// C++ standard headers
#include <cmath> // pow
#include <iostream>
#include <fstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMToZ4c(AthenaArray<Real> & u_adm, AthenaArray<Real> & u)
// \brief Compute Z4c variables from ADM variables
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
// BAM: Z4c_init()
// https://git.tpi.uni-jena.de/bamdev/z4
// https://git.tpi.uni-jena.de/bamdev/z4/blob/master/z4_init.m
//
// The Z4c variables will be set on the whole MeshBlock with the exception of
// the Gamma's that can only be set in the interior of the MeshBlock.

#ifndef DBG_EOM
void Z4c::ADMToZ4c(AthenaArray<Real> & u_adm, AthenaArray<Real> & u) {
  using namespace LinearAlgebra;

  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  //--------------------------------------------------------------------------------------
  // Conformal factor, conformal metric, and trace of extrinsic curvature
  //
  GLOOP2(k,j) {

    GLOOP1(i) {
      z4c.alpha(k,j,i) = adm.alpha(k,j,i);
    }

    for(int a = 0; a < NDIM; ++a)
    GLOOP1(i) {
      z4c.beta_u(a,k,j,i) = adm.beta_u(a,k,j,i);
    }

    // Conformal factor
    if (opt.chi_psi_power == -4.)
    {
      GLOOP1(i) {
        detg(i) = Det3Metric(adm.g_dd, k, j, i);
        // oopsi4(i) = pow(detg(i), -1./3.);
        // z4c.chi(k,j,i) = pow(detg(i), 1./12.*opt.chi_psi_power);

        oopsi4(i) = std::cbrt(1.0 / detg(i));
        z4c.chi(k,j,i) = oopsi4(i);
      }
    }
    else
    {
      GLOOP1(i) {
        detg(i) = Det3Metric(adm.g_dd, k, j, i);
        oopsi4(i) = pow(detg(i), -1./3.);
        z4c.chi(k,j,i) = pow(detg(i), 1./12.*opt.chi_psi_power);
      }
    }

    // Conformal metric and extrinsic curvature
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      GLOOP1(i) {
        z4c.g_dd(a,b,k,j,i) = oopsi4(i) * adm.g_dd(a,b,k,j,i);
        Kt_dd(a,b,i)        = oopsi4(i) * adm.K_dd(a,b,k,j,i);
      }
    }

    // Determinant of the conformal metric and trace of conf. extr. curvature
    GLOOP1(i) {
      detg(i) = Det3Metric(z4c.g_dd, k, j, i);
      z4c.Khat(k,j,i) = TraceRank2(1.0/detg(i),
          z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
          z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
          Kt_dd(0,0,i), Kt_dd(0,1,i), Kt_dd(0,2,i),
          Kt_dd(1,1,i), Kt_dd(1,2,i), Kt_dd(2,2,i));
    }

    // Conformal traceless extrinsic curvatore
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      GLOOP1(i) {
          z4c.A_dd(a,b,k,j,i) = Kt_dd(a,b,i) - (1./3.) * z4c.Khat(k,j,i) * z4c.g_dd(a,b,k,j,i);
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // Gamma's
  //
  // Allocate temporary memory for the inverse conformal metric
  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> g_uu;
  g_uu.NewAthenaTensor(pz4c->mbi.nn3, pz4c->mbi.nn2, pz4c->mbi.nn1);

  // Inverse conformal metric
  GLOOP3(k,j,i) {
    detg(i) = Det3Metric(z4c.g_dd, k, j, i);
    Inv3Metric(1.0/detg(i),
        z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
        z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
        &g_uu(0,0,k,j,i),    &g_uu(0,1,k,j,i),    &g_uu(0,2,k,j,i),
        &g_uu(1,1,k,j,i),    &g_uu(1,2,k,j,i),    &g_uu(2,2,k,j,i));
  }

  // Compute Gamma's
  z4c.Gam_u.ZeroClear();
  ILOOP2(k,j) {
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        // Is it ba or ab like in the pseudocode? Is the contraction correct?
        z4c.Gam_u(a,k,j,i) -= fd->Dx(b, g_uu(b,a,k,j,i));
      }
    }
  }

  g_uu.DeleteAthenaTensor();

  //--------------------------------------------------------------------------------------
  // Theta
  //
  z4c.Theta.ZeroClear();

  //--------------------------------------------------------------------------------------
  // Algebraic constraints enforcement
  //
  AlgConstr(u);
}
#endif // DBG_EOM

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm)
// \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock

#ifndef DBG_EOM
void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm) {
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  GLOOP2(k,j) {
    GLOOP1(i) {
      adm.alpha(k,j,i) = z4c.alpha(k,j,i);
    }

    for(int a = 0; a < NDIM; ++a)
    GLOOP1(i) {
      adm.beta_u(a,k,j,i) = z4c.beta_u(a,k,j,i);
    }

    // psi4
    GLOOP1(i) {
      adm.psi4(k,j,i) = std::pow(z4c.chi(k,j,i), 4./opt.chi_psi_power);
    }
    // g_ab
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      GLOOP1(i) {
        adm.g_dd(a,b,k,j,i) = adm.psi4(k,j,i) * z4c.g_dd(a,b,k,j,i);
      }
    }
    // K_ab
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    {
      GLOOP1(i) {
        adm.K_dd(a,b,k,j,i) = adm.psi4(k,j,i) * z4c.A_dd(a,b,k,j,i) +
          (1./3.) * (z4c.Khat(k,j,i) + 2.*z4c.Theta(k,j,i)) * adm.g_dd(a,b,k,j,i);
      }
    }
  }
}
#endif // DBG_EOM

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
// \brief compute constraints ADM vars
//
// Note: we are assuming that u_adm has been initialized with the correct
// metric and matter quantities
//
// BAM: adm_constraints_N()
// https://git.tpi.uni-jena.de/bamdev/adm
// https://git.tpi.uni-jena.de/bamdev/adm/blob/master/adm_constraints_N.m
//
// The constraints are set only in the MeshBlock interior, because derivatives
// of the ADM quantities are neded to compute them.

void Z4c::ADMConstraints(
  AthenaArray<Real> & u_con, AthenaArray<Real> & u_adm,
  AthenaArray<Real> & u_mat, AthenaArray<Real> & u_z4c)
{
  using namespace LinearAlgebra;
  u_con.ZeroClear();

  Constraint_vars con;
  SetConstraintAliases(u_con, con);

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  Z4c_vars z4c;
  SetZ4cAliases(u_z4c, z4c);

  ILOOP2(k,j) {
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = fd->Dx(c, adm.g_dd(a,b,k,j,i));
        dK_ddd(c,a,b,i) = fd->Dx(c, adm.K_dd(a,b,k,j,i));
      }
    }
    // second derivatives of g
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d) {
      if(a == b) {
        ILOOP1(i) {
          ddg_dddd(a,a,c,d,i) = fd->Dxx(a, adm.g_dd(c,d,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = fd->Dxy(a, b, adm.g_dd(c,d,k,j,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    ILOOP1(i) {
      detg(i) = Det3Metric(adm.g_dd,k,j,i);
      Inv3Metric(1./detg(i),
          adm.g_dd(0,0,k,j,i), adm.g_dd(0,1,k,j,i), adm.g_dd(0,2,k,j,i),
          adm.g_dd(1,1,k,j,i), adm.g_dd(1,2,k,j,i), adm.g_dd(2,2,k,j,i),
          &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
          &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    }

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
      }
    }

    Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }

    Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //

    /*
    R.ZeroClear();
    R_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < NDIM; ++e) {
          ILOOP1(i) {
            R_dd(a,b,i) += g_uu(c,d,i) * Gamma_udd(e,a,c,i) * Gamma_ddd(e,b,d,i);
            R_dd(a,b,i) -= g_uu(c,d,i) * Gamma_udd(e,a,b,i) * Gamma_ddd(e,c,d,i);
          }
        }
        // Wave operator part of the Ricci
        ILOOP1(i) {
          R_dd(a,b,i) += 0.5*g_uu(c,d,i)*(
              - ddg_dddd(c,d,a,b,i) - ddg_dddd(a,b,c,d,i) +
                ddg_dddd(a,c,b,d,i) + ddg_dddd(b,c,a,d,i));
        }
      }
      ILOOP1(i) {
        R(i) += g_uu(a,b,i) * R_dd(a,b,i);
      }
    }
    */

    R_dd.ZeroClear();
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    {
      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      for (int f=0; f<NDIM; ++f)
      ILOOP1(i)
      {
        R_dd(a,b,i) += g_uu(d,f,i) * (
          Gamma_udd(c,a,f,i) * Gamma_ddd(c,b,d,i) -
          Gamma_udd(c,a,b,i) * Gamma_ddd(c,f,d,i)
        );
      }

      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        R_dd(a,b,i) += 0.5 * g_uu(c,d,i) * (
          -ddg_dddd(b,a,c,d,i)
          +ddg_dddd(d,a,b,c,i)
          +ddg_dddd(d,b,a,c,i)
          -ddg_dddd(d,c,a,b,i)
        );
      }
    }

    R.ZeroClear();
    for (int a=0; a<NDIM; ++a)
    for (int b=0; b<NDIM; ++b)
    ILOOP1(i)
    {
      R(i) += g_uu(a,b,i) * R_dd(a,b,i);
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
        for(int c = 0; c < NDIM; ++c) {
          ILOOP1(i) {
            K_ud(a,b,i) += g_uu(a,c,i) * adm.K_dd(c,b,k,j,i);
          }
        }
      }
      ILOOP1(i) {
        K(i) += K_ud(a,a,i);
      }
    }
    // K^a_b K^b_a
    KK.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        KK(i) += K_ud(a,b,i) * K_ud(b,a,i);
      }
    }
    // Covariant derivative of K
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c) {
      ILOOP1(i) {
        DK_ddd(a,b,c,i) = dK_ddd(a,b,c,i);
      }
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,b,i) * adm.K_dd(d,c,k,j,i);
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,c,i) * adm.K_dd(b,d,k,j,i);
        }
      }
    }
    DK_udd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    ILOOP1(i) {
      con.H(k,j,i) = R(i) + SQR(K(i)) - KK(i) - 16*M_PI * mat.rho(k,j,i);
    }
    // Momentum constraint (contravariant)
    //
    M_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        M_u(a,i) -= 8*M_PI * g_uu(a,b,i) * mat.S_d(b,k,j,i);
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          M_u(a,i) += g_uu(a,b,i) * DK_udd(c,b,c,i);
          M_u(a,i) -= g_uu(b,c,i) * DK_udd(a,b,c,i);
        }
      }
    }
    // Momentum constraint (covariant)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.M_d(a,k,j,i) += adm.g_dd(a,b,k,j,i) * M_u(b,i);
      }
    }
    // Momentum constraint (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.M(k,j,i) += adm.g_dd(a,b,k,j,i) * M_u(a,i) * M_u(b,i);
      }
    }
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = fd->Dx(c, z4c.g_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    ILOOP1(i) {
      detg(i) = Det3Metric(z4c.g_dd,k,j,i);
      Inv3Metric(1./detg(i),
          z4c.g_dd(0,0,k,j,i), z4c.g_dd(0,1,k,j,i), z4c.g_dd(0,2,k,j,i),
          z4c.g_dd(1,1,k,j,i), z4c.g_dd(1,2,k,j,i), z4c.g_dd(2,2,k,j,i),
          &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
          &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    }

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
      }
    }

    Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      }
    }

    Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }
    // Constraint violation Z (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        con.Z(k,j,i) += 0.25*adm.g_dd(a,b,k,j,i)*(z4c.Gam_u(a,k,j,i) - Gamma_u(a,i))
                                                *(z4c.Gam_u(b,k,j,i) - Gamma_u(b,i));
      }
    }
    // Constraint violation monitor C^2
    ILOOP1(i) {
      con.C(k,j,i) = SQR(con.H(k,j,i)) + con.M(k,j,i) + SQR(z4c.Theta(k,j,i)) + 4.0*con.Z(k,j,i);
    }
  }


  // zero out values beyond fixed radius
  ILOOP3(k,j,i)
  {
    const Real R = std::sqrt(SQR(mbi.x1(i)) + SQR(mbi.x2(j)) + SQR(mbi.x3(k)));

    if (R > pz4c->opt.r_max_con)
    {
      con.Z(k,j,i) = 0.;
      con.C(k,j,i) = 0.;
      con.H(k,j,i) = 0.;
      con.M(k,j,i) = 0.;
      for(int a = 0; a < NDIM; ++a)
      {
        con.M_d(a,k,j,i) = 0.;
      }
    }
  }

}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMMinkowski(AthenaArray<Real> & u)
// \brief Initialize ADM vars to Minkowski

void Z4c::ADMMinkowski(AthenaArray<Real> & u_adm) {
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  adm.psi4.Fill(1.);
  adm.K_dd.ZeroClear();

  GLOOP3(k,j,i) {
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      adm.g_dd(a,b,k,j,i) = (a == b ? 1. : 0.);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::MatterVacuum(AthenaArray<Real> & u_mat)
// \brief Initialize ADM vars to vacuum

void Z4c::MatterVacuum(AthenaArray<Real> & u_mat) {
  u_mat.ZeroClear();
}


// Debug EOM ------------------------------------------------------------------
// H&R '2018
#ifdef DBG_EOM

// For readability
typedef AthenaArray< Real>                            AA;
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> AT_N_sym;

// symmetric tensor derivatives
typedef AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> AT_N_D1sym;

// Need to populate
// Z4c: chi, g_dd, Khat, A_dd, Gam_u (i.e. Gamtil_U), Theta (take as 0)
void Z4c::ADMToZ4c(AthenaArray<Real> & u_adm, AthenaArray<Real> & u)
{
  using namespace LinearAlgebra;

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sca adm_alpha( u_adm, Z4c::I_ADM_alpha);
  AT_N_vec adm_beta_u(u_adm, Z4c::I_ADM_betax);

  AT_N_sym adm_gamma_dd(u_adm, Z4c::I_ADM_gxx);
  AT_N_sym adm_K_dd(    u_adm, Z4c::I_ADM_Kxx);

  AT_N_sca z4c_alpha(u, Z4c::I_Z4c_alpha);
  AT_N_sca z4c_chi(  u, Z4c::I_Z4c_chi);
  AT_N_sca z4c_Theta(u, Z4c::I_Z4c_Theta);
  AT_N_sca z4c_Khat( u, Z4c::I_Z4c_Khat);

  AT_N_vec z4c_Gamtil_u(   u, Z4c::I_Z4c_Gamx);
  AT_N_vec z4c_beta_u(     u, Z4c::I_Z4c_betax);

  AT_N_sym z4c_Atil_dd(    u, Z4c::I_Z4c_Axx);
  AT_N_sym z4c_gammatil_dd(u, Z4c::I_Z4c_gxx);

  // Scratch & parameters -----------------------------------------------------
  AT_N_sca adm_detgamma_(mbi.nn1);
  AT_N_sca adm_oopsi4_(  mbi.nn1);
  AT_N_sca adm_K_(       mbi.nn1);

  AT_N_sca z4c_detgamma_(   mbi.nn1);
  AT_N_sca z4c_oodetgamma_( mbi.nn1);

  AT_N_sym z4c_gammatil_uu_(mbi.nn1);

  // derivatives
  AT_N_D1sym z4c_dgammatil_ddd_(mbi.nn1);

  const int il = 0;
  const int iu = mbi.nn1-1;

  const Real oo12 = 1.0 / 12.;
  const Real chi_detgamma_pow = oo12 * opt.chi_psi_power;

  // Map quantities -----------------------------------------------------------

  // Theta
  z4c_Theta.ZeroClear();

  GLOOP2(k,j)
  {

    GLOOP1(i) {
      z4c_alpha(k,j,i) = adm_alpha(k,j,i);
    }

    for(int a = 0; a < NDIM; ++a)
    GLOOP1(i) {
      z4c_beta_u(a,k,j,i) = adm_beta_u(a,k,j,i);
    }

    // z4c_chi
    Det3Metric(adm_detgamma_, adm_gamma_dd, k, j, il, iu);

    GLOOP1(i)
    {
      z4c_chi(k,j,i) = std::pow(adm_detgamma_(i), chi_detgamma_pow);
    }

    // z4c_Khat, adm_K_
    TraceRank2(adm_K_, adm_detgamma_, adm_gamma_dd, adm_K_dd, k, j, il, iu);
    GLOOP1(i)
    {
      z4c_Khat(k,j,i) = adm_K_(i) - 2.0*z4c_Theta(k,j,i);
    }

    // z4c_gammatil_dd
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    GLOOP1(i)
    {
      z4c_gammatil_dd(a,b,k,j,i) = z4c_chi(k,j,i) * adm_gamma_dd(a,b,k,j,i);
    }

    // z4c_Atil_dd
    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    GLOOP1(i)
    {
      z4c_Atil_dd(a,b,k,j,i) = z4c_chi(k,j,i) * (
        adm_K_dd(a,b,k,j,i) - ONE_3RD * adm_gamma_dd(a,b,k,j,i) * adm_K_(i)
      );
    }
  }

  z4c_Gamtil_u.ZeroClear();
  ILOOP2(k,j)
  {
    // z4c_detgamma_, z4c_gamma_dd
    Det3Metric(z4c_detgamma_, z4c_gammatil_dd, k, j, mbi.il, mbi.iu);

    ILOOP1(i)
    {
      z4c_oodetgamma_(i) = 1.0 / z4c_detgamma_(i);
    }

    Inv3Metric(z4c_oodetgamma_, z4c_gammatil_dd, z4c_gammatil_uu_,
               k, j, mbi.il, mbi.iu);

    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    for (int c=0; c<NDIM; ++c)
    ILOOP1(i)
    {
      z4c_dgammatil_ddd_(c,a,b,i) = fd->Dx(c, z4c_gammatil_dd(a,b,k,j,i));
    }

    for (int a=0; a<NDIM; a++)
    for (int b=0; b<NDIM; b++)
    for (int c=0; c<NDIM; c++)
    for (int d=0; d<NDIM; d++)
    ILOOP1(i)
    {
      z4c_Gamtil_u(a,k,j,i) += (
        z4c_gammatil_uu_(a,b,i) * z4c_gammatil_uu_(c,d,i)
      ) * z4c_dgammatil_ddd_(d,b,c,i);
    }

  }

  //---------------------------------------------------------------------------
  // Enforce algebraic constraints
  AlgConstr(u);
}

void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm)
{

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sym adm_gamma_dd(u_adm, Z4c::I_ADM_gxx);
  AT_N_sym adm_K_dd(    u_adm, Z4c::I_ADM_Kxx);
  AT_N_sca adm_psi4(    u_adm, Z4c::I_ADM_psi4);

  AT_N_sca adm_alpha( u_adm, Z4c::I_ADM_alpha);
  AT_N_vec adm_beta_u(u_adm, Z4c::I_ADM_betax);

  AT_N_sca z4c_chi(  u, Z4c::I_Z4c_chi);
  AT_N_sca z4c_Theta(u, Z4c::I_Z4c_Theta);
  AT_N_sca z4c_Khat( u, Z4c::I_Z4c_Khat);

  AT_N_sca z4c_alpha( u, Z4c::I_Z4c_alpha);
  AT_N_vec z4c_beta_u(u, Z4c::I_Z4c_betax);

  AT_N_sym z4c_Atil_dd(    u, Z4c::I_Z4c_Axx);
  AT_N_sym z4c_gammatil_dd(u, Z4c::I_Z4c_gxx);

  // Map quantities -----------------------------------------------------------
  GLOOP2(k,j)
  {

    GLOOP1(i)
    {
      adm_alpha(k,j,i) = z4c_alpha(k,j,i);
    }

    for (int a=0; a<NDIM; ++a)
    GLOOP1(i)
    {
      adm_beta_u(a,k,j,i) = z4c_beta_u(a,k,j,i);
    }

    GLOOP1(i)
    {
      adm_psi4(k,j,i) = std::pow(z4c_chi(k,j,i), 4.0 / opt.chi_psi_power);
    }

    for (int a=0; a<NDIM; ++a)
    for (int b=a; b<NDIM; ++b)
    GLOOP1(i)
    {
      adm_gamma_dd(a,b,k,j,i) = z4c_gammatil_dd(a,b,k,j,i) / z4c_chi(k,j,i);
      adm_K_dd(    a,b,k,j,i) = (
        z4c_Atil_dd(a,b,k,j,i) / z4c_chi(k,j,i) +
        ONE_3RD * adm_gamma_dd(a,b,k,j,i) * (
          z4c_Khat(k,j,i) + 2.0 * z4c_Theta(k,j,i)
        )
      );
    }
  }

}

#endif // DBG_EOM