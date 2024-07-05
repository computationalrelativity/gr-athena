//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition

// C++ standard headers
#include <algorithm> // max
#include <cmath> // exp, pow, sqrt
#include <iomanip>
#include <iostream>
#include <fstream>

// Athena++ headers
#include "z4c.hpp"
#include "z4c_macro.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"

// External libraries
#ifdef GSL
#include <gsl/gsl_sf_bessel.h>   // Bessel functions
#endif

#include "puncture_tracker.hpp"


//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cRHS(AthenaArray<Real> & u, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_rhs)
// \brief compute the RHS given the state vector and matter state
//
// This function operates only on the interior points of the MeshBlock

#ifndef DBG_EOM
void Z4c::Z4cRHS(
  AthenaArray<Real> & u, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_rhs)
{
  using namespace LinearAlgebra;

  Z4c_vars z4c, rhs;
  SetZ4cAliases(u, z4c);
  SetZ4cAliases(u_rhs, rhs);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  //---------------------------------------------------------------------------
  // Scratch arrays for spatially dependent eta shift damping
#if defined(Z4C_ETA_CONF)
  int nn1 = pz4c->mbi.nn1;
  // 1/psi^2 (guarded); derivative and shift eta scratch
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> oopsi2;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> doopsi2_d;
  AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> shift_eta_spa;

  oopsi2.NewAthenaTensor(nn1);
  doopsi2_d.NewAthenaTensor(nn1);
  shift_eta_spa.NewAthenaTensor(nn1);

#elif defined(Z4C_ETA_TRACK_TP)

#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP
  //---------------------------------------------------------------------------

  ILOOP2(k,j) {
    // -----------------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dalpha_d(a,i) = fd->Dx(a, z4c.alpha(k,j,i));
        dchi_d(a,i)   = fd->Dx(a, z4c.chi(k,j,i));
        dKhat_d(a,i)  = fd->Dx(a, z4c.Khat(k,j,i));
        dTheta_d(a,i) = fd->Dx(a, z4c.Theta(k,j,i));
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        dbeta_du(b,a,i) = fd->Dx(b, z4c.beta_u(a,k,j,i));
        dGam_du(b,a,i)  = fd->Dx(b, z4c.Gam_u(a,k,j,i));
      }
    }
    // Tensors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = fd->Dx(c, z4c.g_dd(a,b,k,j,i));
        dA_ddd(c,a,b,i) = fd->Dx(c, z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // 2nd derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        ddalpha_dd(a,a,i) = fd->Dxx(a, z4c.alpha(k,j,i));
        ddchi_dd(a,a,i) = fd->Dxx(a, z4c.chi(k,j,i));
      }
      for(int b = a + 1; b < NDIM; ++b) {
        ILOOP1(i) {
          ddalpha_dd(a,b,i) = fd->Dxy(a, b, z4c.alpha(k,j,i));
          ddchi_dd(a,b,i) = fd->Dxy(a, b, z4c.chi(k,j,i));
        }
      }
    }
    // Vectors
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      if(a == b) {
        ILOOP1(i) {
          ddbeta_ddu(a,b,c,i) = fd->Dxx(a, z4c.beta_u(c,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddbeta_ddu(a,b,c,i) = fd->Dxy(a, b, z4c.beta_u(c,k,j,i));
        }
      }
    }
    // Tensors
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      if(a == b) {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = fd->Dxx(a, z4c.g_dd(c,d,k,j,i));
        }
      }
      else {
        ILOOP1(i) {
          ddg_dddd(a,b,c,d,i) = fd->Dxy(a, b, z4c.g_dd(c,d,k,j,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Advective derivatives
    //
    // Scalars
    Lalpha.ZeroClear();
    Lchi.ZeroClear();
    LKhat.ZeroClear();
    LTheta.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        Lalpha(i) += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.alpha(k,j,i));
        Lchi(i)   += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.chi(k,j,i)  );
        LKhat(i)  += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.Khat(k,j,i) );
        LTheta(i) += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.Theta(k,j,i));
      }
    }
    // Vectors
    Lbeta_u.ZeroClear();
    LGam_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Lbeta_u(b,i) += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.beta_u(b,k,j,i));
        LGam_u(b,i)  += fd->Lx(a, z4c.beta_u(a,k,j,i), z4c.Gam_u(b,k,j,i));
      }
    }
    // Tensors
    Lg_dd.ZeroClear();
    LA_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Lg_dd(a,b,i) += fd->Lx(c, z4c.beta_u(c,k,j,i), z4c.g_dd(a,b,k,j,i));
        LA_dd(a,b,i) += fd->Lx(c, z4c.beta_u(c,k,j,i), z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // Get K from Khat
    //
    ILOOP1(i) {
      K(i) = z4c.Khat(k,j,i) + 2.*z4c.Theta(k,j,i);
    }

    // -----------------------------------------------------------------------------------
    // Inverse metric
    //
    ILOOP1(i) {
      detg(i) = Det3Metric(z4c.g_dd, k, j, i);
      Inv3Metric(1.0/detg(i),
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
    // Gamma's computed from the conformal metric (not evolved)
    Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      ILOOP1(i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Curvature of conformal metric
    //
    R_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          R_dd(a,b,i) += 0.5*(z4c.g_dd(c,a,k,j,i)*dGam_du(b,c,i) +
                              z4c.g_dd(c,b,k,j,i)*dGam_du(a,c,i) +
                              Gamma_u(c,i)*(Gamma_ddd(a,b,c,i) + Gamma_ddd(b,a,c,i)));
        }
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          R_dd(a,b,i) -= 0.5*g_uu(c,d,i)*ddg_dddd(c,d,a,b,i);
        }
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d)
      for(int e = 0; e < NDIM; ++e) {
        ILOOP1(i) {
          R_dd(a,b,i) += g_uu(c,d,i)*(
              Gamma_udd(e,c,a,i)*Gamma_ddd(b,e,d,i) +
              Gamma_udd(e,c,b,i)*Gamma_ddd(a,e,d,i) +
              Gamma_udd(e,a,d,i)*Gamma_ddd(e,c,b,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Derivatives of conformal factor phi
    //
    ILOOP1(i) {
      chi_guarded(i) = std::max(z4c.chi(k,j,i), opt.chi_div_floor);
      oopsi4(i) = pow(chi_guarded(i), -4./opt.chi_psi_power);
    }

    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dphi_d(a,i) = dchi_d(a,i)/(chi_guarded(i) * opt.chi_psi_power);
      }
    }
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Real const ddphi_ab = ddchi_dd(a,b,i)/(chi_guarded(i) * opt.chi_psi_power) -
          opt.chi_psi_power * dphi_d(a,i) * dphi_d(b,i);
        Ddphi_dd(a,b,i) = ddphi_ab;
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Ddphi_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dphi_d(c,i);
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Curvature contribution from conformal factor
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Rphi_dd(a,b,i) = 4.*dphi_d(a,i)*dphi_d(b,i) - 2.*Ddphi_dd(a,b,i);
      }
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        ILOOP1(i) {
          Rphi_dd(a,b,i) -= 2.*z4c.g_dd(a,b,k,j,i) * g_uu(c,d,i)*(Ddphi_dd(c,d,i) +
              2.*dphi_d(c,i)*dphi_d(d,i));
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Trace of the matter stress tensor
    //
    S.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        S(i) += oopsi4(i) * g_uu(a,b,i) * mat.S_dd(a,b,k,j,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // 2nd covariant derivative of the lapse
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Ddalpha_dd(a,b,i) = ddalpha_dd(a,b,i)
                          - 2.*(dphi_d(a,i)*dalpha_d(b,i) + dphi_d(b,i)*dalpha_d(a,i));
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Ddalpha_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dalpha_d(c,i);
        }
        for(int d = 0; d < NDIM; ++d) {
          ILOOP1(i) {
            Ddalpha_dd(a,b,i) += 2.*z4c.g_dd(a,b,k,j,i) * g_uu(c,d,i) * dphi_d(c,i) * dalpha_d(d,i);
          }
        }
      }
    }

    Ddalpha.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        Ddalpha(i) += oopsi4(i) * g_uu(a,b,i) * Ddalpha_dd(a,b,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Contractions of A_ab, inverse, and derivatives
    //
    AA_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        AA_dd(a,b,i) += g_uu(c,d,i) * z4c.A_dd(a,c,k,j,i) * z4c.A_dd(d,b,k,j,i);
      }
    }
    AA.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        AA(i) += g_uu(a,b,i) * AA_dd(a,b,i);
      }
    }
    A_uu.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        A_uu(a,b,i) += g_uu(a,c,i) * g_uu(b,d,i) * z4c.A_dd(c,d,k,j,i);
      }
    }
    DA_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          DA_u(a,i) -= (3./2.) * A_uu(a,b,i) * dchi_d(b,i) / chi_guarded(i);
          DA_u(a,i) -= (1./3.) * g_uu(a,b,i) * (2.*dKhat_d(b,i) + dTheta_d(b,i));
        }
      }
      for(int b = 0; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          DA_u(a,i) += Gamma_udd(a,b,c,i) * A_uu(b,c,i);
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci scalar
    //
    R.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        R(i) += oopsi4(i) * g_uu(a,b,i) * (R_dd(a,b,i) + Rphi_dd(a,b,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // Hamiltonian constraint
    //
    ILOOP1(i) {
      Ht(i) = R(i) + (2./3.)*SQR(K(i)) - AA(i);
    }

    // -----------------------------------------------------------------------------------
    // Finalize advective (Lie) derivatives
    //
    // Shift vector contractions
    dbeta.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dbeta(i) += dbeta_du(a,a,i);
      }
    }
    ddbeta_d.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        ddbeta_d(a,i) += (1./3.) * ddbeta_ddu(a,b,b,i);
      }
    }
    // Finalize Lchi
    ILOOP1(i) {
      Lchi(i) += (1./6.) * opt.chi_psi_power * chi_guarded(i) * dbeta(i);
    }
    // Finalize LGam_u (note that this is not a real Lie derivative)
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        LGam_u(a,i) += (2./3.) * Gamma_u(a,i) * dbeta(i);
      }
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          LGam_u(a,i) += g_uu(a,b,i) * ddbeta_d(b,i) - Gamma_u(b,i) * dbeta_du(b,a,i);
        }
        for(int c = 0; c < NDIM; ++c) {
          ILOOP1(i) {
            LGam_u(a,i) += g_uu(b,c,i) * ddbeta_ddu(b,c,a,i);
          }
        }
      }
    }
    // Finalize Lg_dd and LA_dd
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        Lg_dd(a,b,i) -= (2./3.) * z4c.g_dd(a,b,k,j,i) * dbeta(i);
        LA_dd(a,b,i) -= (2./3.) * z4c.A_dd(a,b,k,j,i) * dbeta(i);
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          Lg_dd(a,b,i) += dbeta_du(a,c,i) * z4c.g_dd(b,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(b,c,i) * z4c.A_dd(a,c,k,j,i);
          Lg_dd(a,b,i) += dbeta_du(b,c,i) * z4c.g_dd(a,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(a,c,i) * z4c.A_dd(b,c,k,j,i);
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Assemble RHS
    //

    // Khat, chi, and Theta
    ILOOP1(i) {
      rhs.Khat(k,j,i) = - Ddalpha(i) + z4c.alpha(k,j,i) * (AA(i) + (1./3.)*SQR(K(i))) +
        LKhat(i) + opt.damp_kappa1*(1 - opt.damp_kappa2) * z4c.alpha(k,j,i) * z4c.Theta(k,j,i);
      rhs.Khat(k,j,i) += 4*M_PI * z4c.alpha(k,j,i) * (S(i) + mat.rho(k,j,i));
      rhs.chi(k,j,i) = Lchi(i) - (1./6.) * opt.chi_psi_power *
        chi_guarded(i) * z4c.alpha(k,j,i) * K(i);
      rhs.Theta(k,j,i) = LTheta(i) + z4c.alpha(k,j,i) * (
          0.5*Ht(i) - (2. + opt.damp_kappa2) * opt.damp_kappa1 * z4c.Theta(k,j,i));
      rhs.Theta(k,j,i) -= 8.*M_PI * z4c.alpha(k,j,i) * mat.rho(k,j,i);
    }
    // Gamma's
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        rhs.Gam_u(a,k,j,i) = 2.*z4c.alpha(k,j,i)*DA_u(a,i) + LGam_u(a,i);
        rhs.Gam_u(a,k,j,i) -= 2.*z4c.alpha(k,j,i) * opt.damp_kappa1 *
            (z4c.Gam_u(a,k,j,i) - Gamma_u(a,i));
      }
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          rhs.Gam_u(a,k,j,i) -= 2. * A_uu(a,b,i) * dalpha_d(b,i);
          rhs.Gam_u(a,k,j,i) -= 16.*M_PI * z4c.alpha(k,j,i) * g_uu(a,b,i) * mat.S_d(b,k,j,i);
        }
      }
    }
    // g and A
    //LOOK
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      ILOOP1(i) {
        rhs.g_dd(a,b,k,j,i) = - 2. * z4c.alpha(k,j,i) * z4c.A_dd(a,b,k,j,i) + Lg_dd(a,b,i);
        rhs.A_dd(a,b,k,j,i) = oopsi4(i) *
            (-Ddalpha_dd(a,b,i) + z4c.alpha(k,j,i) * (R_dd(a,b,i) + Rphi_dd(a,b,i)));
        rhs.A_dd(a,b,k,j,i) -= (1./3.) * z4c.g_dd(a,b,k,j,i) * (-Ddalpha(i) + z4c.alpha(k,j,i)*R(i));
        rhs.A_dd(a,b,k,j,i) += z4c.alpha(k,j,i) * (K(i)*z4c.A_dd(a,b,k,j,i) - 2.*AA_dd(a,b,i));
        rhs.A_dd(a,b,k,j,i) += LA_dd(a,b,i);
        rhs.A_dd(a,b,k,j,i) -= 8.*M_PI * z4c.alpha(k,j,i) *
            (oopsi4(i)*mat.S_dd(a,b,k,j,i) - (1./3.)*S(i)*z4c.g_dd(a,b,k,j,i));
      }
    }
    // lapse function
    ILOOP1(i) {
      Real const f = opt.lapse_oplog * opt.lapse_harmonicf + opt.lapse_harmonic * z4c.alpha(k,j,i);
      rhs.alpha(k,j,i) = (opt.lapse_advect * Lalpha(i) -
                          f * z4c.alpha(k,j,i) * (z4c.Khat(k,j,i) +
                                                  2 * opt.lapse_K * z4c.Theta(k,j,i)));
    }

    // shift vector
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        rhs.beta_u(a,k,j,i) = opt.shift_Gamma * z4c.Gam_u(a,k,j,i) + opt.shift_advect * Lbeta_u(a,i);

        // This term is taken care of below
        // rhs.beta_u(a,k,j,i) -= opt.shift_eta * z4c.beta_u(a,k,j,i);
      }
    }

    // harmonic gauge terms
    if (std::fabs(opt.shift_alpha2Gamma) > 0.0) {
      // coutBoldBlue("opt.shift_alpha2Gamma\n");
      for(int a = 0; a < NDIM; ++a) {
        ILOOP1(i) {
          rhs.beta_u(a,k,j,i) += opt.shift_alpha2Gamma *
                                 SQR(z4c.alpha(k,j,i)) * z4c.Gam_u(a,k,j,i);
        }
      }
    }

    if (std::fabs(opt.shift_H) > 0.0) {
      // coutBoldRed("opt.shift_H\n");
      for(int a = 0; a < NDIM; ++a) {
        for(int b = 0; b < NDIM; ++b) {
          ILOOP1(i) {
            rhs.beta_u(a,k,j,i) += opt.shift_H * z4c.alpha(k,j,i) *
              chi_guarded(i) * (0.5 * z4c.alpha(k,j,i) * dchi_d(b,i) -
                                dalpha_d(b,i)) * g_uu(a,b,i);
          }
        }
      }
    }

#if defined(Z4C_ETA_CONF)
    // compute based on conformal factor

    // relevant fields:
    // g_uu(c, d, i) [con. conf. g]
    // z4c.chi(k,j,i)
    // dchi_d(a,i)   [cov. pd of chi]

    eta_damp.ZeroClear();

    for(int a = 0; a < NDIM; ++a) {
      for(int b = 0; b < NDIM; ++b) {
        ILOOP1(i) {
          eta_damp(i) += g_uu(a,b,i) * dchi_d(a,i) * dchi_d(b,i);
        }
      }
    }

    ILOOP1(i) {
      eta_damp(i) = opt.shift_eta_R_0 / 2. * SQRT(eta_damp(i) / chi_guarded(i))
        / pow(1. - pow(chi_guarded(i), opt.shift_eta_a / 2.),
              opt.shift_eta_b);
    }

    // mask and damp
    for(int a = 0; a< NDIM; ++a) {
      ILOOP1(i) {
        rhs.beta_u(a,k,j,i) -= eta_damp(i) * z4c.beta_u(a,k,j,i);
      }
    }

#elif defined(Z4C_ETA_TRACK_TP)
    int const b_ix = opt.shift_eta_TP_ix;

    eta_damp.ZeroClear();

    // compute prefactors
    Real const re_pow2_w = 1. / POW2(opt.shift_eta_w);

    ILOOP1(i) {

      eta_damp(i) += \
        POW2(pmy_block->pmy_mesh->pz4c_tracker[b_ix]->GetPos(0)
          - mbi.x1(i));
      eta_damp(i) += \
        POW2(pmy_block->pmy_mesh->pz4c_tracker[b_ix]->GetPos(1)
          - mbi.x2(j));
      eta_damp(i) += \
        POW2(pmy_block->pmy_mesh->pz4c_tracker[b_ix]->GetPos(2)
          - mbi.x3(k));

      eta_damp(i) = 1. + pow(eta_damp(i) * re_pow2_w, opt.shift_eta_delta);

      eta_damp(i) = opt.shift_eta \
        + (opt.shift_eta_P - opt.shift_eta) / eta_damp(i);
    }

    // mask and damp
    for(int a = 0; a< NDIM; ++a) {
      ILOOP1(i) {
        rhs.beta_u(a,k,j,i) -= eta_damp(i) * z4c.beta_u(a,k,j,i);
      }
    }

#else
    // global constant [original implementation]
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        rhs.beta_u(a,k,j,i) -= opt.shift_eta * z4c.beta_u(a,k,j,i);
      }
    }
#endif // Z4C_ETA_CONF, Z4C_ETA_TRACK_TP

  }

  // ===================================================================================
  // Add dissipation for stability
  //
  for(int n = 0; n < N_Z4c; ++n)
  for(int a = 0; a < NDIM; ++a) {
    ILOOP3(k,j,i) {
      u_rhs(n,k,j,i) += fd->Diss(a, u(n,k,j,i), opt.diss);
    }
  }

}
#endif // DBG_EOM

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cBoundaryRHS(AthenaArray<Real> & u, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_rhs)
// \brief compute the boundary RHS given the state vector and matter state
//
// This function operates only on a thin layer of points at the physical
// boundary of the domain.

void Z4c::Z4cBoundaryRHS(AthenaArray<Real> & u, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_rhs) {
  MeshBlock * pmb = pmy_block;
  BoundaryValues * pbval = pmy_block->pbval;

  if(pbval->block_bcs[BoundaryFace::inner_x1] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.il,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x1] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.iu, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.jl,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.ju, mbi.ju,
                   mbi.kl, mbi.ku);
  }
  if(pbval->block_bcs[BoundaryFace::inner_x3] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.kl, mbi.kl);
  }
  if(pbval->block_bcs[BoundaryFace::outer_x3] == BoundaryFlag::gr_sommerfeld)
  {
    Z4cSommerfeld_(u, u_rhs,
                   mbi.il, mbi.iu,
                   mbi.jl, mbi.ju,
                   mbi.ku, mbi.ku);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cSommerfeld_(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs,
//      int const is, int const ie, int const js, int const je, int const ks, int const ke);
// \brief apply Sommerfeld BCs to the given set of points
//

void Z4c::Z4cSommerfeld_(AthenaArray<Real> & u, AthenaArray<Real> & u_rhs,
  int const is, int const ie,
  int const js, int const je,
  int const ks, int const ke) {

  Z4c_vars z4c, rhs;
  SetZ4cAliases(u, z4c);
  SetZ4cAliases(u_rhs, rhs);

  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j) {
    // -----------------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dKhat_d(a,i) = fd->Ds(a, z4c.Khat(k,j,i));
        dTheta_d(a,i) = fd->Ds(a, z4c.Theta(k,j,i));
      }
    }
    // Vectors
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dGam_du(b,a,i) = fd->Ds(b, z4c.Gam_u(a,k,j,i));
      }
    }
    // Tensors
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        dA_ddd(c,a,b,i) = fd->Ds(c, z4c.A_dd(a,b,k,j,i));
      }
    }

    // -----------------------------------------------------------------------------------
    // Compute pseudo-radial vector
    //
#pragma omp simd
    for(int i = is; i <= ie; ++i) {
      r(i) = std::sqrt(SQR(mbi.x1(i)) + SQR(mbi.x2(j)) + SQR(mbi.x3(k)));
      s_u(0,i) = mbi.x1(i)/r(i);
      s_u(1,i) = mbi.x2(j)/r(i);
      s_u(2,i) = mbi.x3(k)/r(i);
    }

    // -----------------------------------------------------------------------------------
    // Boundary RHS for scalars
    //
#pragma omp simd
    for(int i = is; i <= ie; ++i) {
      rhs.Theta(k,j,i) = - z4c.Theta(k,j,i)/r(i);
      rhs.Khat(k,j,i) = - std::sqrt(2.) * z4c.Khat(k,j,i)/r(i);
    }
    for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs.Theta(k,j,i) -= s_u(a,i) * dTheta_d(a,i);
        rhs.Khat(k,j,i) -= std::sqrt(2.) * s_u(a,i) * dKhat_d(a,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // Boundary RHS for the Gamma's
    //
    for(int a = 0; a < NDIM; ++a) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs.Gam_u(a,k,j,i) = - z4c.Gam_u(a,k,j,i)/r(i);
      }
      for(int b = 0; b < NDIM; ++b) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          rhs.Gam_u(a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
        }
      }
    }

    // -----------------------------------------------------------------------------------
    // Boundary RHS for the A_ab
    //
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
      for(int i = is; i <= ie; ++i) {
        rhs.A_dd(a,b,k,j,i) = - z4c.A_dd(a,b,k,j,i)/r(i);
      }
      for(int c = 0; c < NDIM; ++c) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          rhs.A_dd(a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
        }
      }
    }
  }
}


// Debug EOM ------------------------------------------------------------------
// H&R 2018
#ifdef DBG_EOM
// For readability
typedef AthenaArray< Real>                            AA;
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 0> AT_N_sca;
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> AT_N_vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> AT_N_sym;

// derivatives
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 1> AT_N_D1sca;
typedef AthenaTensor<Real, TensorSymm::NONE, NDIM, 2> AT_N_D1vec;
typedef AthenaTensor<Real, TensorSymm::SYM2, NDIM, 3> AT_N_D1sym;

typedef AthenaTensor<Real, TensorSymm::SYM2,  NDIM, 2> AT_N_d2sca;
typedef AthenaTensor<Real, TensorSymm::ISYM2, NDIM, 3> AT_N_d2vec;
typedef AthenaTensor<Real, TensorSymm::SYM22, NDIM, 4> AT_N_d2sym;

void Z4c::Z4cRHS(AthenaArray<Real> & u,
                 AthenaArray<Real> & u_mat,
                 AthenaArray<Real> & u_rhs)
{
  using namespace LinearAlgebra;

  // Slice rhs 3d quantities --------------------------------------------------
  AT_N_sca rhs_alpha(u_rhs, Z4c::I_Z4c_alpha);
  AT_N_sca rhs_chi(  u_rhs, Z4c::I_Z4c_chi);
  AT_N_sca rhs_Theta(u_rhs, Z4c::I_Z4c_Theta);
  AT_N_sca rhs_Khat( u_rhs, Z4c::I_Z4c_Khat);

  AT_N_vec rhs_beta_u(     u_rhs, Z4c::I_Z4c_betax);
  AT_N_vec rhs_Gammatil_u( u_rhs, Z4c::I_Z4c_Gamx);

  AT_N_sym rhs_Atil_dd(    u_rhs, Z4c::I_Z4c_Axx);
  AT_N_sym rhs_gammatil_dd(u_rhs, Z4c::I_Z4c_gxx);

  // Slice 3d z4c metric quantities  (NDIM=3 in z4c.hpp) ----------------------
  AT_N_sca z4c_alpha(u, Z4c::I_Z4c_alpha);
  AT_N_sca z4c_chi(  u, Z4c::I_Z4c_chi);
  AT_N_sca z4c_Theta(u, Z4c::I_Z4c_Theta);
  AT_N_sca z4c_Khat( u, Z4c::I_Z4c_Khat);

  AT_N_vec z4c_beta_u(     u, Z4c::I_Z4c_betax);
  AT_N_vec z4c_Gammatil_u( u, Z4c::I_Z4c_Gamx);

  AT_N_sym z4c_Atil_dd(    u, Z4c::I_Z4c_Axx);
  AT_N_sym z4c_gammatil_dd(u, Z4c::I_Z4c_gxx);

  AT_N_sca adm_rho( u_mat, Z4c::I_MAT_rho);

  AT_N_vec adm_S_d( u_mat, Z4c::I_MAT_Sx);

  AT_N_sym adm_S_dd(u_mat, Z4c::I_MAT_Sxx);

  // Scratch & parameters -----------------------------------------------------
  AT_N_sca z4c_detgamma_(   mbi.nn1);
  AT_N_sca z4c_oodetgamma_( mbi.nn1);
  AT_N_sca z4c_oochi_(      mbi.nn1);

  AT_N_sym z4c_gammatil_uu_(mbi.nn1);
  AT_N_sym z4c_Atil_uu_(    mbi.nn1);
  AT_N_sym z4c_AAtil_dd_(   mbi.nn1);

  AT_N_sca z4c_AAtil_(      mbi.nn1);

  // for trace-free piece
  AT_N_sca z4c_AtilTr_(   mbi.nn1);

  AT_N_sym z4c_AtilTF_dd_(mbi.nn1);

  AT_N_vec z4c_Gammatild_u_(mbi.nn1);

  AT_N_sym z4c_Rictilchi_dd_(mbi.nn1);
  AT_N_sym z4c_Rictil_dd_(   mbi.nn1);

  AT_N_sca z4c_R_(           mbi.nn1);
  AT_N_sca adm_S_(       mbi.nn1);

  AT_N_sym adm_gamma_dd_(mbi.nn1);
  AT_N_sym adm_gamma_uu_(mbi.nn1);

  // derivatives
  AT_N_D1sca z4c_dalpha_d_(mbi.nn1);
  AT_N_D1sca z4c_dchi_d_(  mbi.nn1);
  AT_N_D1sca z4c_dKhat_d_( mbi.nn1);
  AT_N_D1sca z4c_dTheta_d_(mbi.nn1);

  // contracted
  AT_N_sca z4c_dbeta_(mbi.nn1);

  AT_N_D1vec z4c_dbeta_du_(    mbi.nn1);
  AT_N_D1vec z4c_dGammatil_du_(mbi.nn1);

  AT_N_D1sym z4c_dgammatil_ddd_(mbi.nn1);
  AT_N_D1sym z4c_dAtil_ddd_(    mbi.nn1);

  AT_N_d2sca z4c_ddalpha_dd_( mbi.nn1);
  AT_N_d2sca z4c_ddchi_dd_(   mbi.nn1);

  AT_N_d2vec z4c_ddbeta_ddu_(mbi.nn1);

  AT_N_d2sym z4c_ddgammatil_dddd_(mbi.nn1);

  // derivatives (advective)
  AT_N_sca z4c_Lalpha_(mbi.nn1);
  AT_N_sca z4c_Lchi_(  mbi.nn1);
  AT_N_sca z4c_LKhat_( mbi.nn1);
  AT_N_sca z4c_LTheta_(mbi.nn1);

  AT_N_vec z4c_Lbeta_u_(    mbi.nn1);
  AT_N_vec z4c_LGammatil_u_(mbi.nn1);

  AT_N_sym z4c_Lgammatil_dd_(mbi.nn1);
  AT_N_sym z4c_LAtil_dd_(    mbi.nn1);

  // Christoffels
  AT_N_D1sym z4c_Gammatil_ddd_(mbi.nn1);
  AT_N_D1sym z4c_Gammatil_udd_(mbi.nn1);

  // Difference between connections
  AT_N_D1sym z4c_GammaDDtil_udd_(mbi.nn1);

  // 1st covariants contracted
  AT_N_sca z4c_Dbeta_(mbi.nn1);

  // 2nd covariant of scalars
  AT_N_sca z4c_DDalpha_(       mbi.nn1);
  AT_N_sym z4c_DDalpha_dd_(    mbi.nn1);
  AT_N_sym z4c_DtilDtilchi_dd_(mbi.nn1);

  // debug
  AT_N_D1sca dphi_d(    mbi.nn1);


  const int il = mbi.il;
  const int iu = mbi.iu;

  const Real kappa_1 = opt.damp_kappa1;
  const Real kappa_2 = opt.damp_kappa2;

  // coupling term between connections
  const Real p_connection = -12.0 / (3.0 * opt.chi_psi_power);

  // enforce standard z4c?
#ifdef DBG_EOM_Z4C_ENFORCE
  const Real enf_z4c = 1.;
#else
  const Real enf_z4c = 0.;
#endif // DBG_EOM_Z4C_ENFORCE

  // gauge
  const Real alpha_l   = opt.lapse_oplog;
  const Real alpha_hf  = opt.lapse_harmonicf;
  const Real alpha_h   = opt.lapse_harmonic;
  const Real alpha_adv = opt.lapse_advect;

  const Real sigma_Gamma       = opt.shift_Gamma;
  const Real sigma_eta         = opt.shift_eta;
  const Real sigma_alpha2Gamma = opt.shift_alpha2Gamma;
  const Real sigma_H           = opt.shift_H;
  const Real sigma_adv         = opt.shift_advect;

  ILOOP2(k,j)
  {
    // 1st derivatives --------------------------------------------------------
    {
      // Scalars
      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        z4c_dalpha_d_(a,i) = fd->Dx(a, z4c_alpha(k,j,i));
        z4c_dchi_d_(  a,i) = fd->Dx(a, z4c_chi(  k,j,i));
        z4c_dKhat_d_( a,i) = fd->Dx(a, z4c_Khat( k,j,i));
        z4c_dTheta_d_(a,i) = fd->Dx(a, z4c_Theta(k,j,i));
      }

      // Vectors
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_dbeta_du_(    b,a,i) = fd->Dx(b, z4c_beta_u(    a,k,j,i));
        z4c_dGammatil_du_(b,a,i) = fd->Dx(b, z4c_Gammatil_u(a,k,j,i));
      }

      // Tensors
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        z4c_dgammatil_ddd_(c,a,b,i) = fd->Dx(c, z4c_gammatil_dd(a,b,k,j,i));
        z4c_dAtil_ddd_(    c,a,b,i) = fd->Dx(c, z4c_Atil_dd(    a,b,k,j,i));
      }
    }


    // 2nd derivatives --------------------------------------------------------
    {
      // Scalars
      for (int a=0; a<NDIM; ++a)
      {
        ILOOP1(i)
        {
          z4c_ddalpha_dd_(a,a,i) = fd->Dxx(a, z4c_alpha(k,j,i));
          z4c_ddchi_dd_(  a,a,i) = fd->Dxx(a, z4c_chi(  k,j,i));
        }
        for(int b = a + 1; b < NDIM; ++b)
        ILOOP1(i)
        {
            z4c_ddalpha_dd_(a,b,i) = fd->Dxy(a, b, z4c_alpha(k,j,i));
            z4c_ddchi_dd_(  a,b,i) = fd->Dxy(a, b, z4c_chi(  k,j,i));
        }
      }

      // Vectors
      for (int c=0; c<NDIM; ++c)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        if(a==b)
        {
          ILOOP1(i)
          {
            z4c_ddbeta_ddu_(a,b,c,i) = fd->Dxx(a, z4c_beta_u(c,k,j,i));
          }
        }
        else
        {
          ILOOP1(i)
          {
            z4c_ddbeta_ddu_(a,b,c,i) = fd->Dxy(a, b, z4c_beta_u(c,k,j,i));
          }
        }
      }

      // Tensors
      for (int c=0; c<NDIM; ++c)
      for (int d=c; d<NDIM; ++d)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        if(a == b)
        {
          ILOOP1(i)
          {
            z4c_ddgammatil_dddd_(a,b,c,d,i) = fd->Dxx(
              a, z4c_gammatil_dd(c,d,k,j,i)
            );
          }
        }
        else
        {
          ILOOP1(i)
          {
            z4c_ddgammatil_dddd_(a,b,c,d,i) = fd->Dxy(
              a, b, z4c_gammatil_dd(c,d,k,j,i)
            );
          }
        }
      }
    }


    // Advective derivatives --------------------------------------------------
    {
      // Scalars
      z4c_Lalpha_.ZeroClear();
      z4c_Lchi_.ZeroClear();
      z4c_LKhat_.ZeroClear();
      z4c_LTheta_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        z4c_Lalpha_(i) += fd->Lx(a, z4c_beta_u(a,k,j,i), z4c_alpha(k,j,i));
        z4c_Lchi_(i)   += fd->Lx(a, z4c_beta_u(a,k,j,i), z4c_chi(k,j,i)  );
        z4c_LKhat_(i)  += fd->Lx(a, z4c_beta_u(a,k,j,i), z4c_Khat(k,j,i) );
        z4c_LTheta_(i) += fd->Lx(a, z4c_beta_u(a,k,j,i), z4c_Theta(k,j,i));
      }

      // Vectors
      z4c_Lbeta_u_.ZeroClear();
      z4c_LGammatil_u_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_Lbeta_u_(b,i)     += fd->Lx(a, z4c_beta_u(a,k,j,i),
                                        z4c_beta_u(b,k,j,i));
        z4c_LGammatil_u_(b,i) += fd->Lx(a, z4c_beta_u(a,k,j,i),
                                        z4c_Gammatil_u(b,k,j,i));
      }

      // Tensors
      z4c_Lgammatil_dd_.ZeroClear();
      z4c_LAtil_dd_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        z4c_Lgammatil_dd_(a,b,i) += fd->Lx(c, z4c_beta_u(c,k,j,i),
                                          z4c_gammatil_dd(a,b,k,j,i));
        z4c_LAtil_dd_(a,b,i) += fd->Lx(c, z4c_beta_u(c,k,j,i),
                                      z4c_Atil_dd(a,b,k,j,i));
      }
    }

    // Auxiliary quantities (I) -----------------------------------------------
    // z4c_detgamma_, z4c_oodetgamma_, z4c_gammatil_uu_, z4c_oochi_, z4c_dbeta_
    {

      // prepare z4c_detgamma_, z4c_gammatil_uu_
      Det3Metric(z4c_detgamma_, z4c_gammatil_dd, k, j, il, iu);
      ILOOP1(i)
      {
        z4c_oodetgamma_(i) = 1.0 / z4c_detgamma_(i);
      }
      Inv3Metric(z4c_oodetgamma_, z4c_gammatil_dd, z4c_gammatil_uu_,
                 k, j, il, iu);

      // prepare 1 / chi (note recip. flooring procedure)
      ILOOP1(i)
      {
        z4c_oochi_(i) = 1.0 / std::max(z4c_chi(k,j,i), opt.chi_div_floor);
      }

      z4c_dbeta_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        z4c_dbeta_(i) += z4c_dbeta_du_(a,a,i);
      }

    }

    // Auxiliary quantities (II) ----------------------------------------------
    // adm_gamma_dd_, adm_gamma_uu_, adm_S_;
    {
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        adm_gamma_dd_(a,b,i) = z4c_oochi_(i) * z4c_gammatil_dd(a,b,k,j,i);
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        adm_gamma_uu_(a,b,i) = z4c_chi(k,j,i) * z4c_gammatil_uu_(a,b,i);
      }

      adm_S_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        adm_S_(i) += adm_gamma_uu_(a,b,i) * adm_S_dd(a,b,k,j,i);
      }
    }

    // Christoffels / Gammatil_d_u_ -------------------------------------------
    // z4c_Gammatil_ddd_, z4c_Gammatil_udd_, z4c_Gammatild_u_
    {
      for (int c=0; c<NDIM; ++c)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_Gammatil_ddd_(c,a,b,i) = 0.5*(z4c_dgammatil_ddd_(a,b,c,i) +
                                          z4c_dgammatil_ddd_(b,a,c,i) -
                                          z4c_dgammatil_ddd_(c,a,b,i));
      }

      z4c_Gammatil_udd_.ZeroClear();
      for (int c=0; c<NDIM; ++c)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        z4c_Gammatil_udd_(c,a,b,i) += z4c_gammatil_uu_( c,d,i) *
                                      z4c_Gammatil_ddd_(d,a,b,i);
      }

      // Gamma's computed from the conformal metric (not evolved)
      z4c_Gammatild_u_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        z4c_Gammatild_u_(a,i) += z4c_gammatil_uu_( b,c,i) *
                                 z4c_Gammatil_udd_(a,b,c,i);
      }

    }

    // Connection difference --------------------------------------------------
    // z4c_GammaDDtil_udd_
    {
      for (int c=0; c<NDIM; ++c)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_GammaDDtil_udd_(c,a,b,i) = -0.5 * p_connection * z4c_oochi_(i) * (
          (c==a) * z4c_dchi_d_(b,i) + (c==b) * z4c_dchi_d_(a,i)
        );
      }

      for (int c=0; c<NDIM; ++c)
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        z4c_GammaDDtil_udd_(c,a,b,i) += 0.5 * p_connection * z4c_oochi_(i) * (
          z4c_gammatil_dd(a,b,k,j,i) * z4c_gammatil_uu_(c,d,i) *
                                       z4c_dchi_d_(d,i)
        );
      }
    }

    // Covariant derivatives (I) ----------------------------------------------
    // z4c_Dbeta_, z4c_DtilDtilchi_dd_
    {
      const Real fac = 6.0 / opt.chi_psi_power;
      ILOOP1(i)
      {
        z4c_Dbeta_(i) = fac * z4c_oochi_(i) * z4c_Lchi_(i);
      }

      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        z4c_Dbeta_(i) += z4c_dbeta_du_(a,a,i);
      }


      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        ILOOP1(i)
        {
          z4c_DtilDtilchi_dd_(a,b,i) = z4c_ddchi_dd_(a,b,i);
        }
        for (int c=0; c<NDIM; ++c)
        ILOOP1(i)
        {
          z4c_DtilDtilchi_dd_(a,b,i) -= z4c_Gammatil_udd_(c,a,b,i) *
                                        z4c_dchi_d_(c,i);
        }
      }

    }

    // Covariant derivatives (II) ---------------------------------------------
    // z4c_DDalpha_dd_, z4c_DDalpha_
    {
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        ILOOP1(i)
        {
          z4c_DDalpha_dd_(a,b,i) = z4c_ddalpha_dd_(a,b,i);
        }
        for (int c=0; c<NDIM; ++c)
        ILOOP1(i)
        {
          z4c_DDalpha_dd_(a,b,i) += -z4c_dalpha_d_(c,i) * (
            z4c_GammaDDtil_udd_(c,a,b,i) + z4c_Gammatil_udd_(c,a,b,i)
          );
        }
      }

      TraceRank2(z4c_DDalpha_, adm_gamma_uu_, z4c_DDalpha_dd_, il, iu);


    /*
    for(int a = 0; a < NDIM; ++a) {
      ILOOP1(i) {
        dphi_d(a,i) = z4c_oochi_(i) * z4c_dchi_d_(a,i)/(opt.chi_psi_power);
      }
    }

    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        z4c_DDalpha_dd_(a,b,i) = z4c_ddalpha_dd_(a,b,i)
                          - 2.*(dphi_d(a,i)*z4c_dalpha_d_(b,i) + dphi_d(b,i)*z4c_dalpha_d_(a,i));
      }
      for(int c = 0; c < NDIM; ++c) {
        ILOOP1(i) {
          z4c_DDalpha_dd_(a,b,i) -= z4c_Gammatil_udd_(c,a,b,i)*z4c_dalpha_d_(c,i);
        }
        for(int d = 0; d < NDIM; ++d) {
          ILOOP1(i) {
            z4c_DDalpha_dd_(a,b,i) += 2.*z4c_gammatil_dd(a,b,k,j,i) * z4c_gammatil_uu_(c,d,i) * dphi_d(c,i) * z4c_dalpha_d_(d,i);
          }
        }
      }
    }
    */

    }

    // Auxiliary quantities (III) ---------------------------------------------
    // z4c_Atil_uu_, z4c_AAtil_dd_, z4c_AAtil_
    {
      z4c_Atil_uu_.ZeroClear();
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d)
      ILOOP1(i)
      {
        z4c_Atil_uu_(a,b,i) += z4c_gammatil_uu_(a,c,i) *
                               z4c_gammatil_uu_(b,d,i) *
                               z4c_Atil_dd(c,d,k,j,i);
      }

      z4c_AAtil_dd_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        z4c_AAtil_dd_(a,b,i) += z4c_gammatil_uu_(c,d,i) *
                                z4c_Atil_dd(a,c,k,j,i) *
                                z4c_Atil_dd(d,b,k,j,i);
      }

      z4c_AAtil_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_AAtil_(i) += z4c_gammatil_uu_(a,b,i) *
                         z4c_AAtil_dd_(a,b,i);
      }

    }

    // Auxiliary quantities (IV) ----------------------------------------------
    // z4c_Rictilchi_dd_, z4c_Rictil_dd_, z4c_R_
    {
      // z4c_Rictilchi_dd_
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_Rictilchi_dd_(a,b,i) = 0.5 * z4c_oochi_(i) * (
          z4c_DtilDtilchi_dd_(a,b,i) -
          0.5 * z4c_oochi_(i) * z4c_dchi_d_(a,i) * z4c_dchi_d_(b,i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        z4c_Rictilchi_dd_(a,b,i) += 0.5 * z4c_oochi_(i) * (
          z4c_DtilDtilchi_dd_(c,d,i) -
          1.5 * z4c_oochi_(i) * z4c_dchi_d_(c,i) * z4c_dchi_d_(d,i)
        ) * z4c_gammatil_dd(a,b,k,j,i) * z4c_gammatil_uu_(c,d,i);
      }

      // z4c_Rictil_dd_
      z4c_Rictil_dd_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        z4c_Rictil_dd_(a,b,i) += 0.5 * (
          z4c_gammatil_dd(c,a,k,j,i) * z4c_dGammatil_du_(b,c,i) +
          z4c_gammatil_dd(c,b,k,j,i) * z4c_dGammatil_du_(a,c,i)
        );

        z4c_Rictil_dd_(a,b,i) += 0.5 * z4c_Gammatild_u_(c,i) * (
          z4c_Gammatil_ddd_(a,b,c,i) + z4c_Gammatil_ddd_(b,a,c,i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      ILOOP1(i)
      {
        z4c_Rictil_dd_(a,b,i) -= 0.5 * (
          z4c_gammatil_uu_(c,d,i) * z4c_ddgammatil_dddd_(c,d,a,b,i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      for (int d=0; d<NDIM; ++d)
      for (int e=0; e<NDIM; ++e)
      ILOOP1(i)
      {
        z4c_Rictil_dd_(a,b,i) += z4c_gammatil_uu_(c,d,i) * (
          z4c_Gammatil_udd_(e,c,a,i) * z4c_Gammatil_ddd_(b,e,d,i) +
          z4c_Gammatil_udd_(e,c,b,i) * z4c_Gammatil_ddd_(a,e,d,i) +
          z4c_Gammatil_udd_(e,a,d,i) * z4c_Gammatil_ddd_(e,c,b,i)
        );
      }

      // z4c_R_
      z4c_R_.ZeroClear();
      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_R_(i) += adm_gamma_uu_(a,b,i) * (
          z4c_Rictilchi_dd_(a,b,i) + z4c_Rictil_dd_(a,b,i)
        );
      }


    }

    // Auxiliary quantities (V) -----------------------------------------------
    // z4c_AtilTr_, z4c_AtilTF_dd_
    {
      ILOOP1(i)
      {
        z4c_AtilTr_(i) = (
          -z4c_DDalpha_(i) + z4c_alpha(k,j,i) * (z4c_R_(i) - 8.0 * M_PI *
                                                 adm_S_(i))
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        z4c_AtilTF_dd_(a,b,i) = (
          -z4c_DDalpha_dd_(a,b,i) + z4c_alpha(k,j,i) * (
            (z4c_Rictilchi_dd_(a,b,i) + z4c_Rictil_dd_(a,b,i)) -
            8.0 * M_PI * adm_S_dd(a,b,k,j,i)) -
          ONE_3RD * adm_gamma_dd_(a,b,i) * z4c_AtilTr_(i)
        );
      }

    }

    // Assemble RHS (EOM I) ---------------------------------------------------
    // rhs_chi, rhs_gammatil_dd
    {
      ILOOP1(i)
      {
        rhs_chi(k,j,i) = TWO_3RD * z4c_chi(k,j,i) * (
          z4c_alpha(k,j,i) * (z4c_Khat(k,j,i) + 2.0 * z4c_Theta(k,j,i)) -
          z4c_Dbeta_(i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      {
        ILOOP1(i)
        {
          rhs_gammatil_dd(a,b,k,j,i) = z4c_Lgammatil_dd_(a,b,i) + (
            -2.0 * z4c_alpha(k,j,i) * z4c_Atil_dd(a,b,k,j,i)
          );
        }
        for (int c=0; c<NDIM; ++c)
        ILOOP1(i)
        {
          rhs_gammatil_dd(a,b,k,j,i) += (
            (z4c_gammatil_dd(c,a,k,j,i) * z4c_dbeta_du_(b,c,i) +
             z4c_gammatil_dd(c,b,k,j,i) * z4c_dbeta_du_(a,c,i)) -
            TWO_3RD * z4c_gammatil_dd(a,b,k,j,i) * z4c_dbeta_du_(c,c,i)
          );
        }

      }

    }

    // Assemble RHS (EOM II) --------------------------------------------------
    // rhs_Gammatil_u
    {
      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        rhs_Gammatil_u(a,k,j,i) = (
          z4c_LGammatil_u_(a,i) -
          2.0 * z4c_alpha(k,j,i) * kappa_1 * (
            z4c_Gammatil_u(a,k,j,i) - z4c_Gammatild_u_(a,i)
          )
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        rhs_Gammatil_u(a,k,j,i) += (
          -2.0 * z4c_Atil_uu_(a,b,i) * z4c_dalpha_d_(b,i)
          -z4c_Gammatild_u_(b,i) * z4c_dbeta_du_(b,a,i) +
          TWO_3RD * z4c_Gammatild_u_(a,i) * z4c_dbeta_du_(b,b,i) +
          2.0 * z4c_alpha(k,j,i) * (
            -1.5 * z4c_Atil_uu_(a,b,i) * z4c_oochi_(i) * z4c_dchi_d_(b,i)
            -ONE_3RD * z4c_gammatil_uu_(a,b,i) * (
              2.0 * z4c_dKhat_d_(b,i) + enf_z4c * z4c_dTheta_d_(b,i))
            -8.0 * M_PI * z4c_gammatil_uu_(a,b,i) * adm_S_d(b,k,j,i)
          )
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        rhs_Gammatil_u(a,k,j,i) += (
          2.0 * z4c_alpha(k,j,i) * z4c_Gammatil_udd_(a,b,c,i) *
                                   z4c_Atil_uu_(b,c,i) +
          z4c_gammatil_uu_(b,c,i) * z4c_ddbeta_ddu_(b,c,a,i) +
          ONE_3RD * z4c_gammatil_uu_(a,b,i) *
                    z4c_ddbeta_ddu_(b,c,c,i)
        );
      }


    }

    // Assemble RHS (EOM III) -------------------------------------------------
    // rhs_Theta
    {
      ILOOP1(i)
      {
        rhs_Theta(k,j,i) = z4c_LTheta_(i) +  0.5 * z4c_alpha(k,j,i) * (
          z4c_R_(i) - z4c_AAtil_(i) +
          TWO_3RD * SQR(z4c_Khat(k,j,i) + 2.0 * z4c_Theta(k,j,i))
        ) - z4c_alpha(k,j,i) * (
          8.0 * M_PI * adm_rho(k,j,i) +
          kappa_1 * (2.0 + kappa_2) * z4c_Theta(k,j,i)
        );
      }
    }

    // Assemble RHS (EOM IV) --------------------------------------------------
    // rhs_Khat
    {
      ILOOP1(i)
      {
        rhs_Khat(k,j,i) = z4c_LKhat_(i) + z4c_alpha(k,j,i) * (
          z4c_AAtil_(i) + ONE_3RD * SQR(z4c_Khat(k,j,i) + 2.0 *
                                        z4c_Theta(k,j,i))
        ) + (
          4.0 * M_PI * z4c_alpha(k,j,i) * (adm_rho(k,j,i) + adm_S_(i)) +
          z4c_alpha(k,j,i) * kappa_1 * (1.0 - kappa_2) * z4c_Theta(k,j,i)
        ) - z4c_DDalpha_(i);
      }
    }

    // Assemble RHS (EOM V) ---------------------------------------------------
    // rhs_Atil_dd
    {
      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      ILOOP1(i)
      {
        rhs_Atil_dd(a,b,k,j,i) = z4c_LAtil_dd_(a,b,i) + (
          z4c_chi(k,j,i) * z4c_AtilTF_dd_(a,b,i)
          + z4c_alpha(k,j,i) * (
            (z4c_Khat(k,j,i) + 2.0 * z4c_Theta(k,j,i)) *
            z4c_Atil_dd(a,b,k,j,i) - 2.0 * z4c_AAtil_dd_(a,b,i)
          )
        ) - TWO_3RD * z4c_Atil_dd(a,b,k,j,i) * z4c_dbeta_(i);
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=a; b<NDIM; ++b)
      for (int c=0; c<NDIM; ++c)
      ILOOP1(i)
      {
        rhs_Atil_dd(a,b,k,j,i) += (
          z4c_Atil_dd(c,a,k,j,i) * z4c_dbeta_du_(b,c,i) +
          z4c_Atil_dd(c,b,k,j,i) * z4c_dbeta_du_(a,c,i)
        );
      }


    }

    // Assemble RHS [gauge subsystem] -----------------------------------------
    // rhs_alpha, rhs_beta_u
    {
      ILOOP1(i)
      {
        const Real f = alpha_l * alpha_hf + alpha_h * z4c_alpha(k,j,i);
        rhs_alpha(k,j,i) = (
          alpha_adv * z4c_Lalpha_(i) -
          f * z4c_alpha(k,j,i) * z4c_Khat(k,j,i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      ILOOP1(i)
      {
        rhs_beta_u(a,k,j,i) = sigma_adv * z4c_Lbeta_u_(a,i) + (
          (sigma_Gamma + sigma_alpha2Gamma * SQR(z4c_alpha(k,j,i))) *
          z4c_Gammatil_u(a,k,j,i) -
          sigma_eta * z4c_beta_u(a,k,j,i)
        );
      }

      for (int a=0; a<NDIM; ++a)
      for (int b=0; b<NDIM; ++b)
      ILOOP1(i)
      {
        rhs_beta_u(a,k,j,i) += sigma_H * z4c_alpha(k,j,i) * z4c_chi(k,j,i) * (
          0.5 * z4c_alpha(k,j,i) * z4c_dchi_d_(b,i) -
          z4c_dalpha_d_(b,i)
        ) * z4c_gammatil_uu_(a,b,i);
      }

    }

  }

  // ==========================================================================
  // Add dissipation for stability
  //
  for (int n=0; n<N_Z4c; ++n)
  for (int a=0; a<NDIM; ++a)
  ILOOP3(k,j,i)
  {
    u_rhs(n,k,j,i) += fd->Diss(a, u(n,k,j,i), opt.diss);
  }

}

#endif // DBG_EOM