//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation of Weyl scalars

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

#if defined(DBG_WEYL_SWSH_SEED_CART)
#include "wave_extract.hpp"
#endif // DBG_WEYL_SWSH_SEED_CART


template<typename T>
static int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

// BD: TODO - refactor

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_weyl)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock

void Z4c::Z4cWeyl(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat, AthenaArray<Real> & u_weyl) {
  using namespace LinearAlgebra;

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  Weyl_vars weyl;
  SetWeylAliases(u_weyl, weyl);
  weyl.rpsi4.Fill(NAN);
  weyl.ipsi4.Fill(NAN);

  // Simplify constants (2 & sqrt 2 factors) featured in re/im[psi4]
  const Real FR4 = 0.25;

#if defined(DBG_WEYL_SWSH_SEED_CART)

  ILOOP3(k,j,i)
  {

    const Real x = mbi.x1(i);
    const Real y = mbi.x2(j);
    const Real z = mbi.x3(k);

    const Real r_xy = std::sqrt(SQR(x) + SQR(y));
    const Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));

    Real theta = std::acos(z / r);
    Real phi = sgn(y) * std::acos(x / (std::sqrt(SQR(x) + SQR(y))));

    // deal with axis
    if (r == 0)
    {
      theta = 0;
    }

    if (r_xy == 0)
    {
      phi = 0;
    }

    Real ylmR;
    Real ylmI;

    WaveExtractLocal::swsh(&ylmR, &ylmI, 2, 2, theta, phi);

    weyl.rpsi4(k,j,i) = ylmR;
    weyl.ipsi4(k,j,i) = ylmI;

    // weyl.rpsi4(k,j,i) = std::cos(phi);
    // weyl.ipsi4(k,j,i) = 0;


    WaveExtractLocal::swsh(&ylmR, &ylmI, 4, 0, theta, phi);

    weyl.rpsi4(k,j,i) -= ylmR;
    weyl.ipsi4(k,j,i) -= ylmI;

  }

  return;
#endif // DBG_WEYL_SWSH_SEED_CART

  ILOOP2(k,j) {


    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        dg_ddd(c,a,b,i) = fd->Dx(c, adm.g_dd(a,b,k,j,i));
        dK_ddd(c,a,b,i) = fd->Dx(c, adm.K_dd(a,b,k,j,i));
      }
    }
    // second derivatives of g
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
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

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = 0; b < NDIM; ++b) {
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
    for(int c = 0; c < NDIM; ++c) {
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
    for(int c = 0; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      ILOOP1(i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      }
    }





    // -----------------------------------------------------------------------------------
    // Trace of the matter stress tensor
    //
    S.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      ILOOP1(i) {
        S(i) += g_uu(a,b,i) * mat.S_dd(a,b,k,j,i);
      }
    }

    //------------------------------------------------------------------------------------
    //     Construct tetrad
    //
    //     Initial tetrad guess. NB, aligned with z axis - possible problem if points lie on z axis
    //     theta and phi vectors degenerate
    //     Like BAM start with phi vector
    //     uvec = radial vec
    //     vvec = theta vec
    //     wvec = phi vec
    ILOOP1(i){
      Real xx = mbi.x1(i);
      if(SQR(mbi.x1(i)) +  SQR(mbi.x2(j)) < 1e-10)
        xx = xx + 1e-8;
      uvec(0,i) = xx;
      uvec(1,i) = mbi.x2(j);
      uvec(2,i) = mbi.x3(k);
      vvec(0,i) = xx*mbi.x3(k);
      vvec(1,i) = mbi.x2(j)*mbi.x3(k);
      vvec(2,i) = -SQR(xx)-SQR(mbi.x2(j));
      wvec(0,i) = mbi.x2(j)*-1.0;
      wvec(1,i) = xx;
      wvec(2,i) = 0.0;
    }

    //Gram-Schmidt orthonormalisation with spacetime metric.
    //
    dotp1.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
	    for(int b = 0; b<NDIM; ++b){
	      ILOOP1(i){
          dotp1(i) += adm.g_dd(a,b,k,j,i)*wvec(a,i)*wvec(b,i);
	      }
	    }
    }
    for(int a =0; a<NDIM; ++a){
      ILOOP1(i){
	      wvec(a,i) = wvec(a,i)/sqrt(dotp1(i));
      }
    }

    dotp1.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
      for( int b = 0; b<NDIM; ++b){
	      ILOOP1(i){
	        dotp1(i) += adm.g_dd(a,b,k,j,i)*wvec(a,i)*uvec(b,i);
	      }
      }
    }
    for(int a = 0; a<NDIM; ++a){
      ILOOP1(i){
	      uvec(a,i) -= dotp1(i)*wvec(a,i);
	    }
    }
    dotp1.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
	    for(int b = 0; b<NDIM; ++b) {
	      ILOOP1(i){
	        dotp1(i) += adm.g_dd(a,b,k,j,i)*uvec(a,i)*uvec(b,i);
	      }
	    }
    }

    for(int a =0; a<NDIM; ++a){
	    ILOOP1(i){
	      uvec(a,i) = uvec(a,i)/sqrt(dotp1(i));
	    }
    }

    dotp1.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
      for(int b = 0; b<NDIM; ++b) {
	      ILOOP1(i){
	        dotp1(i) += adm.g_dd(a,b,k,j,i)*wvec(a,i)*vvec(b,i);
	      }
	    }
    }
    dotp2.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
	    for( int b = 0; b<NDIM; ++b) {
	      ILOOP1(i){
	        dotp2(i) += adm.g_dd(a,b,k,j,i)*uvec(a,i)*vvec(b,i);
	      }
	    }
    }

    for(int a = 0; a<NDIM; ++a){
	    ILOOP1(i){
	      vvec(a,i) -= dotp1(i)*wvec(a,i)+dotp2(i)*uvec(a,i);
	    }
    }

    dotp1.ZeroClear();
    for(int a = 0; a<NDIM; ++a){
	    for( int b = 0; b<NDIM; ++b) {
	      ILOOP1(i){
	        dotp1(i) += adm.g_dd(a,b,k,j,i)*vvec(a,i)*vvec(b,i);
	      }
	    }
    }

    for(int a =0; a<NDIM; ++a){
	    ILOOP1(i){
	      vvec(a,i) = vvec(a,i)/sqrt(dotp1(i));
	    }
    }

    //   Riem3_dddd = Riemann tensor of spacelike hypersurface
    //   Riemm4_dddd = Riemann tensor of 4D spacetime
    //   Riemm4_ddd  = Riemann tensor of 4D spacetime contracted once with n
    //   Riemm4_dd  = Riemann tensor of 4D spacetime contracted twice with n
    Riem3_dddd.ZeroClear();
    Riemm4_dddd.ZeroClear();
    Riemm4_ddd.ZeroClear();
    Riemm4_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a){
      for(int b = 0; b < NDIM; ++b){
        for(int c = 0; c < NDIM; ++c){
          for(int d = 0; d < NDIM; ++d){
            ILOOP1(i){
              Riem3_dddd(a,b,c,d,i) = adm.g_dd(a,c,k,j,i)*R_dd(b,d,i) +adm.g_dd(b,d,k,j,i)*R_dd(a,c,i)
                                      - adm.g_dd(a,d,k,j,i)*R_dd(b,c,i) - adm.g_dd(b,c,k,j,i)*R_dd(a,d,i)
                                      - 0.5*R(i)*adm.g_dd(a,c,k,j,i)*adm.g_dd(b,d,k,j,i)
                                      + 0.5*R(i)*adm.g_dd(a,d,k,j,i)*adm.g_dd(b,c,k,j,i);
              Riemm4_dddd(a,b,c,d,i) = Riem3_dddd(a,b,c,d,i) + adm.K_dd(a,c,k,j,i)*adm.K_dd(b,d,k,j,i)
                                      - adm.K_dd(a,d,k,j,i)*adm.K_dd(b,c,k,j,i);
            }
          }
        }
      }
    }

    for(int a = 0; a < NDIM; ++a){
      for(int b = 0; b < NDIM; ++b){
        for(int c = 0; c < NDIM; ++c){
          ILOOP1(i){
            Riemm4_ddd(a,b,c,i) = - (DK_ddd(c,a,b,i) - DK_ddd(b,a,c,i));
          }
        }
      }
    }


    for(int a = 0; a < NDIM; ++a){
      for(int b = 0; b < NDIM; ++b){
        ILOOP1(i){
          Riemm4_dd(a,b,i) = R_dd(a,b,i) + K(i)*adm.K_dd(a,b,k,j,i);
        }
        for(int c = 0; c < NDIM; ++c){
          for(int d = 0; d < NDIM; ++d){
            ILOOP1(i){
              Riemm4_dd(a,b,i) += - g_uu(c,d,i)*adm.K_dd(a,c,k,j,i)*adm.K_dd(d,b,k,j,i);
            }
          }
        }
      }
    }

    ILOOP1(i) {
      weyl.rpsi4(k,j,i) = 0;
      weyl.ipsi4(k,j,i) = 0;
    }

    for(int a = 0; a < NDIM; ++a){
      for(int b = 0; b < NDIM; ++b){
        ILOOP1(i){
          weyl.rpsi4(k,j,i) += - FR4 * Riemm4_dd(a,b,i) * (
            vvec(a,i) * vvec(b,i) - (-wvec(a,i) * (-wvec(b,i)))
          );
          weyl.ipsi4(k,j,i) += - FR4 * Riemm4_dd(a,b,i) * (
            -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
          );
        }
        for(int c = 0; c < NDIM; ++c){
          ILOOP1(i){
            weyl.rpsi4(k,j,i) += 0.5 * Riemm4_ddd(a,c,b,i) * uvec(c,i) * (
              vvec(a,i) * vvec(b,i) - (-wvec(a,i)*(-wvec(b,i)))
            );
            weyl.ipsi4(k,j,i) += 0.5 * Riemm4_ddd(a,c,b,i) * uvec(c,i) * (
              -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
            );
          }
          for(int d = 0; d < NDIM; ++d){
            ILOOP1(i){
              weyl.rpsi4(k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b,i) * uvec(d,i) * uvec(c,i)) * (
                vvec(a,i) * vvec(b,i) - (-wvec(a,i)*(-wvec(b,i)))
              );
              weyl.ipsi4(k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b,i) * uvec(d,i) * uvec(c,i)) * (
                -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
              );
            }
          }
        }
      }
    }

  }
}
