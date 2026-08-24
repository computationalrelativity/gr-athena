//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation
//  of Weyl scalars

// C++ standard headers
#include <algorithm>  // max
#include <cmath>      // exp, pow, sqrt
#include <fstream>
#include <iomanip>
#include <iostream>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"

#if defined(DBG_WEYL_SWSH_SEED_CART)
#include "../utils/spherical_harmonics.hpp"
#endif  // DBG_WEYL_SWSH_SEED_CART

template <typename T>
static int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

// Fully antisymmetric permutation symbol [abc] for a,b,c in {0,1,2}.
// Returns +1 for even permutations of (0,1,2), -1 for odd, 0 if any indices
// repeat. Used to build the 3D Levi-Civita tensor eps^{ijk} = [ijk]/sqrt(detg).
static inline int LeviCivita3(int a, int b, int c)
{
  return (a - b) * (b - c) * (c - a) / 2;
}

// BD: TODO - refactor

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat,
// AthenaArray<Real> & u_weyl)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock

void Z4c::Z4cWeyl(AthenaArray<Real>& u_adm,
                  AthenaArray<Real>& u_mat,
                  AthenaArray<Real>& u_weyl)
{
  using namespace LinearAlgebra;

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  // Z4c aliases needed for chain-rule assembly of physical metric derivatives
  Z4c_vars z4c;
  SetZ4cAliases(storage.u, z4c);

  Matter_vars mat;
  SetMatterAliases(u_mat, mat);

  Weyl_vars weyl;
  SetWeylAliases(u_weyl, weyl);
  weyl.rpsi4.Fill(NAN);
  weyl.ipsi4.Fill(NAN);
  weyl.E_dd.Fill(NAN);
  weyl.B_dd.Fill(NAN);

  // Simplify constants (2 & sqrt 2 factors) featured in re/im[psi4]
  const Real FR4 = 0.25;

#if defined(DBG_WEYL_SWSH_SEED_CART)

  ILOOP3(k, j, i)
  {
    const Real x = mbi.x1(i);
    const Real y = mbi.x2(j);
    const Real z = mbi.x3(k);

    const Real r_xy = std::sqrt(SQR(x) + SQR(y));
    const Real r    = std::sqrt(SQR(x) + SQR(y) + SQR(z));

    Real theta = std::acos(z / r);
    Real phi   = sgn(y) * std::acos(x / (std::sqrt(SQR(x) + SQR(y))));

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

    gra::sph_harm::sYlm(-2, 2, 2, theta, phi, &ylmR, &ylmI);

    weyl.rpsi4(k, j, i) = ylmR;
    weyl.ipsi4(k, j, i) = ylmI;

    // weyl.rpsi4(k,j,i) = std::cos(phi);
    // weyl.ipsi4(k,j,i) = 0;

    gra::sph_harm::sYlm(-2, 4, 0, theta, phi, &ylmR, &ylmI);

    weyl.rpsi4(k, j, i) -= ylmR;
    weyl.ipsi4(k, j, i) -= ylmI;
  }

  return;
#endif  // DBG_WEYL_SWSH_SEED_CART

  //---------------------------------------------------------------------------
  // NOTE: 3D conformal first derivatives (dalpha_d_3d, dchi_d_3d,
  // dbeta_du_3d, dg_ddd_3d) are pre-computed by PrepareZ4cDerivatives().
  // Physical metric derivatives are assembled via chain rule below.
  //---------------------------------------------------------------------------

  ILOOP2(k, j)
  {
    // -----------------------------------------------------------------------------------
    // Chain-rule assembly of physical metric derivatives from conformal
    // quantities
    //
    // gamma_ab = chi^(4/p) * g~_ab
    // d_c(gamma_ab) = chi^(4/p) * [d_c(g~_ab) + (4/p)/chi * g~_ab * dchi_c]
    //
    // d_a d_b(gamma_cd) = chi^(4/p) * {d_a d_b(g~_cd)
    //   + (4/p)/chi * [dchi_a * d_b(g~_cd) + dchi_b * d_a(g~_cd) + ddchi_ab *
    //   g~_cd]
    //   + (4/p)(4/p - 1)/chi^2 * dchi_a * dchi_b * g~_cd}
    //
    // For chi_psi_power == -4 (default): 4/p = -1, 4/p - 1 = -2
    //   chi^(4/p) = 1/chi  (no std::pow needed)
    //
    const Real p = pz4c->opt.chi_psi_power;

    // --- Pass 0: load ddchi_dd from 3D storage ---
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          ddchi_dd(a, b, i) = ddchi_dd_3d(a, b, k, j, i);
        }
      }

    if (p == -4.)
    {
      // --- Specialized path: p == -4, fac = -1, fac2 = -2, chi^(4/p) = 1/chi
      // ---

      // --- Pass 1: first derivatives of physical metric + K ---
      for (int c = 0; c < NDIM; ++c)
        for (int a = 0; a < NDIM; ++a)
          for (int b = a; b < NDIM; ++b)
          {
            ILOOP1(i)
            {
              const Real oochi   = 1.0 / chiRegularized(z4c.chi(k, j, i));
              dg_ddd(c, a, b, i) = oochi * (dg_ddd_3d(c, a, b, k, j, i) -
                                            oochi * z4c.g_dd(a, b, k, j, i) *
                                              dchi_d_3d(c, k, j, i));
              dK_ddd(c, a, b, i) = fd->Dx(c, adm.K_dd(a, b, k, j, i));
            }
          }

      // --- Pass 2: second derivatives of physical metric ---
      for (int c = 0; c < NDIM; ++c)
        for (int d = c; d < NDIM; ++d)
          for (int a = 0; a < NDIM; ++a)
            for (int b = a; b < NDIM; ++b)
            {
              if (a == b)
              {
                ILOOP1(i)
                {
                  const Real oochi    = 1.0 / chiRegularized(z4c.chi(k, j, i));
                  const Real dchi_a   = dchi_d_3d(a, k, j, i);
                  const Real gtil_cd  = z4c.g_dd(c, d, k, j, i);
                  const Real ddg_conf = fd->Dxx(a, z4c.g_dd(c, d, k, j, i));
                  ddg_dddd(a, a, c, d, i) =
                    oochi *
                    (ddg_conf -
                     oochi * (2.0 * dchi_a * dg_ddd_3d(a, c, d, k, j, i) +
                              ddchi_dd(a, a, i) * gtil_cd) +
                     2.0 * oochi * oochi * dchi_a * dchi_a * gtil_cd);
                }
              }
              else
              {
                ILOOP1(i)
                {
                  const Real oochi    = 1.0 / chiRegularized(z4c.chi(k, j, i));
                  const Real dchi_a   = dchi_d_3d(a, k, j, i);
                  const Real dchi_b   = dchi_d_3d(b, k, j, i);
                  const Real gtil_cd  = z4c.g_dd(c, d, k, j, i);
                  const Real ddg_conf = fd->Dx(a, dg_ddd_3d(b, c, d, k, j, i));
                  ddg_dddd(a, b, c, d, i) =
                    oochi * (ddg_conf -
                             oochi * (dchi_a * dg_ddd_3d(b, c, d, k, j, i) +
                                      dchi_b * dg_ddd_3d(a, c, d, k, j, i) +
                                      ddchi_dd(a, b, i) * gtil_cd) +
                             2.0 * oochi * oochi * dchi_a * dchi_b * gtil_cd);
                }
              }
            }
    }
    else
    {
      // --- General path: arbitrary chi_psi_power ---
      const Real fac  = 4.0 / p;    // 4/p
      const Real fac2 = fac - 1.0;  // 4/p - 1

      // --- Pass 1: first derivatives of physical metric + K ---
      for (int c = 0; c < NDIM; ++c)
        for (int a = 0; a < NDIM; ++a)
          for (int b = a; b < NDIM; ++b)
          {
            ILOOP1(i)
            {
              const Real chi_g = chiRegularized(z4c.chi(k, j, i));
              const Real psi4  = std::pow(chi_g, fac);
              const Real oochi = 1.0 / chi_g;
              dg_ddd(c, a, b, i) =
                psi4 * (dg_ddd_3d(c, a, b, k, j, i) +
                        fac * oochi * z4c.g_dd(a, b, k, j, i) *
                          dchi_d_3d(c, k, j, i));
              dK_ddd(c, a, b, i) = fd->Dx(c, adm.K_dd(a, b, k, j, i));
            }
          }

      // --- Pass 2: second derivatives of physical metric ---
      for (int c = 0; c < NDIM; ++c)
        for (int d = c; d < NDIM; ++d)
          for (int a = 0; a < NDIM; ++a)
            for (int b = a; b < NDIM; ++b)
            {
              if (a == b)
              {
                ILOOP1(i)
                {
                  const Real chi_g    = chiRegularized(z4c.chi(k, j, i));
                  const Real psi4     = std::pow(chi_g, fac);
                  const Real oochi    = 1.0 / chi_g;
                  const Real dchi_a   = dchi_d_3d(a, k, j, i);
                  const Real gtil_cd  = z4c.g_dd(c, d, k, j, i);
                  const Real ddg_conf = fd->Dxx(a, z4c.g_dd(c, d, k, j, i));
                  ddg_dddd(a, a, c, d, i) =
                    psi4 *
                    (ddg_conf +
                     fac * oochi *
                       (2.0 * dchi_a * dg_ddd_3d(a, c, d, k, j, i) +
                        ddchi_dd(a, a, i) * gtil_cd) +
                     fac * fac2 * oochi * oochi * dchi_a * dchi_a * gtil_cd);
                }
              }
              else
              {
                ILOOP1(i)
                {
                  const Real chi_g    = chiRegularized(z4c.chi(k, j, i));
                  const Real psi4     = std::pow(chi_g, fac);
                  const Real oochi    = 1.0 / chi_g;
                  const Real dchi_a   = dchi_d_3d(a, k, j, i);
                  const Real dchi_b   = dchi_d_3d(b, k, j, i);
                  const Real gtil_cd  = z4c.g_dd(c, d, k, j, i);
                  const Real ddg_conf = fd->Dx(a, dg_ddd_3d(b, c, d, k, j, i));
                  ddg_dddd(a, b, c, d, i) =
                    psi4 *
                    (ddg_conf +
                     fac * oochi *
                       (dchi_a * dg_ddd_3d(b, c, d, k, j, i) +
                        dchi_b * dg_ddd_3d(a, c, d, k, j, i) +
                        ddchi_dd(a, b, i) * gtil_cd) +
                     fac * fac2 * oochi * oochi * dchi_a * dchi_b * gtil_cd);
                }
              }
            }

    }  // end chi_psi_power branch

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    ILOOP1(i)
    {
      detg(i) = Det3Metric(adm.g_dd, k, j, i);
      Inv3Metric(1. / detg(i),
                 adm.g_dd(0, 0, k, j, i),
                 adm.g_dd(0, 1, k, j, i),
                 adm.g_dd(0, 2, k, j, i),
                 adm.g_dd(1, 1, k, j, i),
                 adm.g_dd(1, 2, k, j, i),
                 adm.g_dd(2, 2, k, j, i),
                 &g_uu(0, 0, i),
                 &g_uu(0, 1, i),
                 &g_uu(0, 2, i),
                 &g_uu(1, 1, i),
                 &g_uu(1, 2, i),
                 &g_uu(2, 2, i));
    }

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for (int c = 0; c < NDIM; ++c)
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          ILOOP1(i)
          {
            Gamma_ddd(c, a, b, i) =
              0.5 *
              (dg_ddd(a, b, c, i) + dg_ddd(b, a, c, i) - dg_ddd(c, a, b, i));
          }
        }

    Gamma_udd.ZeroClear();
    for (int c = 0; c < NDIM; ++c)
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
          for (int d = 0; d < NDIM; ++d)
          {
            ILOOP1(i)
            {
              Gamma_udd(c, a, b, i) += g_uu(c, d, i) * Gamma_ddd(d, a, b, i);
            }
          }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    R.ZeroClear();
    R_dd.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        for (int c = 0; c < NDIM; ++c)
          for (int d = 0; d < NDIM; ++d)
          {
            // Part with the Christoffel symbols
            for (int e = 0; e < NDIM; ++e)
            {
              ILOOP1(i)
              {
                R_dd(a, b, i) += g_uu(c, d, i) * Gamma_udd(e, a, c, i) *
                                 Gamma_ddd(e, b, d, i);
                R_dd(a, b, i) -= g_uu(c, d, i) * Gamma_udd(e, a, b, i) *
                                 Gamma_ddd(e, c, d, i);
              }
            }
            // Wave operator part of the Ricci
            ILOOP1(i)
            {
              R_dd(a, b, i) +=
                0.5 * g_uu(c, d, i) *
                (-ddg_dddd(c, d, a, b, i) - ddg_dddd(a, b, c, d, i) +
                 ddg_dddd(a, c, b, d, i) + ddg_dddd(b, c, a, d, i));
            }
          }
        // R = g^{ab} R_{ab}; triangular loop needs 2x on off-diagonal (a != b)
        // terms since g^{ab} R_{ab} = sum_a g^{aa} R_{aa} + 2 * sum_{a<b}
        // g^{ab} R_{ab}
        ILOOP1(i)
        {
          R(i) += ((a == b) ? 1.0 : 2.0) * g_uu(a, b, i) * R_dd(a, b, i);
        }
      }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        for (int c = 0; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            K_ud(a, b, i) += g_uu(a, c, i) * adm.K_dd(c, b, k, j, i);
          }
        }
      }
      ILOOP1(i)
      {
        K(i) += K_ud(a, a, i);
      }
    }
    // Covariant derivative of K
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = 0; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            DK_ddd(a, b, c, i) = dK_ddd(a, b, c, i);
          }
          for (int d = 0; d < NDIM; ++d)
          {
            ILOOP1(i)
            {
              DK_ddd(a, b, c, i) -=
                Gamma_udd(d, a, b, i) * adm.K_dd(d, c, k, j, i);
              DK_ddd(a, b, c, i) -=
                Gamma_udd(d, a, c, i) * adm.K_dd(b, d, k, j, i);
            }
          }
        }

    // -----------------------------------------------------------------------------------
    // Electric and magnetic parts of the Weyl tensor: E_ij, B_ij
    //
    // These are the projections of the 4D Weyl tensor onto the spatial slice
    // w.r.t. the unit timelike normal n^mu:
    //   E_ij = C_{i mu j nu} n^mu n^nu
    //   B_ij = -*C_{i mu j nu} n^mu n^nu  (dual Weyl, up to sign convention)
    //
    // Built entirely from ADM 3+1 quantities already assembled above
    // (3-Ricci R_dd, trace K, extrinsic curvature K_dd/K_ud, inverse metric
    // g_uu, and the covariant derivative of K, DK_ddd) which themselves come
    // from the ADM metric, its spatial derivatives (dg_ddd/ddg_dddd), and
    // K_ij. Both tensors are symmetric and (in vacuum) trace-free.
    //
    // Gauss equation (electric part):
    //   E_ij = R_ij + K K_ij - K_ik g^kl K_lj
    //
    // Codazzi equation (magnetic part), symmetrized:
    //   B_ij = 1/2 * [ eps_i^{kl} D_k K_lj + eps_j^{kl} D_k K_li ]
    // where eps^{ikl} = [ikl] / sqrt(detg) is the spatial Levi-Civita tensor
    // built from the permutation symbol [ikl] (LeviCivita3 above) and
    // eps_i^{kl} = g_im eps^{mkl}.
    // -----------------------------------------------------------------------------------
    ILOOP1(i)
    {
      const Real oosqrtg = 1.0 / std::sqrt(detg(i));

      // Unsymmetrized magnetic-part candidate: rawB(a,b) = eps_a^{kl} D_k K_lb
      Real rawB[NDIM][NDIM];
      for (int a = 0; a < NDIM; ++a)
        for (int b = 0; b < NDIM; ++b)
        {
          Real val = 0.0;
          for (int m = 0; m < NDIM; ++m)
            for (int kk = 0; kk < NDIM; ++kk)
              for (int ll = 0; ll < NDIM; ++ll)
              {
                const int eps = LeviCivita3(m, kk, ll);
                if (eps == 0) continue;
                val += eps * adm.g_dd(a, m, k, j, i) * oosqrtg *
                       DK_ddd(kk, ll, b, i);
              }
          rawB[a][b] = val;
        }

      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          // --- Electric part: E_ab = R_ab + K K_ab - K_ac g^cd K_db ---
          Real Eab = R_dd(a, b, i) + K(i) * adm.K_dd(a, b, k, j, i);
          for (int c = 0; c < NDIM; ++c)
            for (int d = 0; d < NDIM; ++d)
            {
              Eab -= g_uu(c, d, i) * adm.K_dd(a, c, k, j, i) *
                     adm.K_dd(d, b, k, j, i);
            }

          weyl.E_dd(a, b, k, j, i) = Eab;
          if (a != b) weyl.E_dd(b, a, k, j, i) = Eab;

          // --- Magnetic part: symmetrize rawB ---
          const Real Bab = 0.5 * (rawB[a][b] + rawB[b][a]);

          weyl.B_dd(a, b, k, j, i) = Bab;
          if (a != b) weyl.B_dd(b, a, k, j, i) = Bab;
        }
    }

    //------------------------------------------------------------------------------------
    //     Construct tetrad
    //
    //     Initial tetrad guess. NB, aligned with z axis - possible problem if
    //     points lie on z axis theta and phi vectors degenerate Like BAM start
    //     with phi vector uvec = radial vec vvec = theta vec wvec = phi vec
    ILOOP1(i)
    {
      Real xx = mbi.x1(i);
      if (SQR(mbi.x1(i)) + SQR(mbi.x2(j)) < 1e-10)
        xx = xx + 1e-8;
      uvec(0, i) = xx;
      uvec(1, i) = mbi.x2(j);
      uvec(2, i) = mbi.x3(k);
      vvec(0, i) = xx * mbi.x3(k);
      vvec(1, i) = mbi.x2(j) * mbi.x3(k);
      vvec(2, i) = -SQR(xx) - SQR(mbi.x2(j));
      wvec(0, i) = mbi.x2(j) * -1.0;
      wvec(1, i) = xx;
      wvec(2, i) = 0.0;
    }

    // Gram-Schmidt orthonormalisation with spacetime metric.
    //
    dotp1.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp1(i) += adm.g_dd(a, b, k, j, i) * wvec(a, i) * wvec(b, i);
        }
      }
    }
    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        wvec(a, i) = wvec(a, i) / sqrt(dotp1(i));
      }
    }

    dotp1.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp1(i) += adm.g_dd(a, b, k, j, i) * wvec(a, i) * uvec(b, i);
        }
      }
    }
    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        uvec(a, i) -= dotp1(i) * wvec(a, i);
      }
    }
    dotp1.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp1(i) += adm.g_dd(a, b, k, j, i) * uvec(a, i) * uvec(b, i);
        }
      }
    }

    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        uvec(a, i) = uvec(a, i) / sqrt(dotp1(i));
      }
    }

    dotp1.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp1(i) += adm.g_dd(a, b, k, j, i) * wvec(a, i) * vvec(b, i);
        }
      }
    }
    dotp2.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp2(i) += adm.g_dd(a, b, k, j, i) * uvec(a, i) * vvec(b, i);
        }
      }
    }

    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        vvec(a, i) -= dotp1(i) * wvec(a, i) + dotp2(i) * uvec(a, i);
      }
    }

    dotp1.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
    {
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          dotp1(i) += adm.g_dd(a, b, k, j, i) * vvec(a, i) * vvec(b, i);
        }
      }
    }

    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        vvec(a, i) = vvec(a, i) / sqrt(dotp1(i));
      }
    }

    // ---------------------------------------------------------------------------------
    // Fused Psi4 computation: each 4D Riemann component is evaluated as a
    // local scalar and immediately accumulated into rpsi4/ipsi4, avoiding
    // intermediate storage of Riem3_dddd (81), Riemm4_dddd (81), Riemm4_ddd
    // (27), and Riemm4_dd (9) scratch arrays.
    // ---------------------------------------------------------------------------------
    ILOOP1(i)
    {
      weyl.rpsi4(k, j, i) = 0;
      weyl.ipsi4(k, j, i) = 0;
    }

    // Phase A: Riemm4_dd contribution (Ricci + K*K - K_ac g^cd K_db)
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          Real R4_ab = R_dd(a, b, i) + K(i) * adm.K_dd(a, b, k, j, i);
          for (int c = 0; c < NDIM; ++c)
            for (int d = 0; d < NDIM; ++d)
              R4_ab -= g_uu(c, d, i) * adm.K_dd(a, c, k, j, i) *
                       adm.K_dd(d, b, k, j, i);

          const Real Tr = vvec(a, i) * vvec(b, i) - wvec(a, i) * wvec(b, i);
          const Real Ti = -vvec(a, i) * wvec(b, i) - wvec(a, i) * vvec(b, i);
          weyl.rpsi4(k, j, i) += -FR4 * R4_ab * Tr;
          weyl.ipsi4(k, j, i) += -FR4 * R4_ab * Ti;
        }
      }

    // Phase B: Riemm4_ddd contribution (Codazzi equation)
    // Riemm4_ddd(A,B,C) = -(DK_ddd(C,A,B) - DK_ddd(B,A,C))
    // Consumed as Riemm4_ddd(a,c,b) => A=a, B=c, C=b
    //   = -(DK_ddd(b,a,c) - DK_ddd(c,a,b))
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = 0; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            const Real R4_acb = -(DK_ddd(b, a, c, i) - DK_ddd(c, a, b, i));
            const Real Tr = vvec(a, i) * vvec(b, i) - wvec(a, i) * wvec(b, i);
            const Real Ti = -vvec(a, i) * wvec(b, i) - wvec(a, i) * vvec(b, i);
            weyl.rpsi4(k, j, i) += 0.5 * R4_acb * uvec(c, i) * Tr;
            weyl.ipsi4(k, j, i) += 0.5 * R4_acb * uvec(c, i) * Ti;
          }
        }

    // Phase C: Riemm4_dddd contribution (Gauss equation)
    // Riemm4_dddd(A,B,C,D) = g(A,C)*R(B,D) + g(B,D)*R(A,C)
    //                       - g(A,D)*R(B,C) - g(B,C)*R(A,D)
    //                       - 0.5*R*(g(A,C)*g(B,D) - g(A,D)*g(B,C))
    //                       + K(A,C)*K(B,D) - K(A,D)*K(B,C)
    // Consumed as Riemm4_dddd(d,a,c,b) => A=d, B=a, C=c, D=b
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = 0; c < NDIM; ++c)
          for (int d = 0; d < NDIM; ++d)
          {
            ILOOP1(i)
            {
              const Real R4_dacb =
                adm.g_dd(d, c, k, j, i) * R_dd(a, b, i) +
                adm.g_dd(a, b, k, j, i) * R_dd(d, c, i) -
                adm.g_dd(d, b, k, j, i) * R_dd(a, c, i) -
                adm.g_dd(a, c, k, j, i) * R_dd(d, b, i) -
                0.5 * R(i) *
                  (adm.g_dd(d, c, k, j, i) * adm.g_dd(a, b, k, j, i) -
                   adm.g_dd(d, b, k, j, i) * adm.g_dd(a, c, k, j, i)) +
                adm.K_dd(d, c, k, j, i) * adm.K_dd(a, b, k, j, i) -
                adm.K_dd(d, b, k, j, i) * adm.K_dd(a, c, k, j, i);

              const Real Tr =
                vvec(a, i) * vvec(b, i) - wvec(a, i) * wvec(b, i);
              const Real Ti =
                -vvec(a, i) * wvec(b, i) - wvec(a, i) * vvec(b, i);
              weyl.rpsi4(k, j, i) +=
                -FR4 * R4_dacb * uvec(d, i) * uvec(c, i) * Tr;
              weyl.ipsi4(k, j, i) +=
                -FR4 * R4_dacb * uvec(d, i) * uvec(c, i) * Ti;
            }
          }
  }
}
