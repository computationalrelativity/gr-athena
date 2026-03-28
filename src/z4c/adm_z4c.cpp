//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM
//  decomposition

// C++ standard headers
#include <cmath>  // pow
#include <fstream>
#include <iostream>

// Athena++ headers
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../utils/linear_algebra.hpp"
#include "z4c.hpp"
#include "z4c_macro.hpp"

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

void Z4c::ADMToZ4c(AthenaArray<Real>& u_adm, AthenaArray<Real>& u)
{
  using namespace LinearAlgebra;

  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  //--------------------------------------------------------------------------------------
  // Conformal factor, conformal metric, and trace of extrinsic curvature
  //
  GLOOP2(k, j)
  {
    GLOOP1(i)
    {
      z4c.alpha(k, j, i) = adm.alpha(k, j, i);
    }

    for (int a = 0; a < NDIM; ++a)
      GLOOP1(i)
      {
        z4c.beta_u(a, k, j, i) = adm.beta_u(a, k, j, i);
      }

    // Conformal factor
    if (opt.chi_psi_power == -4.)
    {
      GLOOP1(i)
      {
        detg(i) = Det3Metric(adm.g_dd, k, j, i);
        // oopsi4(i) = pow(detg(i), -1./3.);
        // z4c.chi(k,j,i) = pow(detg(i), 1./12.*opt.chi_psi_power);

        oopsi4(i)        = std::cbrt(1.0 / detg(i));
        z4c.chi(k, j, i) = oopsi4(i);
      }
    }
    else
    {
      GLOOP1(i)
      {
        detg(i)          = Det3Metric(adm.g_dd, k, j, i);
        oopsi4(i)        = pow(detg(i), -1. / 3.);
        z4c.chi(k, j, i) = pow(detg(i), 1. / 12. * opt.chi_psi_power);
      }
    }

    // Conformal metric and extrinsic curvature
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        GLOOP1(i)
        {
          z4c.g_dd(a, b, k, j, i) = oopsi4(i) * adm.g_dd(a, b, k, j, i);
          Kt_dd(a, b, i)          = oopsi4(i) * adm.K_dd(a, b, k, j, i);
        }
      }

    // Determinant of the conformal metric and trace of conf. extr. curvature
    GLOOP1(i)
    {
      detg(i)           = Det3Metric(z4c.g_dd, k, j, i);
      z4c.Khat(k, j, i) = TraceRank2(1.0 / detg(i),
                                     z4c.g_dd(0, 0, k, j, i),
                                     z4c.g_dd(0, 1, k, j, i),
                                     z4c.g_dd(0, 2, k, j, i),
                                     z4c.g_dd(1, 1, k, j, i),
                                     z4c.g_dd(1, 2, k, j, i),
                                     z4c.g_dd(2, 2, k, j, i),
                                     Kt_dd(0, 0, i),
                                     Kt_dd(0, 1, i),
                                     Kt_dd(0, 2, i),
                                     Kt_dd(1, 1, i),
                                     Kt_dd(1, 2, i),
                                     Kt_dd(2, 2, i));
    }

    // Conformal traceless extrinsic curvatore
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        GLOOP1(i)
        {
          z4c.A_dd(a, b, k, j, i) = Kt_dd(a, b, i) - (1. / 3.) *
                                                       z4c.Khat(k, j, i) *
                                                       z4c.g_dd(a, b, k, j, i);
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
  GLOOP3(k, j, i)
  {
    detg(i) = Det3Metric(z4c.g_dd, k, j, i);
    Inv3Metric(1.0 / detg(i),
               z4c.g_dd(0, 0, k, j, i),
               z4c.g_dd(0, 1, k, j, i),
               z4c.g_dd(0, 2, k, j, i),
               z4c.g_dd(1, 1, k, j, i),
               z4c.g_dd(1, 2, k, j, i),
               z4c.g_dd(2, 2, k, j, i),
               &g_uu(0, 0, k, j, i),
               &g_uu(0, 1, k, j, i),
               &g_uu(0, 2, k, j, i),
               &g_uu(1, 1, k, j, i),
               &g_uu(1, 2, k, j, i),
               &g_uu(2, 2, k, j, i));
  }

  // Compute Gamma's
  z4c.Gam_u.ZeroClear();
  ILOOP2(k, j)
  {
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          // Is it ba or ab like in the pseudocode? Is the contraction correct?
          z4c.Gam_u(a, k, j, i) -= fd->Dx(b, g_uu(b, a, k, j, i));
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

  // compute derived ADM
  PrepareAuxExtended(storage.aux_extended, u, u_adm);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cToADM(AthenaArray<Real> & u, AthenaArray<Real> & u_adm)
// \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock

void Z4c::Z4cToADM(AthenaArray<Real>& u, AthenaArray<Real>& u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  GLOOP2(k, j)
  {
    GLOOP1(i)
    {
      adm.alpha(k, j, i) = z4c.alpha(k, j, i);
    }

    for (int a = 0; a < NDIM; ++a)
      GLOOP1(i)
      {
        adm.beta_u(a, k, j, i) = z4c.beta_u(a, k, j, i);
      }

    // psi4
    if (opt.chi_psi_power == -4.)
    {
      GLOOP1(i)
      {
        adm.psi4(k, j, i) = 1.0 / z4c.chi(k, j, i);
      }
    }
    else
    {
      GLOOP1(i)
      {
        adm.psi4(k, j, i) = std::pow(z4c.chi(k, j, i), 4. / opt.chi_psi_power);
      }
    }
    // g_ab
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        GLOOP1(i)
        {
          adm.g_dd(a, b, k, j, i) =
            adm.psi4(k, j, i) * z4c.g_dd(a, b, k, j, i);
        }
      }
    // K_ab
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        GLOOP1(i)
        {
          adm.K_dd(a, b, k, j, i) =
            adm.psi4(k, j, i) * z4c.A_dd(a, b, k, j, i) +
            (1. / 3.) * (z4c.Khat(k, j, i) + 2. * z4c.Theta(k, j, i)) *
              adm.g_dd(a, b, k, j, i);
        }
      }
  }

  PrepareAuxExtended(storage.aux_extended, u, u_adm);
}

void Z4c::Z4cToADM(AA& u,
                   AA& u_adm,
                   const int il,
                   const int iu,
                   const int jl,
                   const int ju,
                   const int kl,
                   const int ku,
                   bool skip_physical)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    {
      const bool sp_kj = (skip_physical && (mbi.jl <= j) && (j <= mbi.ju) &&
                          (mbi.kl <= k) && (k <= mbi.ku));

#pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
        {
          continue;
        }
        adm.alpha(k, j, i) = z4c.alpha(k, j, i);
      }

      for (int a = 0; a < NDIM; ++a)
#pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
          {
            continue;
          }
          adm.beta_u(a, k, j, i) = z4c.beta_u(a, k, j, i);
        }

      // psi4
      if (opt.chi_psi_power == -4.)
      {
#pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
          {
            continue;
          }
          adm.psi4(k, j, i) = 1.0 / z4c.chi(k, j, i);
        }
      }
      else
      {
#pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
          {
            continue;
          }
          adm.psi4(k, j, i) =
            std::pow(z4c.chi(k, j, i), 4. / opt.chi_psi_power);
        }
      }

      // g_ab
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
          {
            if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
            {
              continue;
            }

            adm.g_dd(a, b, k, j, i) =
              adm.psi4(k, j, i) * z4c.g_dd(a, b, k, j, i);
          }
      // K_ab
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
#pragma omp simd
          for (int i = il; i <= iu; ++i)
          {
            if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
            {
              continue;
            }

            adm.K_dd(a, b, k, j, i) =
              adm.psi4(k, j, i) * z4c.A_dd(a, b, k, j, i) +
              (1. / 3.) * (z4c.Khat(k, j, i) + 2. * z4c.Theta(k, j, i)) *
                adm.g_dd(a, b, k, j, i);
          }
    }

  PrepareAuxExtended(
    storage.aux_extended, u, u_adm, il, iu, jl, ju, kl, ku, skip_physical);
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> &
// u_mat)
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

void Z4c::ADMConstraints(AthenaArray<Real>& u_con,
                         AthenaArray<Real>& u_adm,
                         AthenaArray<Real>& u_mat,
                         AthenaArray<Real>& u_z4c)
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
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
            for (int d = c; d < NDIM; ++d)
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
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
          for (int c = 0; c < NDIM; ++c)
            for (int d = c; d < NDIM; ++d)
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

    // Gamma^a = g^{bc} Gamma^a_{bc}; both g^{bc} and Gamma^a_{bc} are
    // symmetric in (b,c), so use triangular sum with 2x off-diagonal factor.
    Gamma_u.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = b; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            Gamma_u(a, i) +=
              ((b == c) ? 1.0 : 2.0) * g_uu(b, c, i) * Gamma_udd(a, b, c, i);
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
            for (int f = 0; f < NDIM; ++f)
              ILOOP1(i)
              {
                R_dd(a, b, i) +=
                  g_uu(d, f, i) *
                  (Gamma_udd(c, a, f, i) * Gamma_ddd(c, b, d, i) -
                   Gamma_udd(c, a, b, i) * Gamma_ddd(c, f, d, i));
              }

        for (int c = 0; c < NDIM; ++c)
          for (int d = 0; d < NDIM; ++d)
            ILOOP1(i)
            {
              R_dd(a, b, i) +=
                0.5 * g_uu(c, d, i) *
                (-ddg_dddd(b, a, c, d, i) + ddg_dddd(d, a, b, c, i) +
                 ddg_dddd(d, b, a, c, i) - ddg_dddd(d, c, a, b, i));
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
    // K^a_b K^b_a
    KK.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          KK(i) += K_ud(a, b, i) * K_ud(b, a, i);
        }
      }
    // Covariant derivative of K
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = b; c < NDIM; ++c)
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
    DK_udd.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = b; c < NDIM; ++c)
          for (int d = 0; d < NDIM; ++d)
          {
            ILOOP1(i)
            {
              DK_udd(a, b, c, i) += g_uu(a, d, i) * DK_ddd(d, b, c, i);
            }
          }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    ILOOP1(i)
    {
      con.H(k, j, i) = R(i) + SQR(K(i)) - KK(i) - 16 * M_PI * mat.rho(k, j, i);
    }
    // Momentum constraint (contravariant)
    //
    M_u.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          M_u(a, i) -= 8 * M_PI * g_uu(a, b, i) * mat.S_d(b, k, j, i);
        }
        for (int c = 0; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            M_u(a, i) += g_uu(a, b, i) * DK_udd(c, b, c, i);
            M_u(a, i) -= g_uu(b, c, i) * DK_udd(a, b, c, i);
          }
        }
      }
    // Momentum constraint (covariant)
    for (int a = 0; a < NDIM; ++a)
    {
      ILOOP1(i)
      {
        con.M_d(a, k, j, i) = 0;
      }

      for (int b = 0; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          con.M_d(a, k, j, i) += adm.g_dd(a, b, k, j, i) * M_u(b, i);
        }
      }
    }

    // |M|^2 = g_{ab} M^a M^b; g_{ab} symmetric and M^a M^b symmetric in (a,b),
    // so use triangular sum with 2x off-diagonal factor.
    ILOOP1(i)
    {
      con.M(k, j, i) = 0.;
    }
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          con.M(k, j, i) += ((a == b) ? 1.0 : 2.0) * adm.g_dd(a, b, k, j, i) *
                            M_u(a, i) * M_u(b, i);
        }
      }
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of conformal metric (read from stored 3D arrays)
    for (int c = 0; c < NDIM; ++c)
      for (int a = 0; a < NDIM; ++a)
        for (int b = a; b < NDIM; ++b)
        {
          ILOOP1(i)
          {
            dg_ddd(c, a, b, i) = dg_ddd_3d(c, a, b, k, j, i);
          }
        }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    ILOOP1(i)
    {
      detg(i) = Det3Metric(z4c.g_dd, k, j, i);
      Inv3Metric(1. / detg(i),
                 z4c.g_dd(0, 0, k, j, i),
                 z4c.g_dd(0, 1, k, j, i),
                 z4c.g_dd(0, 2, k, j, i),
                 z4c.g_dd(1, 1, k, j, i),
                 z4c.g_dd(1, 2, k, j, i),
                 z4c.g_dd(2, 2, k, j, i),
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

    // Gamma^a = g^{bc} Gamma^a_{bc}; both symmetric in (b,c), so use
    // triangular sum with 2x off-diagonal factor.
    Gamma_u.ZeroClear();
    for (int a = 0; a < NDIM; ++a)
      for (int b = 0; b < NDIM; ++b)
        for (int c = b; c < NDIM; ++c)
        {
          ILOOP1(i)
          {
            Gamma_u(a, i) +=
              ((b == c) ? 1.0 : 2.0) * g_uu(b, c, i) * Gamma_udd(a, b, c, i);
          }
        }
    // Constraint violation Z (norm squared)
    ILOOP1(i)
    {
      con.Z(k, j, i) = 0.;
    }
    // |Z|^2 = g_{ab} Z^a Z^b; g_{ab} symmetric and Z^a Z^b symmetric in (a,b),
    // so use triangular sum with 2x off-diagonal factor.
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        ILOOP1(i)
        {
          con.Z(k, j, i) += ((a == b) ? 0.25 : 0.5) * adm.g_dd(a, b, k, j, i) *
                            (z4c.Gam_u(a, k, j, i) - Gamma_u(a, i)) *
                            (z4c.Gam_u(b, k, j, i) - Gamma_u(b, i));
        }
      }
    // Radius cut + constraint violation monitor C^2
    {
      const Real r2_max = SQR(pz4c->opt.r_max_con);
      ILOOP1(i)
      {
        const Real R2 = SQR(mbi.x1(i)) + SQR(mbi.x2(j)) + SQR(mbi.x3(k));
        if (R2 > r2_max)
        {
          con.H(k, j, i) = 0.;
          con.M(k, j, i) = 0.;
          con.Z(k, j, i) = 0.;
          con.C(k, j, i) = 0.;
          for (int a = 0; a < NDIM; ++a)
            con.M_d(a, k, j, i) = 0.;
        }
        else
        {
          con.C(k, j, i) = SQR(con.H(k, j, i)) + con.M(k, j, i) +
                           SQR(z4c.Theta(k, j, i)) + 4.0 * con.Z(k, j, i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMMinkowski(AthenaArray<Real> & u)
// \brief Initialize ADM vars to Minkowski

void Z4c::ADMMinkowski(AthenaArray<Real>& u_adm)
{
  ADM_vars adm;
  SetADMAliases(u_adm, adm);
  adm.psi4.Fill(1.);
  adm.K_dd.ZeroClear();

  GLOOP3(k, j, i)
  {
    for (int a = 0; a < NDIM; ++a)
      for (int b = a; b < NDIM; ++b)
      {
        adm.g_dd(a, b, k, j, i) = (a == b ? 1. : 0.);
      }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::MatterVacuum(AthenaArray<Real> & u_mat)
// \brief Initialize ADM vars to vacuum

void Z4c::MatterVacuum(AthenaArray<Real>& u_mat)
{
  u_mat.ZeroClear();
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::PrepareZ4cDerivatives(AthenaArray<Real> & u)
// \brief Pre-compute 3D first derivatives of conformal Z4c fields
//
// Computes d(alpha)/dx_b, d(chi)/dx_b, d(beta^c)/dx_b, d(g~_cd)/dx_b
// over extended stencil ranges and stores in 3D arrays (dalpha_d_3d, etc.).
// These arrays are consumed by:
//   - Z4cRHS (next substep, replacing its own pre-pass)
//   - ADMConstraints, Z4cWeyl (last substep, via chain rule)

void Z4c::PrepareZ4cDerivatives(AthenaArray<Real>& u)
{
  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  const int ng_ext = NGHOST - 1;
  for (int b = 0; b < NDIM; ++b)
  {
    const int klo = (b == 2) ? IX_KL : (IX_KL - (pz4c->mbi.f3 ? ng_ext : 0));
    const int khi = (b == 2) ? IX_KU : (IX_KU + (pz4c->mbi.f3 ? ng_ext : 0));
    const int jlo = (b == 1) ? IX_JL : (IX_JL - (pz4c->mbi.f2 ? ng_ext : 0));
    const int jhi = (b == 1) ? IX_JU : (IX_JU + (pz4c->mbi.f2 ? ng_ext : 0));
    const int ilo = (b == 0) ? IX_IL : (IX_IL - ng_ext);
    const int ihi = (b == 0) ? IX_IU : (IX_IU + ng_ext);
    for (int k = klo; k <= khi; ++k)
      for (int j = jlo; j <= jhi; ++j)
      {
        // Scalars: alpha, chi
        _Pragma("omp simd") for (int i = ilo; i <= ihi; ++i)
        {
          dalpha_d_3d(b, k, j, i) = fd->Dx(b, z4c.alpha(k, j, i));
          dchi_d_3d(b, k, j, i)   = fd->Dx(b, z4c.chi(k, j, i));
        }
        // Vectors: beta
        for (int c = 0; c < NDIM; ++c)
        {
          _Pragma("omp simd") for (int i = ilo; i <= ihi; ++i)
          {
            dbeta_du_3d(b, c, k, j, i) = fd->Dx(b, z4c.beta_u(c, k, j, i));
          }
        }
        // Symmetric tensors: g_dd (conformal)
        for (int c = 0; c < NDIM; ++c)
          for (int d = c; d < NDIM; ++d)
          {
            _Pragma("omp simd") for (int i = ilo; i <= ihi; ++i)
            {
              dg_ddd_3d(b, c, d, k, j, i) = fd->Dx(b, z4c.g_dd(c, d, k, j, i));
            }
          }
      }
  }

  // Second derivatives of chi: 3 diagonal Dxx + 3 off-diagonal Dx on
  // dchi_d_3d. Stored in 3D so Z4cRHS, ADMConstraints, and Z4cWeyl can read
  // without recomputing.
  for (int a = 0; a < NDIM; ++a)
  {
    ILOOP2(k, j)
    {
      ILOOP1(i)
      {
        ddchi_dd_3d(a, a, k, j, i) = fd->Dxx(a, z4c.chi(k, j, i));
      }
    }
    for (int b = a + 1; b < NDIM; ++b)
    {
      ILOOP2(k, j)
      {
        ILOOP1(i)
        {
          ddchi_dd_3d(a, b, k, j, i) = fd->Dx(a, dchi_d_3d(b, k, j, i));
        }
      }
    }
  }

  // Fill storage.aux with physical (ADM) derivatives via chain rule
  // This makes derivatives available for hydro/M1 source terms.
  //
  // d_c(gamma_ab) = chi^(4/p) * [d_c(g~_ab) + (4/p)/chi * g~_ab * dchi_c]
  //
  const Real p = pz4c->opt.chi_psi_power;

  if (p == -4.)
  {
    // Specialized path: 4/p = -1, chi^(4/p) = 1/chi
    ILOOP3(k, j, i)
    {
      const Real oochi = 1.0 / chiRegularized(z4c.chi(k, j, i));

      for (int c = 0; c < NDIM; ++c)
      {
        const Real dchi_c = dchi_d_3d(c, k, j, i);
        for (int a = 0; a < NDIM; ++a)
          for (int b = a; b < NDIM; ++b)
          {
            aux.dg_ddd(c, a, b, k, j, i) =
              oochi * (dg_ddd_3d(c, a, b, k, j, i) -
                       oochi * z4c.g_dd(a, b, k, j, i) * dchi_c);
            if (a != b)
              aux.dg_ddd(c, b, a, k, j, i) = aux.dg_ddd(c, a, b, k, j, i);
          }
      }

      // Shift derivatives (direct copy from stored conformal = physical)
      for (int c = 0; c < NDIM; ++c)
        for (int a = 0; a < NDIM; ++a)
        {
          aux.dbeta_du(c, a, k, j, i) = dbeta_du_3d(c, a, k, j, i);
        }

      // Lapse derivatives (direct copy from stored conformal = physical)
      for (int c = 0; c < NDIM; ++c)
      {
        aux.dalpha_d(c, k, j, i) = dalpha_d_3d(c, k, j, i);
      }
    }
  }
  else
  {
    // General path: arbitrary chi_psi_power
    const Real fac = 4.0 / p;

    ILOOP3(k, j, i)
    {
      const Real chi_g   = chiRegularized(z4c.chi(k, j, i));
      const Real chi_pow = std::pow(chi_g, fac);
      const Real oochi   = 1.0 / chi_g;

      for (int c = 0; c < NDIM; ++c)
      {
        const Real dchi_c = dchi_d_3d(c, k, j, i);
        for (int a = 0; a < NDIM; ++a)
          for (int b = a; b < NDIM; ++b)
          {
            aux.dg_ddd(c, a, b, k, j, i) =
              chi_pow * (dg_ddd_3d(c, a, b, k, j, i) +
                         fac * oochi * z4c.g_dd(a, b, k, j, i) * dchi_c);
            if (a != b)
              aux.dg_ddd(c, b, a, k, j, i) = aux.dg_ddd(c, a, b, k, j, i);
          }
      }

      // Shift derivatives (direct copy from stored conformal = physical)
      for (int c = 0; c < NDIM; ++c)
        for (int a = 0; a < NDIM; ++a)
        {
          aux.dbeta_du(c, a, k, j, i) = dbeta_du_3d(c, a, k, j, i);
        }

      // Lapse derivatives (direct copy from stored conformal = physical)
      for (int c = 0; c < NDIM; ++c)
      {
        aux.dalpha_d(c, k, j, i) = dalpha_d_3d(c, k, j, i);
      }
    }
  }  // end chi_psi_power branch
}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::InitializeZ4cDerivatives(AthenaArray<Real> & u)
// \brief One-time initialization of 3D derivative arrays for fresh MeshBlocks
//
// On a fresh MeshBlock (or after AMR), the stored 3D derivative arrays have
// not yet been filled by PrepareZ4cDerivatives. This function fills them so
// that the first Z4cRHS call can safely read from them. Subsequent substeps
// use PREP_Z4C_DERIV instead.

void Z4c::InitializeZ4cDerivatives(AthenaArray<Real>& u)
{
  if (z4c_derivs_initialized)
    return;
  PrepareZ4cDerivatives(u);
  z4c_derivs_initialized = true;
}

void Z4c::PrepareAuxExtended(AA& u_aux_extended, AA& u, AA& u_adm)
{
  using namespace LinearAlgebra;

  MeshBlock* pmb      = pmy_block;
  GRDynamical* pco_gr = static_cast<GRDynamical*>(pmb->pcoord);

  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Aux_extended_vars aux_extended;
  SetAuxExtendedAliases(u_aux_extended, aux_extended);

  // direct method
  /*
  GLOOP3(k, j, i)
  {
    aux_extended.gs_sqrt_detgamma(k,j,i) = std::sqrt(
      Det3Metric(adm.g_dd,k,j,i)
    );
  }
  */
  // conformal factor based
  // For chi_psi_power == -4: chi_pow/2 = -3/2, so chi^(-3/2) =
  // 1/(chi*sqrt(chi))
  if (pz4c->opt.chi_psi_power == -4.)
  {
    GLOOP3(k, j, i)
    {
      const Real chi         = std::abs(z4c.chi(k, j, i));
      const Real chi_guarded = std::max(chi, pz4c->opt.chi_div_floor);
      aux_extended.gs_sqrt_detgamma(k, j, i) =
        1.0 / (chi_guarded * std::sqrt(chi_guarded));
    }
  }
  else
  {
    const Real chi_pow = 12.0 / pz4c->opt.chi_psi_power;
    GLOOP3(k, j, i)
    {
      const Real chi         = std::abs(z4c.chi(k, j, i));
      const Real chi_guarded = std::max(chi, pz4c->opt.chi_div_floor);
      aux_extended.gs_sqrt_detgamma(k, j, i) =
        std::pow(chi_guarded, chi_pow / 2.0);
    }
  }

  // Pre-compute physical inverse metric gamma^{ij} = Inv(gamma_{ij})
  // Uses det(gamma) = SQR(sqrt_detgamma) already computed above.
  GLOOP3(k, j, i)
  {
    const Real oo_det_gamma =
      1.0 / SQR(aux_extended.gs_sqrt_detgamma(k, j, i));
    Inv3Metric(oo_det_gamma,
               adm.g_dd(0, 0, k, j, i),
               adm.g_dd(0, 1, k, j, i),
               adm.g_dd(0, 2, k, j, i),
               adm.g_dd(1, 1, k, j, i),
               adm.g_dd(1, 2, k, j, i),
               adm.g_dd(2, 2, k, j, i),
               &aux_extended.g_uu(0, 0, k, j, i),
               &aux_extended.g_uu(0, 1, k, j, i),
               &aux_extended.g_uu(0, 2, k, j, i),
               &aux_extended.g_uu(1, 1, k, j, i),
               &aux_extended.g_uu(1, 2, k, j, i),
               &aux_extended.g_uu(2, 2, k, j, i));
  }

#if FLUID_ENABLED

#if defined(Z4C_VC_ENABLED)
  ILOOP2(k, j)
  {
    pco_gr->GetMatterField(ms_detgamma_, aux_extended.gs_sqrt_detgamma, k, j);
    ILOOP1(i)
    {
      aux_extended.ms_sqrt_detgamma(k, j, i) = std::abs(ms_detgamma_(i));
    }
  }
#else

  GLOOP3(k, j, i)
  {
    aux_extended.ms_sqrt_detgamma(k, j, i) =
      (aux_extended.gs_sqrt_detgamma(k, j, i));
  }
#endif

#endif  // FLUID_ENABLED
}

void Z4c::PrepareAuxExtended(AA& u_aux_extended,
                             AA& u,
                             AA& u_adm,
                             const int il,
                             const int iu,
                             const int jl,
                             const int ju,
                             const int kl,
                             const int ku,
                             bool skip_physical)
{
  using namespace LinearAlgebra;

  Z4c_vars z4c;
  SetZ4cAliases(u, z4c);

  ADM_vars adm;
  SetADMAliases(u_adm, adm);

  Aux_extended_vars aux_extended;
  SetAuxExtendedAliases(u_aux_extended, aux_extended);

  MeshBlock* pmb = pmy_block;

  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    {
      const bool sp_kj = (skip_physical && (mbi.jl <= j) && (j <= mbi.ju) &&
                          (mbi.kl <= k) && (k <= mbi.ku));

      // conformal factor based
      // For chi_psi_power == -4: chi_pow/2 = -3/2, so chi^(-3/2) =
      // 1/(chi*sqrt(chi))
      if (pz4c->opt.chi_psi_power == -4.)
      {
#pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
          {
            continue;
          }
          const Real chi         = std::abs(z4c.chi(k, j, i));
          const Real chi_guarded = std::max(chi, pz4c->opt.chi_div_floor);
          aux_extended.gs_sqrt_detgamma(k, j, i) =
            1.0 / (chi_guarded * std::sqrt(chi_guarded));
        }
      }
      else
      {
        const Real chi_pow = 12.0 / pz4c->opt.chi_psi_power;
#pragma omp simd
        for (int i = il; i <= iu; ++i)
        {
          if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
          {
            continue;
          }
          const Real chi         = std::abs(z4c.chi(k, j, i));
          const Real chi_guarded = std::max(chi, pz4c->opt.chi_div_floor);
          aux_extended.gs_sqrt_detgamma(k, j, i) =
            std::pow(chi_guarded, chi_pow / 2.0);
        }
      }
    }

  // Pre-compute physical inverse metric gamma^{ij} = Inv(gamma_{ij})
  // Uses det(gamma) = SQR(sqrt_detgamma) already computed above.
  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    {
      const bool sp_kj = (skip_physical && (mbi.jl <= j) && (j <= mbi.ju) &&
                          (mbi.kl <= k) && (k <= mbi.ku));

#pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
        {
          continue;
        }
        const Real oo_det_gamma =
          1.0 / SQR(aux_extended.gs_sqrt_detgamma(k, j, i));
        Inv3Metric(oo_det_gamma,
                   adm.g_dd(0, 0, k, j, i),
                   adm.g_dd(0, 1, k, j, i),
                   adm.g_dd(0, 2, k, j, i),
                   adm.g_dd(1, 1, k, j, i),
                   adm.g_dd(1, 2, k, j, i),
                   adm.g_dd(2, 2, k, j, i),
                   &aux_extended.g_uu(0, 0, k, j, i),
                   &aux_extended.g_uu(0, 1, k, j, i),
                   &aux_extended.g_uu(0, 2, k, j, i),
                   &aux_extended.g_uu(1, 1, k, j, i),
                   &aux_extended.g_uu(1, 2, k, j, i),
                   &aux_extended.g_uu(2, 2, k, j, i));
      }
    }

#if FLUID_ENABLED

#if defined(Z4C_VC_ENABLED)
  ILOOP2(k, j)
  {
    const bool sp_kj = (skip_physical && (mbi.jl <= j) && (j <= mbi.ju) &&
                        (mbi.kl <= k) && (k <= mbi.ku));

    pco_gr->GetMatterField(ms_detgamma_, aux_extended.gs_sqrt_detgamma, k, j);

    ILOOP1(i)
    {
      if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
      {
        continue;
      }

      aux_extended.ms_sqrt_detgamma(k, j, i) = std::abs(ms_detgamma_(i));
    }
  }
#else
  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
    {
      const bool sp_kj = (skip_physical && (mbi.jl <= j) && (j <= mbi.ju) &&
                          (mbi.kl <= k) && (k <= mbi.ku));

#pragma omp simd
      for (int i = il; i <= iu; ++i)
      {
        if (sp_kj && (mbi.il <= i) && (i <= mbi.iu))
        {
          continue;
        }

        aux_extended.ms_sqrt_detgamma(k, j, i) =
          (aux_extended.gs_sqrt_detgamma(k, j, i));
      }
    }

#endif

#endif  // FLUID_ENABLED
}