//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_refinement_cc.cpp
//  \brief Cell-centered restrict and prolongate operators for mesh refinement

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

namespace {

inline Real WenoZChild(const Real cm2,
                       const Real cm1,
                       const Real c0,
                       const Real cp1,
                       const Real cp2,
                       const int b)
{
  const Real d0 = (1.0 / 8.0) * cm2 - (1.0 / 2.0) * cm1 + (3.0 / 8.0) * c0;
  const Real d1 = -(1.0 / 8.0) * cm1 + (1.0 / 8.0) * cp1;
  const Real d2 = -(3.0 / 8.0) * c0 + (1.0 / 2.0) * cp1 - (1.0 / 8.0) * cp2;

  const Real scale =
    std::max(std::max(std::abs(cm2), std::abs(cm1)),
             std::max(std::max(std::abs(c0), std::abs(cp1)), std::abs(cp2)));
  if (scale == 0.0)
    return c0;

  const Real qm2 = cm2 / scale;
  const Real qm1 = cm1 / scale;
  const Real q0  = c0 / scale;
  const Real qp1 = cp1 / scale;
  const Real qp2 = cp2 / scale;

  const Real b0a   = qm2 - 2.0 * qm1 + q0;
  const Real b0b   = qm2 - 4.0 * qm1 + 3.0 * q0;
  const Real beta0 = (13.0 / 12.0) * b0a * b0a + (1.0 / 4.0) * b0b * b0b;
  const Real b1a   = qm1 - 2.0 * q0 + qp1;
  const Real b1b   = qm1 - qp1;
  const Real beta1 = (13.0 / 12.0) * b1a * b1a + (1.0 / 4.0) * b1b * b1b;
  const Real b2a   = q0 - 2.0 * qp1 + qp2;
  const Real b2b   = 3.0 * q0 - 4.0 * qp1 + qp2;
  const Real beta2 = (13.0 / 12.0) * b2a * b2a + (1.0 / 4.0) * b2b * b2b;

  const Real tau = std::abs(beta0 - beta2);
  const Real eps = std::numeric_limits<Real>::epsilon();
  const Real alpha0 = (3.0 / 16.0) *
                      (1.0 + (tau / (beta0 + eps)) * (tau / (beta0 + eps)));
  const Real alpha1 = (5.0 / 8.0) *
                      (1.0 + (tau / (beta1 + eps)) * (tau / (beta1 + eps)));
  const Real alpha2 = (3.0 / 16.0) *
                      (1.0 + (tau / (beta2 + eps)) * (tau / (beta2 + eps)));
  const Real alpha_sum = alpha0 + alpha1 + alpha2;
  const Real delta = (alpha0 * d0 + alpha1 * d1 + alpha2 * d2) / alpha_sum;

  return c0 + (b == 0 ? -delta : delta);
}

inline Real WenoZChildX(const AthenaArray<Real>& coarse,
                        const int n,
                        const int k,
                        const int j,
                        const int i,
                        const int bx)
{
  return WenoZChild(coarse(n, k, j, i - 2),
                    coarse(n, k, j, i - 1),
                    coarse(n, k, j, i),
                    coarse(n, k, j, i + 1),
                    coarse(n, k, j, i + 2),
                    bx);
}

inline Real WenoZChildY(const AthenaArray<Real>& coarse,
                        const int n,
                        const int k,
                        const int j,
                        const int i,
                        const int bx,
                        const int by)
{
  return WenoZChild(WenoZChildX(coarse, n, k, j - 2, i, bx),
                    WenoZChildX(coarse, n, k, j - 1, i, bx),
                    WenoZChildX(coarse, n, k, j, i, bx),
                    WenoZChildX(coarse, n, k, j + 1, i, bx),
                    WenoZChildX(coarse, n, k, j + 2, i, bx),
                    by);
}

inline Real WenoZChildZ(const AthenaArray<Real>& coarse,
                        const int n,
                        const int k,
                        const int j,
                        const int i,
                        const int bx,
                        const int by,
                        const int bz)
{
  return WenoZChild(WenoZChildY(coarse, n, k - 2, j, i, bx, by),
                    WenoZChildY(coarse, n, k - 1, j, i, bx, by),
                    WenoZChildY(coarse, n, k, j, i, bx, by),
                    WenoZChildY(coarse, n, k + 1, j, i, bx, by),
                    WenoZChildY(coarse, n, k + 2, j, i, bx, by),
                    bz);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCellCenteredValues(const AthenaArray<Real>
//! &fine,
//                           AthenaArray<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int
//                           cek)
//  \brief restrict cell centered values

void MeshRefinement::RestrictCellCenteredValues(const AthenaArray<Real>& fine,
                                                AthenaArray<Real>& coarse,
                                                int sn,
                                                int en,
                                                int csi,
                                                int cei,
                                                int csj,
                                                int cej,
                                                int csk,
                                                int cek)
{
  MeshBlock* pmb   = pmy_block_;
  Coordinates* pco = pmb->pcoord;
  int si           = (csi - pmb->cis) * 2 + pmb->is,
      ei           = (cei - pmb->cis) * 2 + pmb->is + 1;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1)
  {  // 3D
    if (uniform_cart_)
    {
      for (int n = sn; n <= en; ++n)
      {
        for (int ck = csk; ck <= cek; ck++)
        {
          int k = (ck - pmb->cks) * 2 + pmb->ks;
          for (int cj = csj; cj <= cej; cj++)
          {
            int j = (cj - pmb->cjs) * 2 + pmb->js;
            for (int ci = csi; ci <= cei; ci++)
            {
              int i = (ci - pmb->cis) * 2 + pmb->is;
              coarse(n, ck, cj, ci) =
                0.125 *
                (((fine(n, k, j, i) + fine(n, k, j + 1, i)) +
                  (fine(n, k, j, i + 1) + fine(n, k, j + 1, i + 1))) +
                 ((fine(n, k + 1, j, i) + fine(n, k + 1, j + 1, i)) +
                  (fine(n, k + 1, j, i + 1) + fine(n, k + 1, j + 1, i + 1))));
            }
          }
        }
      }
    }
    else
    {
      for (int n = sn; n <= en; ++n)
      {
        for (int ck = csk; ck <= cek; ck++)
        {
          int k = (ck - pmb->cks) * 2 + pmb->ks;
          for (int cj = csj; cj <= cej; cj++)
          {
            int j = (cj - pmb->cjs) * 2 + pmb->js;
            pco->CellVolume(k, j, si, ei, fvol_[0][0]);
            pco->CellVolume(k, j + 1, si, ei, fvol_[0][1]);
            pco->CellVolume(k + 1, j, si, ei, fvol_[1][0]);
            pco->CellVolume(k + 1, j + 1, si, ei, fvol_[1][1]);
            for (int ci = csi; ci <= cei; ci++)
            {
              int i = (ci - pmb->cis) * 2 + pmb->is;
              // KGF: add the off-centered quantities first to preserve FP
              // symmetry
              Real tvol = ((fvol_[0][0](i) + fvol_[0][1](i)) +
                           (fvol_[0][0](i + 1) + fvol_[0][1](i + 1))) +
                          ((fvol_[1][0](i) + fvol_[1][1](i)) +
                           (fvol_[1][0](i + 1) + fvol_[1][1](i + 1)));
              // KGF: add the off-centered quantities first to preserve FP
              // symmetry
              coarse(n, ck, cj, ci) =
                (((fine(n, k, j, i) * fvol_[0][0](i) +
                   fine(n, k, j + 1, i) * fvol_[0][1](i)) +
                  (fine(n, k, j, i + 1) * fvol_[0][0](i + 1) +
                   fine(n, k, j + 1, i + 1) * fvol_[0][1](i + 1))) +
                 ((fine(n, k + 1, j, i) * fvol_[1][0](i) +
                   fine(n, k + 1, j + 1, i) * fvol_[1][1](i)) +
                  (fine(n, k + 1, j, i + 1) * fvol_[1][0](i + 1) +
                   fine(n, k + 1, j + 1, i + 1) * fvol_[1][1](i + 1)))) /
                tvol;
            }
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    if (uniform_cart_)
    {
      for (int n = sn; n <= en; ++n)
      {
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          for (int ci = csi; ci <= cei; ci++)
          {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            coarse(n, 0, cj, ci) =
              0.25 * ((fine(n, 0, j, i) + fine(n, 0, j + 1, i)) +
                      (fine(n, 0, j, i + 1) + fine(n, 0, j + 1, i + 1)));
          }
        }
      }
    }
    else
    {
      for (int n = sn; n <= en; ++n)
      {
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          pco->CellVolume(0, j, si, ei, fvol_[0][0]);
          pco->CellVolume(0, j + 1, si, ei, fvol_[0][1]);
          for (int ci = csi; ci <= cei; ci++)
          {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            // KGF: add the off-centered quantities first to preserve FP
            // symmetry
            Real tvol = (fvol_[0][0](i) + fvol_[0][1](i)) +
                        (fvol_[0][0](i + 1) + fvol_[0][1](i + 1));

            // KGF: add the off-centered quantities first to preserve FP
            // symmetry
            coarse(n, 0, cj, ci) =
              ((fine(n, 0, j, i) * fvol_[0][0](i) +
                fine(n, 0, j + 1, i) * fvol_[0][1](i)) +
               (fine(n, 0, j, i + 1) * fvol_[0][0](i + 1) +
                fine(n, 0, j + 1, i + 1) * fvol_[0][1](i + 1))) /
              tvol;
          }
        }
      }
    }
  }
  else
  {  // 1D
    int j = pmb->js, cj = pmb->cjs, k = pmb->ks, ck = pmb->cks;
    if (uniform_cart_)
    {
      for (int n = sn; n <= en; ++n)
      {
        for (int ci = csi; ci <= cei; ci++)
        {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          coarse(n, ck, cj, ci) =
            0.5 * (fine(n, k, j, i) + fine(n, k, j, i + 1));
        }
      }
    }
    else
    {
      for (int n = sn; n <= en; ++n)
      {
        pco->CellVolume(k, j, si, ei, fvol_[0][0]);
        for (int ci = csi; ci <= cei; ci++)
        {
          int i                 = (ci - pmb->cis) * 2 + pmb->is;
          Real tvol             = fvol_[0][0](i) + fvol_[0][0](i + 1);
          coarse(n, ck, cj, ci) = (fine(n, k, j, i) * fvol_[0][0](i) +
                                   fine(n, k, j, i + 1) * fvol_[0][0](i + 1)) /
                                  tvol;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateCellCenteredValues(
//        const AthenaArray<Real> &coarse,AthenaArray<Real> &fine, int sn, int
//        en,, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate cell centered values

void MeshRefinement::ProlongateCellCenteredValues(
  const AthenaArray<Real>& coarse,
  AthenaArray<Real>& fine,
  int sn,
  int en,
  int si,
  int ei,
  int sj,
  int ej,
  int sk,
  int ek)
{
  MeshBlock* pmb   = pmy_block_;
  Coordinates* pco = pmb->pcoord;

  if (pmb->block_size.nx3 > 1)
  {
    if (uniform_cart_)
    {
      const Real oo_H1 = 0.5 / uc_h1_;
      const Real oo_H2 = 0.5 / uc_h2_;
      const Real oo_H3 = 0.5 / uc_h3_;
      const Real hh1   = 0.5 * uc_h1_;
      const Real hh2   = 0.5 * uc_h2_;
      const Real hh3   = 0.5 * uc_h3_;
      for (int n = sn; n <= en; n++)
      {
        for (int k = sk; k <= ek; k++)
        {
          int fk = (k - pmb->cks) * 2 + pmb->ks;
          for (int j = sj; j <= ej; j++)
          {
            int fj = (j - pmb->cjs) * 2 + pmb->js;
            for (int i = si; i <= ei; i++)
            {
              int fi     = (i - pmb->cis) * 2 + pmb->is;
              Real ccval = coarse(n, k, j, i);

              Real gx1m = (ccval - coarse(n, k, j, i - 1)) * oo_H1;
              Real gx1p = (coarse(n, k, j, i + 1) - ccval) * oo_H1;
              Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                          std::min(std::abs(gx1m), std::abs(gx1p));

              Real gx2m = (ccval - coarse(n, k, j - 1, i)) * oo_H2;
              Real gx2p = (coarse(n, k, j + 1, i) - ccval) * oo_H2;
              Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                          std::min(std::abs(gx2m), std::abs(gx2p));

              Real gx3m = (ccval - coarse(n, k - 1, j, i)) * oo_H3;
              Real gx3p = (coarse(n, k + 1, j, i) - ccval) * oo_H3;
              Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                          std::min(std::abs(gx3m), std::abs(gx3p));

              fine(n, fk, fj, fi) =
                ccval - (gx1c * hh1 + gx2c * hh2 + gx3c * hh3);
              fine(n, fk, fj, fi + 1) =
                ccval + (gx1c * hh1 - gx2c * hh2 - gx3c * hh3);
              fine(n, fk, fj + 1, fi) =
                ccval - (gx1c * hh1 - gx2c * hh2 + gx3c * hh3);
              fine(n, fk, fj + 1, fi + 1) =
                ccval + (gx1c * hh1 + gx2c * hh2 - gx3c * hh3);
              fine(n, fk + 1, fj, fi) =
                ccval - (gx1c * hh1 + gx2c * hh2 - gx3c * hh3);
              fine(n, fk + 1, fj, fi + 1) =
                ccval + (gx1c * hh1 - gx2c * hh2 + gx3c * hh3);
              fine(n, fk + 1, fj + 1, fi) =
                ccval - (gx1c * hh1 - gx2c * hh2 - gx3c * hh3);
              fine(n, fk + 1, fj + 1, fi + 1) =
                ccval + (gx1c * hh1 + gx2c * hh2 + gx3c * hh3);
            }
          }
        }
      }
    }
    else
    {
      for (int n = sn; n <= en; n++)
      {
        for (int k = sk; k <= ek; k++)
        {
          int fk           = (k - pmb->cks) * 2 + pmb->ks;
          const Real& x3m  = pcoarsec->x3v(k - 1);
          const Real& x3c  = pcoarsec->x3v(k);
          const Real& x3p  = pcoarsec->x3v(k + 1);
          Real dx3m        = x3c - x3m;
          Real dx3p        = x3p - x3c;
          const Real& fx3m = pco->x3v(fk);
          const Real& fx3p = pco->x3v(fk + 1);
          Real dx3fm       = x3c - fx3m;
          Real dx3fp       = fx3p - x3c;
          for (int j = sj; j <= ej; j++)
          {
            int fj           = (j - pmb->cjs) * 2 + pmb->js;
            const Real& x2m  = pcoarsec->x2v(j - 1);
            const Real& x2c  = pcoarsec->x2v(j);
            const Real& x2p  = pcoarsec->x2v(j + 1);
            Real dx2m        = x2c - x2m;
            Real dx2p        = x2p - x2c;
            const Real& fx2m = pco->x2v(fj);
            const Real& fx2p = pco->x2v(fj + 1);
            Real dx2fm       = x2c - fx2m;
            Real dx2fp       = fx2p - x2c;
            for (int i = si; i <= ei; i++)
            {
              int fi           = (i - pmb->cis) * 2 + pmb->is;
              const Real& x1m  = pcoarsec->x1v(i - 1);
              const Real& x1c  = pcoarsec->x1v(i);
              const Real& x1p  = pcoarsec->x1v(i + 1);
              Real dx1m        = x1c - x1m;
              Real dx1p        = x1p - x1c;
              const Real& fx1m = pco->x1v(fi);
              const Real& fx1p = pco->x1v(fi + 1);
              Real dx1fm       = x1c - fx1m;
              Real dx1fp       = fx1p - x1c;
              Real ccval       = coarse(n, k, j, i);

              Real gx1c, gx2c, gx3c;

              Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
              Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
              gx1c      = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                     std::min(std::abs(gx1m), std::abs(gx1p));

              Real gx2m = (ccval - coarse(n, k, j - 1, i)) / dx2m;
              Real gx2p = (coarse(n, k, j + 1, i) - ccval) / dx2p;
              gx2c      = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                     std::min(std::abs(gx2m), std::abs(gx2p));

              Real gx3m = (ccval - coarse(n, k - 1, j, i)) / dx3m;
              Real gx3p = (coarse(n, k + 1, j, i) - ccval) / dx3p;
              gx3c      = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                     std::min(std::abs(gx3m), std::abs(gx3p));

              fine(n, fk, fj, fi) =
                ccval - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
              fine(n, fk, fj, fi + 1) =
                ccval + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
              fine(n, fk, fj + 1, fi) =
                ccval - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
              fine(n, fk, fj + 1, fi + 1) =
                ccval + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
              fine(n, fk + 1, fj, fi) =
                ccval - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
              fine(n, fk + 1, fj, fi + 1) =
                ccval + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
              fine(n, fk + 1, fj + 1, fi) =
                ccval - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
              fine(n, fk + 1, fj + 1, fi + 1) =
                ccval + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
            }
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    if (uniform_cart_)
    {
      const Real oo_H1 = 0.5 / uc_h1_;
      const Real oo_H2 = 0.5 / uc_h2_;
      const Real hh1   = 0.5 * uc_h1_;
      const Real hh2   = 0.5 * uc_h2_;
      int k = pmb->cks, fk = pmb->ks;
      for (int n = sn; n <= en; n++)
      {
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi     = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(n, k, j, i);

            Real gx1m = (ccval - coarse(n, k, j, i - 1)) * oo_H1;
            Real gx1p = (coarse(n, k, j, i + 1) - ccval) * oo_H1;
            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));

            Real gx2m = (ccval - coarse(n, k, j - 1, i)) * oo_H2;
            Real gx2p = (coarse(n, k, j + 1, i) - ccval) * oo_H2;
            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));

            fine(n, fk, fj, fi)         = ccval - (gx1c * hh1 + gx2c * hh2);
            fine(n, fk, fj, fi + 1)     = ccval + (gx1c * hh1 - gx2c * hh2);
            fine(n, fk, fj + 1, fi)     = ccval - (gx1c * hh1 - gx2c * hh2);
            fine(n, fk, fj + 1, fi + 1) = ccval + (gx1c * hh1 + gx2c * hh2);
          }
        }
      }
    }
    else
    {
      int k = pmb->cks, fk = pmb->ks;
      for (int n = sn; n <= en; n++)
      {
        for (int j = sj; j <= ej; j++)
        {
          int fj           = (j - pmb->cjs) * 2 + pmb->js;
          const Real& x2m  = pcoarsec->x2v(j - 1);
          const Real& x2c  = pcoarsec->x2v(j);
          const Real& x2p  = pcoarsec->x2v(j + 1);
          Real dx2m        = x2c - x2m;
          Real dx2p        = x2p - x2c;
          const Real& fx2m = pco->x2v(fj);
          const Real& fx2p = pco->x2v(fj + 1);
          Real dx2fm       = x2c - fx2m;
          Real dx2fp       = fx2p - x2c;
          for (int i = si; i <= ei; i++)
          {
            int fi           = (i - pmb->cis) * 2 + pmb->is;
            const Real& x1m  = pcoarsec->x1v(i - 1);
            const Real& x1c  = pcoarsec->x1v(i);
            const Real& x1p  = pcoarsec->x1v(i + 1);
            Real dx1m        = x1c - x1m;
            Real dx1p        = x1p - x1c;
            const Real& fx1m = pco->x1v(fi);
            const Real& fx1p = pco->x1v(fi + 1);
            Real dx1fm       = x1c - fx1m;
            Real dx1fp       = fx1p - x1c;
            Real ccval       = coarse(n, k, j, i);

            Real gx1c, gx2c;
            Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
            Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
            gx1c      = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                   std::min(std::abs(gx1m), std::abs(gx1p));

            Real gx2m = (ccval - coarse(n, k, j - 1, i)) / dx2m;
            Real gx2p = (coarse(n, k, j + 1, i) - ccval) / dx2p;
            gx2c      = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                   std::min(std::abs(gx2m), std::abs(gx2p));

            fine(n, fk, fj, fi)     = ccval - (gx1c * dx1fm + gx2c * dx2fm);
            fine(n, fk, fj, fi + 1) = ccval + (gx1c * dx1fp - gx2c * dx2fm);
            fine(n, fk, fj + 1, fi) = ccval - (gx1c * dx1fm - gx2c * dx2fp);
            fine(n, fk, fj + 1, fi + 1) =
              ccval + (gx1c * dx1fp + gx2c * dx2fp);
          }
        }
      }
    }
  }
  else
  {  // 1D
    int k = pmb->cks, fk = pmb->ks, j = pmb->cjs, fj = pmb->js;
    if (uniform_cart_)
    {
      const Real oo_H1 = 0.5 / uc_h1_;
      const Real hh1   = 0.5 * uc_h1_;
      for (int n = sn; n <= en; n++)
      {
        for (int i = si; i <= ei; i++)
        {
          int fi     = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(n, k, j, i);

          Real gx1m = (ccval - coarse(n, k, j, i - 1)) * oo_H1;
          Real gx1p = (coarse(n, k, j, i + 1) - ccval) * oo_H1;
          Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                      std::min(std::abs(gx1m), std::abs(gx1p));

          fine(n, fk, fj, fi)     = ccval - gx1c * hh1;
          fine(n, fk, fj, fi + 1) = ccval + gx1c * hh1;
        }
      }
    }
    else
    {
      for (int n = sn; n <= en; n++)
      {
        for (int i = si; i <= ei; i++)
        {
          int fi           = (i - pmb->cis) * 2 + pmb->is;
          const Real& x1m  = pcoarsec->x1v(i - 1);
          const Real& x1c  = pcoarsec->x1v(i);
          const Real& x1p  = pcoarsec->x1v(i + 1);
          Real dx1m        = x1c - x1m;
          Real dx1p        = x1p - x1c;
          const Real& fx1m = pco->x1v(fi);
          const Real& fx1p = pco->x1v(fi + 1);
          Real dx1fm       = x1c - fx1m;
          Real dx1fp       = fx1p - x1c;
          Real ccval       = coarse(n, k, j, i);

          Real gx1c;
          Real gx1m = (ccval - coarse(n, k, j, i - 1)) / dx1m;
          Real gx1p = (coarse(n, k, j, i + 1) - ccval) / dx1p;
          gx1c      = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                 std::min(std::abs(gx1m), std::abs(gx1p));

          fine(n, fk, fj, fi)     = ccval - gx1c * dx1fm;
          fine(n, fk, fj, fi + 1) = ccval + gx1c * dx1fp;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateCellCenteredWenoZValues(
//!        const AthenaArray<Real> &coarse, AthenaArray<Real> &fine, int sn, int
//!        en, int si, int ei, int sj, int ej, int sk, int ek)
//! \brief prolongate cell-centered values with WENO-Z child averages

void MeshRefinement::ProlongateCellCenteredWenoZValues(
  const AthenaArray<Real>& coarse,
  AthenaArray<Real>& fine,
  int sn,
  int en,
  int si,
  int ei,
  int sj,
  int ej,
  int sk,
  int ek)
{
  MeshBlock* pmb = pmy_block_;

  if (pmb->block_size.nx3 > 1)
  {
    for (int n = sn; n <= en; ++n)
    {
      for (int k = sk; k <= ek; ++k)
      {
        const int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; ++j)
        {
          const int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; ++i)
          {
            const int fi = (i - pmb->cis) * 2 + pmb->is;
            for (int bz = 0; bz < 2; ++bz)
              for (int by = 0; by < 2; ++by)
                for (int bx = 0; bx < 2; ++bx)
                  fine(n, fk + bz, fj + by, fi + bx) =
                    WenoZChildZ(coarse, n, k, j, i, bx, by, bz);
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    const int k  = pmb->cks;
    const int fk = pmb->ks;
    for (int n = sn; n <= en; ++n)
    {
      for (int j = sj; j <= ej; ++j)
      {
        const int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; ++i)
        {
          const int fi = (i - pmb->cis) * 2 + pmb->is;
          for (int by = 0; by < 2; ++by)
            for (int bx = 0; bx < 2; ++bx)
              fine(n, fk, fj + by, fi + bx) =
                WenoZChildY(coarse, n, k, j, i, bx, by);
        }
      }
    }
  }
  else
  {
    const int k  = pmb->cks;
    const int fk = pmb->ks;
    const int j  = pmb->cjs;
    const int fj = pmb->js;
    for (int n = sn; n <= en; ++n)
    {
      for (int i = si; i <= ei; ++i)
      {
        const int fi = (i - pmb->cis) * 2 + pmb->is;
        fine(n, fk, fj, fi)     = WenoZChildX(coarse, n, k, j, i, 0);
        fine(n, fk, fj, fi + 1) = WenoZChildX(coarse, n, k, j, i, 1);
      }
    }
  }
}
