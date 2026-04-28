//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_refinement_fc.cpp
//  \brief Face-centered restrict and prolongate operators for mesh refinement

// C++ headers
#include <algorithm>  // min
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX1(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk,
//      int cek)
//  \brief restrict the x1 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX1(const AthenaArray<Real>& fine,
                                     AthenaArray<Real>& coarse,
                                     int csi,
                                     int cei,
                                     int csj,
                                     int cej,
                                     int csk,
                                     int cek)
{
  MeshBlock* pmb   = pmy_block_;
  Coordinates* pco = pmb->pcoord;
  int si = (csi - pmb->cis) * 2 + pmb->is, ei = (cei - pmb->cis) * 2 + pmb->is;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1)
  {  // 3D
    if (uniform_cart_)
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
            coarse(ck, cj, ci) =
              0.25 * ((fine(k, j, i) + fine(k, j + 1, i)) +
                      (fine(k + 1, j, i) + fine(k + 1, j + 1, i)));
          }
        }
      }
    }
    else
    {
      for (int ck = csk; ck <= cek; ck++)
      {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          pco->Face1Area(k, j, si, ei, sarea_x1_[0][0]);
          pco->Face1Area(k, j + 1, si, ei, sarea_x1_[0][1]);
          pco->Face1Area(k + 1, j, si, ei, sarea_x1_[1][0]);
          pco->Face1Area(k + 1, j + 1, si, ei, sarea_x1_[1][1]);
          for (int ci = csi; ci <= cei; ci++)
          {
            int i      = (ci - pmb->cis) * 2 + pmb->is;
            Real tarea = sarea_x1_[0][0](i) + sarea_x1_[0][1](i) +
                         sarea_x1_[1][0](i) + sarea_x1_[1][1](i);
            coarse(ck, cj, ci) = (fine(k, j, i) * sarea_x1_[0][0](i) +
                                  fine(k, j + 1, i) * sarea_x1_[0][1](i) +
                                  fine(k + 1, j, i) * sarea_x1_[1][0](i) +
                                  fine(k + 1, j + 1, i) * sarea_x1_[1][1](i)) /
                                 tarea;
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    if (uniform_cart_)
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++)
        {
          int i               = (ci - pmb->cis) * 2 + pmb->is;
          coarse(csk, cj, ci) = 0.5 * (fine(k, j, i) + fine(k, j + 1, i));
        }
      }
    }
    else
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        pco->Face1Area(k, j, si, ei, sarea_x1_[0][0]);
        pco->Face1Area(k, j + 1, si, ei, sarea_x1_[0][1]);
        for (int ci = csi; ci <= cei; ci++)
        {
          int i               = (ci - pmb->cis) * 2 + pmb->is;
          Real tarea          = sarea_x1_[0][0](i) + sarea_x1_[0][1](i);
          coarse(csk, cj, ci) = (fine(k, j, i) * sarea_x1_[0][0](i) +
                                 fine(k, j + 1, i) * sarea_x1_[0][1](i)) /
                                tarea;
        }
      }
    }
  }
  else
  {  // 1D - no restriction, just copy
    for (int ci = csi; ci <= cei; ci++)
    {
      int i                = (ci - pmb->cis) * 2 + pmb->is;
      coarse(csk, csj, ci) = fine(pmb->ks, pmb->js, i);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX2(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk,
//      int cek)
//  \brief restrict the x2 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX2(const AthenaArray<Real>& fine,
                                     AthenaArray<Real>& coarse,
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
      for (int ck = csk; ck <= cek; ck++)
      {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          for (int ci = csi; ci <= cei; ci++)
          {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            coarse(ck, cj, ci) =
              0.25 * ((fine(k, j, i) + fine(k, j, i + 1)) +
                      (fine(k + 1, j, i) + fine(k + 1, j, i + 1)));
          }
        }
      }
    }
    else
    {
      for (int ck = csk; ck <= cek; ck++)
      {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++)
        {
          int j     = (cj - pmb->cjs) * 2 + pmb->js;
          bool pole = pco->IsPole(j);
          if (!pole)
          {
            pco->Face2Area(k, j, si, ei, sarea_x2_[0][0]);
            pco->Face2Area(k + 1, j, si, ei, sarea_x2_[1][0]);
          }
          else
          {
            for (int ci = csi; ci <= cei; ++ci)
            {
              int i              = (ci - pmb->cis) * 2 + pmb->is;
              sarea_x2_[0][0](i) = pco->dx1f(i);
              sarea_x2_[1][0](i) = pco->dx1f(i);
            }
          }
          for (int ci = csi; ci <= cei; ci++)
          {
            int i      = (ci - pmb->cis) * 2 + pmb->is;
            Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i + 1) +
                         sarea_x2_[1][0](i) + sarea_x2_[1][0](i + 1);
            coarse(ck, cj, ci) =
              (fine(k, j, i) * sarea_x2_[0][0](i) +
               fine(k, j, i + 1) * sarea_x2_[0][0](i + 1) +
               fine(k + 1, j, i) * sarea_x2_[1][0](i) +
               fine(k + 1, j, i + 1) * sarea_x2_[1][0](i + 1)) /
              tarea;
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    if (uniform_cart_)
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++)
        {
          int i                    = (ci - pmb->cis) * 2 + pmb->is;
          coarse(pmb->cks, cj, ci) = 0.5 * (fine(k, j, i) + fine(k, j, i + 1));
        }
      }
    }
    else
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j     = (cj - pmb->cjs) * 2 + pmb->js;
        bool pole = pco->IsPole(j);
        if (!pole)
        {
          pco->Face2Area(k, j, si, ei, sarea_x2_[0][0]);
        }
        else
        {
          for (int ci = csi; ci <= cei; ++ci)
          {
            int i              = (ci - pmb->cis) * 2 + pmb->is;
            sarea_x2_[0][0](i) = pco->dx1f(i);
          }
        }
        for (int ci = csi; ci <= cei; ci++)
        {
          int i      = (ci - pmb->cis) * 2 + pmb->is;
          Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i + 1);
          coarse(pmb->cks, cj, ci) =
            (fine(k, j, i) * sarea_x2_[0][0](i) +
             fine(k, j, i + 1) * sarea_x2_[0][0](i + 1)) /
            tarea;
        }
      }
    }
  }
  else
  {  // 1D
    if (uniform_cart_)
    {
      int k = pmb->ks, j = pmb->js;
      for (int ci = csi; ci <= cei; ci++)
      {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(pmb->cks, pmb->cjs, ci) =
          0.5 * (fine(k, j, i) + fine(k, j, i + 1));
      }
    }
    else
    {
      int k = pmb->ks, j = pmb->js;
      pco->Face2Area(k, j, si, ei, sarea_x2_[0][0]);
      for (int ci = csi; ci <= cei; ci++)
      {
        int i      = (ci - pmb->cis) * 2 + pmb->is;
        Real tarea = sarea_x2_[0][0](i) + sarea_x2_[0][0](i + 1);
        coarse(pmb->cks, pmb->cjs, ci) =
          (fine(k, j, i) * sarea_x2_[0][0](i) +
           fine(k, j, i + 1) * sarea_x2_[0][0](i + 1)) /
          tarea;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFieldX3(const AthenaArray<Real> &fine
//      AthenaArray<Real> &coarse, int csi, int cei, int csj, int cej, int csk,
//      int cek)
//  \brief restrict the x3 field data and set them into the coarse buffer

void MeshRefinement::RestrictFieldX3(const AthenaArray<Real>& fine,
                                     AthenaArray<Real>& coarse,
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
      for (int ck = csk; ck <= cek; ck++)
      {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          for (int ci = csi; ci <= cei; ci++)
          {
            int i = (ci - pmb->cis) * 2 + pmb->is;
            coarse(ck, cj, ci) =
              0.25 * ((fine(k, j, i) + fine(k, j, i + 1)) +
                      (fine(k, j + 1, i) + fine(k, j + 1, i + 1)));
          }
        }
      }
    }
    else
    {
      for (int ck = csk; ck <= cek; ck++)
      {
        int k = (ck - pmb->cks) * 2 + pmb->ks;
        for (int cj = csj; cj <= cej; cj++)
        {
          int j = (cj - pmb->cjs) * 2 + pmb->js;
          pco->Face3Area(k, j, si, ei, sarea_x3_[0][0]);
          pco->Face3Area(k, j + 1, si, ei, sarea_x3_[0][1]);
          for (int ci = csi; ci <= cei; ci++)
          {
            int i      = (ci - pmb->cis) * 2 + pmb->is;
            Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i + 1) +
                         sarea_x3_[0][1](i) + sarea_x3_[0][1](i + 1);
            coarse(ck, cj, ci) =
              (fine(k, j, i) * sarea_x3_[0][0](i) +
               fine(k, j, i + 1) * sarea_x3_[0][0](i + 1) +
               fine(k, j + 1, i) * sarea_x3_[0][1](i) +
               fine(k, j + 1, i + 1) * sarea_x3_[0][1](i + 1)) /
              tarea;
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    if (uniform_cart_)
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        for (int ci = csi; ci <= cei; ci++)
        {
          int i = (ci - pmb->cis) * 2 + pmb->is;
          coarse(pmb->cks, cj, ci) =
            0.25 * ((fine(k, j, i) + fine(k, j, i + 1)) +
                    (fine(k, j + 1, i) + fine(k, j + 1, i + 1)));
        }
      }
    }
    else
    {
      int k = pmb->ks;
      for (int cj = csj; cj <= cej; cj++)
      {
        int j = (cj - pmb->cjs) * 2 + pmb->js;
        pco->Face3Area(k, j, si, ei, sarea_x3_[0][0]);
        pco->Face3Area(k, j + 1, si, ei, sarea_x3_[0][1]);
        for (int ci = csi; ci <= cei; ci++)
        {
          int i      = (ci - pmb->cis) * 2 + pmb->is;
          Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i + 1) +
                       sarea_x3_[0][1](i) + sarea_x3_[0][1](i + 1);
          coarse(pmb->cks, cj, ci) =
            (fine(k, j, i) * sarea_x3_[0][0](i) +
             fine(k, j, i + 1) * sarea_x3_[0][0](i + 1) +
             fine(k, j + 1, i) * sarea_x3_[0][1](i) +
             fine(k, j + 1, i + 1) * sarea_x3_[0][1](i + 1)) /
            tarea;
        }
      }
    }
  }
  else
  {  // 1D
    if (uniform_cart_)
    {
      int k = pmb->ks, j = pmb->js;
      for (int ci = csi; ci <= cei; ci++)
      {
        int i = (ci - pmb->cis) * 2 + pmb->is;
        coarse(pmb->cks, pmb->cjs, ci) =
          0.5 * (fine(k, j, i) + fine(k, j, i + 1));
      }
    }
    else
    {
      int k = pmb->ks, j = pmb->js;
      pco->Face3Area(k, j, si, ei, sarea_x3_[0][0]);
      for (int ci = csi; ci <= cei; ci++)
      {
        int i      = (ci - pmb->cis) * 2 + pmb->is;
        Real tarea = sarea_x3_[0][0](i) + sarea_x3_[0][0](i + 1);
        coarse(pmb->cks, pmb->cjs, ci) =
          (fine(k, j, i) * sarea_x3_[0][0](i) +
           fine(k, j, i + 1) * sarea_x3_[0][0](i + 1)) /
          tarea;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX1(const AthenaArray<Real>
//! &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int
//      ek)
//  \brief prolongate x1 face-centered fields shared between coarse and fine
//  levels

void MeshRefinement::ProlongateSharedFieldX1(const AthenaArray<Real>& coarse,
                                             AthenaArray<Real>& fine,
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
      const Real oo_H2 = 0.5 / uc_h2_;
      const Real oo_H3 = 0.5 / uc_h3_;
      const Real hh2   = 0.5 * uc_h2_;
      const Real hh3   = 0.5 * uc_h3_;
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi     = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(k, j, i);

            Real gx2m = (ccval - coarse(k, j - 1, i)) * oo_H2;
            Real gx2p = (coarse(k, j + 1, i) - ccval) * oo_H2;
            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));
            Real gx3m = (ccval - coarse(k - 1, j, i)) * oo_H3;
            Real gx3p = (coarse(k + 1, j, i) - ccval) * oo_H3;
            Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                        std::min(std::abs(gx3m), std::abs(gx3p));

            fine(fk, fj, fi)         = ccval - gx2c * hh2 - gx3c * hh3;
            fine(fk, fj + 1, fi)     = ccval + gx2c * hh2 - gx3c * hh3;
            fine(fk + 1, fj, fi)     = ccval - gx2c * hh2 + gx3c * hh3;
            fine(fk + 1, fj + 1, fi) = ccval + gx2c * hh2 + gx3c * hh3;
          }
        }
      }
    }
    else
    {
      for (int k = sk; k <= ek; k++)
      {
        int fk           = (k - pmb->cks) * 2 + pmb->ks;
        const Real& x3m  = pcoarsec->x3s1(k - 1);
        const Real& x3c  = pcoarsec->x3s1(k);
        const Real& x3p  = pcoarsec->x3s1(k + 1);
        Real dx3m        = x3c - x3m;
        Real dx3p        = x3p - x3c;
        const Real& fx3m = pco->x3s1(fk);
        const Real& fx3p = pco->x3s1(fk + 1);
        for (int j = sj; j <= ej; j++)
        {
          int fj           = (j - pmb->cjs) * 2 + pmb->js;
          const Real& x2m  = pcoarsec->x2s1(j - 1);
          const Real& x2c  = pcoarsec->x2s1(j);
          const Real& x2p  = pcoarsec->x2s1(j + 1);
          Real dx2m        = x2c - x2m;
          Real dx2p        = x2p - x2c;
          const Real& fx2m = pco->x2s1(fj);
          const Real& fx2p = pco->x2s1(fj + 1);
          for (int i = si; i <= ei; i++)
          {
            int fi     = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(k, j, i);

            Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
            Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));
            Real gx3m = (ccval - coarse(k - 1, j, i)) / dx3m;
            Real gx3p = (coarse(k + 1, j, i) - ccval) / dx3p;
            Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                        std::min(std::abs(gx3m), std::abs(gx3p));

            fine(fk, fj, fi) =
              ccval - gx2c * (x2c - fx2m) - gx3c * (x3c - fx3m);
            fine(fk, fj + 1, fi) =
              ccval + gx2c * (fx2p - x2c) - gx3c * (x3c - fx3m);
            fine(fk + 1, fj, fi) =
              ccval - gx2c * (x2c - fx2m) + gx3c * (fx3p - x3c);
            fine(fk + 1, fj + 1, fi) =
              ccval + gx2c * (fx2p - x2c) + gx3c * (fx3p - x3c);
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    if (uniform_cart_)
    {
      const Real oo_H2 = 0.5 / uc_h2_;
      const Real hh2   = 0.5 * uc_h2_;
      int k = pmb->cks, fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++)
        {
          int fi     = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx2m = (ccval - coarse(k, j - 1, i)) * oo_H2;
          Real gx2p = (coarse(k, j + 1, i) - ccval) * oo_H2;
          Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                      std::min(std::abs(gx2m), std::abs(gx2p));

          fine(fk, fj, fi)     = ccval - gx2c * hh2;
          fine(fk, fj + 1, fi) = ccval + gx2c * hh2;
        }
      }
    }
    else
    {
      int k = pmb->cks, fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj           = (j - pmb->cjs) * 2 + pmb->js;
        const Real& x2m  = pcoarsec->x2s1(j - 1);
        const Real& x2c  = pcoarsec->x2s1(j);
        const Real& x2p  = pcoarsec->x2s1(j + 1);
        Real dx2m        = x2c - x2m;
        Real dx2p        = x2p - x2c;
        const Real& fx2m = pco->x2s1(fj);
        const Real& fx2p = pco->x2s1(fj + 1);
        for (int i = si; i <= ei; i++)
        {
          int fi     = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
          Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
          Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                      std::min(std::abs(gx2m), std::abs(gx2p));

          fine(fk, fj, fi)     = ccval - gx2c * (x2c - fx2m);
          fine(fk, fj + 1, fi) = ccval + gx2c * (fx2p - x2c);
        }
      }
    }
  }
  else
  {  // 1D
    for (int i = si; i <= ei; i++)
    {
      int fi         = (i - pmb->cis) * 2 + pmb->is;
      fine(0, 0, fi) = coarse(0, 0, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX2(const AthenaArray<Real>
//! &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int
//      ek)
//  \brief prolongate x2 face-centered fields shared between coarse and fine
//  levels

void MeshRefinement::ProlongateSharedFieldX2(const AthenaArray<Real>& coarse,
                                             AthenaArray<Real>& fine,
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
      const Real oo_H3 = 0.5 / uc_h3_;
      const Real hh1   = 0.5 * uc_h1_;
      const Real hh3   = 0.5 * uc_h3_;
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi     = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(k, j, i);

            Real gx1m = (ccval - coarse(k, j, i - 1)) * oo_H1;
            Real gx1p = (coarse(k, j, i + 1) - ccval) * oo_H1;
            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));
            Real gx3m = (ccval - coarse(k - 1, j, i)) * oo_H3;
            Real gx3p = (coarse(k + 1, j, i) - ccval) * oo_H3;
            Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                        std::min(std::abs(gx3m), std::abs(gx3p));

            fine(fk, fj, fi)         = ccval - gx1c * hh1 - gx3c * hh3;
            fine(fk, fj, fi + 1)     = ccval + gx1c * hh1 - gx3c * hh3;
            fine(fk + 1, fj, fi)     = ccval - gx1c * hh1 + gx3c * hh3;
            fine(fk + 1, fj, fi + 1) = ccval + gx1c * hh1 + gx3c * hh3;
          }
        }
      }
    }
    else
    {
      for (int k = sk; k <= ek; k++)
      {
        int fk           = (k - pmb->cks) * 2 + pmb->ks;
        const Real& x3m  = pcoarsec->x3s2(k - 1);
        const Real& x3c  = pcoarsec->x3s2(k);
        const Real& x3p  = pcoarsec->x3s2(k + 1);
        Real dx3m        = x3c - x3m;
        Real dx3p        = x3p - x3c;
        const Real& fx3m = pco->x3s2(fk);
        const Real& fx3p = pco->x3s2(fk + 1);
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi           = (i - pmb->cis) * 2 + pmb->is;
            const Real& x1m  = pcoarsec->x1s2(i - 1);
            const Real& x1c  = pcoarsec->x1s2(i);
            const Real& x1p  = pcoarsec->x1s2(i + 1);
            Real dx1m        = x1c - x1m;
            Real dx1p        = x1p - x1c;
            const Real& fx1m = pco->x1s2(fi);
            const Real& fx1p = pco->x1s2(fi + 1);
            Real ccval       = coarse(k, j, i);

            Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
            Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));
            Real gx3m = (ccval - coarse(k - 1, j, i)) / dx3m;
            Real gx3p = (coarse(k + 1, j, i) - ccval) / dx3p;
            Real gx3c = 0.5 * (SIGN(gx3m) + SIGN(gx3p)) *
                        std::min(std::abs(gx3m), std::abs(gx3p));

            fine(fk, fj, fi) =
              ccval - gx1c * (x1c - fx1m) - gx3c * (x3c - fx3m);
            fine(fk, fj, fi + 1) =
              ccval + gx1c * (fx1p - x1c) - gx3c * (x3c - fx3m);
            fine(fk + 1, fj, fi) =
              ccval - gx1c * (x1c - fx1m) + gx3c * (fx3p - x3c);
            fine(fk + 1, fj, fi + 1) =
              ccval + gx1c * (fx1p - x1c) + gx3c * (fx3p - x3c);
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
      const Real hh1   = 0.5 * uc_h1_;
      int k = pmb->cks, fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++)
        {
          int fi     = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1)) * oo_H1;
          Real gx1p = (coarse(k, j, i + 1) - ccval) * oo_H1;
          Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                      std::min(std::abs(gx1m), std::abs(gx1p));

          fine(fk, fj, fi)     = ccval - gx1c * hh1;
          fine(fk, fj, fi + 1) = ccval + gx1c * hh1;
        }
      }
    }
    else
    {
      int k = pmb->cks, fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++)
        {
          int fi           = (i - pmb->cis) * 2 + pmb->is;
          const Real& x1m  = pcoarsec->x1s2(i - 1);
          const Real& x1c  = pcoarsec->x1s2(i);
          const Real& x1p  = pcoarsec->x1s2(i + 1);
          const Real& fx1m = pco->x1s2(fi);
          const Real& fx1p = pco->x1s2(fi + 1);
          Real ccval       = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1)) / (x1c - x1m);
          Real gx1p = (coarse(k, j, i + 1) - ccval) / (x1p - x1c);
          Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                      std::min(std::abs(gx1m), std::abs(gx1p));

          fine(fk, fj, fi)     = ccval - gx1c * (x1c - fx1m);
          fine(fk, fj, fi + 1) = ccval + gx1c * (fx1p - x1c);
        }
      }
    }
  }
  else
  {
    if (uniform_cart_)
    {
      const Real oo_H1 = 0.5 / uc_h1_;
      const Real hh1   = 0.5 * uc_h1_;
      for (int i = si; i <= ei; i++)
      {
        int fi     = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(0, 0, i);
        Real gxm   = (ccval - coarse(0, 0, i - 1)) * oo_H1;
        Real gxp   = (coarse(0, 0, i + 1) - ccval) * oo_H1;
        Real gxc   = 0.5 * (SIGN(gxm) + SIGN(gxp)) *
                   std::min(std::abs(gxm), std::abs(gxp));
        fine(0, 0, fi) = fine(0, 1, fi) = ccval - gxc * hh1;
        fine(0, 0, fi + 1) = fine(0, 1, fi + 1) = ccval + gxc * hh1;
      }
    }
    else
    {
      for (int i = si; i <= ei; i++)
      {
        int fi   = (i - pmb->cis) * 2 + pmb->is;
        Real gxm = (coarse(0, 0, i) - coarse(0, 0, i - 1)) /
                   (pcoarsec->x1s2(i) - pcoarsec->x1s2(i - 1));
        Real gxp = (coarse(0, 0, i + 1) - coarse(0, 0, i)) /
                   (pcoarsec->x1s2(i + 1) - pcoarsec->x1s2(i));
        Real gxc = 0.5 * (SIGN(gxm) + SIGN(gxp)) *
                   std::min(std::abs(gxm), std::abs(gxp));
        fine(0, 0, fi) = fine(0, 1, fi) =
          coarse(0, 0, i) - gxc * (pcoarsec->x1s2(i) - pco->x1s2(fi));
        fine(0, 0, fi + 1) = fine(0, 1, fi + 1) =
          coarse(0, 0, i) + gxc * (pco->x1s2(fi + 1) - pcoarsec->x1s2(i));
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateSharedFieldX3(const AthenaArray<Real>
//! &coarse,
//      AthenaArray<Real> &fine, int si, int ei, int sj, int ej, int sk, int
//      ek)
//  \brief prolongate x3 face-centered fields shared between coarse and fine
//  levels

void MeshRefinement::ProlongateSharedFieldX3(const AthenaArray<Real>& coarse,
                                             AthenaArray<Real>& fine,
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
      const Real hh1   = 0.5 * uc_h1_;
      const Real hh2   = 0.5 * uc_h2_;
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi     = (i - pmb->cis) * 2 + pmb->is;
            Real ccval = coarse(k, j, i);

            Real gx1m = (ccval - coarse(k, j, i - 1)) * oo_H1;
            Real gx1p = (coarse(k, j, i + 1) - ccval) * oo_H1;
            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));
            Real gx2m = (ccval - coarse(k, j - 1, i)) * oo_H2;
            Real gx2p = (coarse(k, j + 1, i) - ccval) * oo_H2;
            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));

            fine(fk, fj, fi)         = ccval - gx1c * hh1 - gx2c * hh2;
            fine(fk, fj, fi + 1)     = ccval + gx1c * hh1 - gx2c * hh2;
            fine(fk, fj + 1, fi)     = ccval - gx1c * hh1 + gx2c * hh2;
            fine(fk, fj + 1, fi + 1) = ccval + gx1c * hh1 + gx2c * hh2;
          }
        }
      }
    }
    else
    {
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj           = (j - pmb->cjs) * 2 + pmb->js;
          const Real& x2m  = pcoarsec->x2s3(j - 1);
          const Real& x2c  = pcoarsec->x2s3(j);
          const Real& x2p  = pcoarsec->x2s3(j + 1);
          Real dx2m        = x2c - x2m;
          Real dx2p        = x2p - x2c;
          const Real& fx2m = pco->x2s3(fj);
          const Real& fx2p = pco->x2s3(fj + 1);
          for (int i = si; i <= ei; i++)
          {
            int fi           = (i - pmb->cis) * 2 + pmb->is;
            const Real& x1m  = pcoarsec->x1s3(i - 1);
            const Real& x1c  = pcoarsec->x1s3(i);
            const Real& x1p  = pcoarsec->x1s3(i + 1);
            Real dx1m        = x1c - x1m;
            Real dx1p        = x1p - x1c;
            const Real& fx1m = pco->x1s3(fi);
            const Real& fx1p = pco->x1s3(fi + 1);
            Real ccval       = coarse(k, j, i);

            Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
            Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
            Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                        std::min(std::abs(gx1m), std::abs(gx1p));
            Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
            Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
            Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                        std::min(std::abs(gx2m), std::abs(gx2p));

            fine(fk, fj, fi) =
              ccval - gx1c * (x1c - fx1m) - gx2c * (x2c - fx2m);
            fine(fk, fj, fi + 1) =
              ccval + gx1c * (fx1p - x1c) - gx2c * (x2c - fx2m);
            fine(fk, fj + 1, fi) =
              ccval - gx1c * (x1c - fx1m) + gx2c * (fx2p - x2c);
            fine(fk, fj + 1, fi + 1) =
              ccval + gx1c * (fx1p - x1c) + gx2c * (fx2p - x2c);
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
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++)
        {
          int fi     = (i - pmb->cis) * 2 + pmb->is;
          Real ccval = coarse(k, j, i);

          Real gx1m = (ccval - coarse(k, j, i - 1)) * oo_H1;
          Real gx1p = (coarse(k, j, i + 1) - ccval) * oo_H1;
          Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                      std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx2m = (ccval - coarse(k, j - 1, i)) * oo_H2;
          Real gx2p = (coarse(k, j + 1, i) - ccval) * oo_H2;
          Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                      std::min(std::abs(gx2m), std::abs(gx2p));

          fine(fk, fj, fi) = fine(fk + 1, fj, fi) =
            ccval - gx1c * hh1 - gx2c * hh2;
          fine(fk, fj, fi + 1) = fine(fk + 1, fj, fi + 1) =
            ccval + gx1c * hh1 - gx2c * hh2;
          fine(fk, fj + 1, fi) = fine(fk + 1, fj + 1, fi) =
            ccval - gx1c * hh1 + gx2c * hh2;
          fine(fk, fj + 1, fi + 1) = fine(fk + 1, fj + 1, fi + 1) =
            ccval + gx1c * hh1 + gx2c * hh2;
        }
      }
    }
    else
    {
      int k = pmb->cks, fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj           = (j - pmb->cjs) * 2 + pmb->js;
        const Real& x2m  = pcoarsec->x2s3(j - 1);
        const Real& x2c  = pcoarsec->x2s3(j);
        const Real& x2p  = pcoarsec->x2s3(j + 1);
        Real dx2m        = x2c - x2m;
        Real dx2p        = x2p - x2c;
        const Real& fx2m = pco->x2s3(fj);
        const Real& fx2p = pco->x2s3(fj + 1);
        Real dx2fm       = x2c - fx2m;
        Real dx2fp       = fx2p - x2c;
        for (int i = si; i <= ei; i++)
        {
          int fi           = (i - pmb->cis) * 2 + pmb->is;
          const Real& x1m  = pcoarsec->x1s3(i - 1);
          const Real& x1c  = pcoarsec->x1s3(i);
          const Real& x1p  = pcoarsec->x1s3(i + 1);
          Real dx1m        = x1c - x1m;
          Real dx1p        = x1p - x1c;
          const Real& fx1m = pco->x1s3(fi);
          const Real& fx1p = pco->x1s3(fi + 1);
          Real dx1fm       = x1c - fx1m;
          Real dx1fp       = fx1p - x1c;
          Real ccval       = coarse(k, j, i);

          // calculate 2D gradients using the minmod limiter
          Real gx1m = (ccval - coarse(k, j, i - 1)) / dx1m;
          Real gx1p = (coarse(k, j, i + 1) - ccval) / dx1p;
          Real gx1c = 0.5 * (SIGN(gx1m) + SIGN(gx1p)) *
                      std::min(std::abs(gx1m), std::abs(gx1p));
          Real gx2m = (ccval - coarse(k, j - 1, i)) / dx2m;
          Real gx2p = (coarse(k, j + 1, i) - ccval) / dx2p;
          Real gx2c = 0.5 * (SIGN(gx2m) + SIGN(gx2p)) *
                      std::min(std::abs(gx2m), std::abs(gx2p));

          // interpolate on to the finer grid
          fine(fk, fj, fi) = fine(fk + 1, fj, fi) =
            ccval - gx1c * dx1fm - gx2c * dx2fm;
          fine(fk, fj, fi + 1) = fine(fk + 1, fj, fi + 1) =
            ccval + gx1c * dx1fp - gx2c * dx2fm;
          fine(fk, fj + 1, fi) = fine(fk + 1, fj + 1, fi) =
            ccval - gx1c * dx1fm + gx2c * dx2fp;
          fine(fk, fj + 1, fi + 1) = fine(fk + 1, fj + 1, fi + 1) =
            ccval + gx1c * dx1fp + gx2c * dx2fp;
        }
      }
    }
  }
  else
  {
    if (uniform_cart_)
    {
      const Real oo_H1 = 0.5 / uc_h1_;
      const Real hh1   = 0.5 * uc_h1_;
      for (int i = si; i <= ei; i++)
      {
        int fi     = (i - pmb->cis) * 2 + pmb->is;
        Real ccval = coarse(0, 0, i);
        Real gxm   = (ccval - coarse(0, 0, i - 1)) * oo_H1;
        Real gxp   = (coarse(0, 0, i + 1) - ccval) * oo_H1;
        Real gxc   = 0.5 * (SIGN(gxm) + SIGN(gxp)) *
                   std::min(std::abs(gxm), std::abs(gxp));
        fine(0, 0, fi) = fine(1, 0, fi) = ccval - gxc * hh1;
        fine(0, 0, fi + 1) = fine(1, 0, fi + 1) = ccval + gxc * hh1;
      }
    }
    else
    {
      for (int i = si; i <= ei; i++)
      {
        int fi   = (i - pmb->cis) * 2 + pmb->is;
        Real gxm = (coarse(0, 0, i) - coarse(0, 0, i - 1)) /
                   (pcoarsec->x1s3(i) - pcoarsec->x1s3(i - 1));
        Real gxp = (coarse(0, 0, i + 1) - coarse(0, 0, i)) /
                   (pcoarsec->x1s3(i + 1) - pcoarsec->x1s3(i));
        Real gxc = 0.5 * (SIGN(gxm) + SIGN(gxp)) *
                   std::min(std::abs(gxm), std::abs(gxp));
        fine(0, 0, fi) = fine(1, 0, fi) =
          coarse(0, 0, i) - gxc * (pcoarsec->x1s3(i) - pco->x1s3(fi));
        fine(0, 0, fi + 1) = fine(1, 0, fi + 1) =
          coarse(0, 0, i) + gxc * (pco->x1s3(fi + 1) - pcoarsec->x1s3(i));
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateInternalField(FaceField &fine,
//                           int si, int ei, int sj, int ej, int sk, int ek)
//  \brief prolongate the internal face-centered fields

void MeshRefinement::ProlongateInternalField(FaceField& fine,
                                             int si,
                                             int ei,
                                             int sj,
                                             int ej,
                                             int sk,
                                             int ek)
{
  MeshBlock* pmb   = pmy_block_;
  Coordinates* pco = pmb->pcoord;
  int fsi          = (si - pmb->cis) * 2 + pmb->is,
      fei          = (ei - pmb->cis) * 2 + pmb->is + 1;
  if (pmb->block_size.nx3 > 1)
  {
    if (uniform_cart_)
    {
      const Real A1      = uc_h2_ * uc_h3_;
      const Real A2      = uc_h1_ * uc_h3_;
      const Real A3      = uc_h1_ * uc_h2_;
      const Real oo_A1   = 1.0 / A1;
      const Real oo_A2   = 1.0 / A2;
      const Real oo_A3   = 1.0 / A3;
      const Real Sdx1    = SQR(2.0 * uc_h1_);
      const Real Sdx2    = SQR(2.0 * uc_h2_);
      const Real Sdx3    = SQR(2.0 * uc_h3_);
      const Real oo_SH23 = 0.125 / (Sdx2 + Sdx3);
      const Real oo_SH13 = 0.125 / (Sdx1 + Sdx3);
      const Real oo_SH12 = 0.125 / (Sdx1 + Sdx2);
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          for (int i = si; i <= ei; i++)
          {
            int fi   = (i - pmb->cis) * 2 + pmb->is;
            Real Uxx = 0.0, Vyy = 0.0, Wzz = 0.0;
            Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
            for (int jj = 0; jj < 2; jj++)
            {
              int js = 2 * jj - 1, fjj = fj + jj, fjp = fj + 2 * jj;
              for (int ii = 0; ii < 2; ii++)
              {
                int is = 2 * ii - 1, fii = fi + ii, fip = fi + 2 * ii;
                Uxx +=
                  is *
                  (js * A2 *
                     (fine.x2f(fk, fjp, fii) + fine.x2f(fk + 1, fjp, fii)) +
                   A3 * (fine.x3f(fk + 2, fjj, fii) - fine.x3f(fk, fjj, fii)));
                Vyy +=
                  js *
                  (A3 * (fine.x3f(fk + 2, fjj, fii) - fine.x3f(fk, fjj, fii)) +
                   is * A1 *
                     (fine.x1f(fk, fjj, fip) + fine.x1f(fk + 1, fjj, fip)));
                Wzz +=
                  is * A1 *
                    (fine.x1f(fk + 1, fjj, fip) - fine.x1f(fk, fjj, fip)) +
                  js * A2 *
                    (fine.x2f(fk + 1, fjp, fii) - fine.x2f(fk, fjp, fii));
                Uxyz += is * js * A1 *
                        (fine.x1f(fk + 1, fjj, fip) - fine.x1f(fk, fjj, fip));
                Vxyz += is * js * A2 *
                        (fine.x2f(fk + 1, fjp, fii) - fine.x2f(fk, fjp, fii));
                Wxyz += is * js * A3 *
                        (fine.x3f(fk + 2, fjj, fii) - fine.x3f(fk, fjj, fii));
              }
            }
            Uxx *= 0.125;
            Vyy *= 0.125;
            Wzz *= 0.125;
            Uxyz *= oo_SH23;
            Vxyz *= oo_SH13;
            Wxyz *= oo_SH12;
            fine.x1f(fk, fj, fi + 1) =
              0.5 * (fine.x1f(fk, fj, fi) + fine.x1f(fk, fj, fi + 2)) +
              (Uxx - Sdx3 * Vxyz - Sdx2 * Wxyz) * oo_A1;
            fine.x1f(fk, fj + 1, fi + 1) =
              0.5 * (fine.x1f(fk, fj + 1, fi) + fine.x1f(fk, fj + 1, fi + 2)) +
              (Uxx - Sdx3 * Vxyz + Sdx2 * Wxyz) * oo_A1;
            fine.x1f(fk + 1, fj, fi + 1) =
              0.5 * (fine.x1f(fk + 1, fj, fi) + fine.x1f(fk + 1, fj, fi + 2)) +
              (Uxx + Sdx3 * Vxyz - Sdx2 * Wxyz) * oo_A1;
            fine.x1f(fk + 1, fj + 1, fi + 1) =
              0.5 * (fine.x1f(fk + 1, fj + 1, fi) +
                     fine.x1f(fk + 1, fj + 1, fi + 2)) +
              (Uxx + Sdx3 * Vxyz + Sdx2 * Wxyz) * oo_A1;

            fine.x2f(fk, fj + 1, fi) =
              0.5 * (fine.x2f(fk, fj, fi) + fine.x2f(fk, fj + 2, fi)) +
              (Vyy - Sdx3 * Uxyz - Sdx1 * Wxyz) * oo_A2;
            fine.x2f(fk, fj + 1, fi + 1) =
              0.5 * (fine.x2f(fk, fj, fi + 1) + fine.x2f(fk, fj + 2, fi + 1)) +
              (Vyy - Sdx3 * Uxyz + Sdx1 * Wxyz) * oo_A2;
            fine.x2f(fk + 1, fj + 1, fi) =
              0.5 * (fine.x2f(fk + 1, fj, fi) + fine.x2f(fk + 1, fj + 2, fi)) +
              (Vyy + Sdx3 * Uxyz - Sdx1 * Wxyz) * oo_A2;
            fine.x2f(fk + 1, fj + 1, fi + 1) =
              0.5 * (fine.x2f(fk + 1, fj, fi + 1) +
                     fine.x2f(fk + 1, fj + 2, fi + 1)) +
              (Vyy + Sdx3 * Uxyz + Sdx1 * Wxyz) * oo_A2;

            fine.x3f(fk + 1, fj, fi) =
              0.5 * (fine.x3f(fk + 2, fj, fi) + fine.x3f(fk, fj, fi)) +
              (Wzz - Sdx2 * Uxyz - Sdx1 * Vxyz) * oo_A3;
            fine.x3f(fk + 1, fj, fi + 1) =
              0.5 * (fine.x3f(fk + 2, fj, fi + 1) + fine.x3f(fk, fj, fi + 1)) +
              (Wzz - Sdx2 * Uxyz + Sdx1 * Vxyz) * oo_A3;
            fine.x3f(fk + 1, fj + 1, fi) =
              0.5 * (fine.x3f(fk + 2, fj + 1, fi) + fine.x3f(fk, fj + 1, fi)) +
              (Wzz + Sdx2 * Uxyz - Sdx1 * Vxyz) * oo_A3;
            fine.x3f(fk + 1, fj + 1, fi + 1) =
              0.5 * (fine.x3f(fk + 2, fj + 1, fi + 1) +
                     fine.x3f(fk, fj + 1, fi + 1)) +
              (Wzz + Sdx2 * Uxyz + Sdx1 * Vxyz) * oo_A3;
          }
        }
      }
    }
    else
    {
      for (int k = sk; k <= ek; k++)
      {
        int fk = (k - pmb->cks) * 2 + pmb->ks;
        for (int j = sj; j <= ej; j++)
        {
          int fj = (j - pmb->cjs) * 2 + pmb->js;
          pco->Face1Area(fk, fj, fsi, fei + 1, sarea_x1_[0][0]);
          pco->Face1Area(fk, fj + 1, fsi, fei + 1, sarea_x1_[0][1]);
          pco->Face1Area(fk + 1, fj, fsi, fei + 1, sarea_x1_[1][0]);
          pco->Face1Area(fk + 1, fj + 1, fsi, fei + 1, sarea_x1_[1][1]);
          pco->Face2Area(fk, fj, fsi, fei, sarea_x2_[0][0]);
          pco->Face2Area(fk, fj + 1, fsi, fei, sarea_x2_[0][1]);
          pco->Face2Area(fk, fj + 2, fsi, fei, sarea_x2_[0][2]);
          pco->Face2Area(fk + 1, fj, fsi, fei, sarea_x2_[1][0]);
          pco->Face2Area(fk + 1, fj + 1, fsi, fei, sarea_x2_[1][1]);
          pco->Face2Area(fk + 1, fj + 2, fsi, fei, sarea_x2_[1][2]);
          pco->Face3Area(fk, fj, fsi, fei, sarea_x3_[0][0]);
          pco->Face3Area(fk, fj + 1, fsi, fei, sarea_x3_[0][1]);
          pco->Face3Area(fk + 1, fj, fsi, fei, sarea_x3_[1][0]);
          pco->Face3Area(fk + 1, fj + 1, fsi, fei, sarea_x3_[1][1]);
          pco->Face3Area(fk + 2, fj, fsi, fei, sarea_x3_[2][0]);
          pco->Face3Area(fk + 2, fj + 1, fsi, fei, sarea_x3_[2][1]);
          for (int i = si; i <= ei; i++)
          {
            int fi   = (i - pmb->cis) * 2 + pmb->is;
            Real Uxx = 0.0, Vyy = 0.0, Wzz = 0.0;
            Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
            for (int jj = 0; jj < 2; jj++)
            {
              int js = 2 * jj - 1, fjj = fj + jj, fjp = fj + 2 * jj;
              for (int ii = 0; ii < 2; ii++)
              {
                int is = 2 * ii - 1, fii = fi + ii, fip = fi + 2 * ii;
                Uxx +=
                  is *
                  (js *
                     (fine.x2f(fk, fjp, fii) * sarea_x2_[0][2 * jj](fii) +
                      fine.x2f(fk + 1, fjp, fii) * sarea_x2_[1][2 * jj](fii)) +
                   (fine.x3f(fk + 2, fjj, fii) * sarea_x3_[2][jj](fii) -
                    fine.x3f(fk, fjj, fii) * sarea_x3_[0][jj](fii)));
                Vyy +=
                  js *
                  ((fine.x3f(fk + 2, fjj, fii) * sarea_x3_[2][jj](fii) -
                    fine.x3f(fk, fjj, fii) * sarea_x3_[0][jj](fii)) +
                   is * (fine.x1f(fk, fjj, fip) * sarea_x1_[0][jj](fip) +
                         fine.x1f(fk + 1, fjj, fip) * sarea_x1_[1][jj](fip)));
                Wzz +=
                  is * (fine.x1f(fk + 1, fjj, fip) * sarea_x1_[1][jj](fip) -
                        fine.x1f(fk, fjj, fip) * sarea_x1_[0][jj](fip)) +
                  js *
                    (fine.x2f(fk + 1, fjp, fii) * sarea_x2_[1][2 * jj](fii) -
                     fine.x2f(fk, fjp, fii) * sarea_x2_[0][2 * jj](fii));
                Uxyz += is * js *
                        (fine.x1f(fk + 1, fjj, fip) * sarea_x1_[1][jj](fip) -
                         fine.x1f(fk, fjj, fip) * sarea_x1_[0][jj](fip));
                Vxyz +=
                  is * js *
                  (fine.x2f(fk + 1, fjp, fii) * sarea_x2_[1][2 * jj](fii) -
                   fine.x2f(fk, fjp, fii) * sarea_x2_[0][2 * jj](fii));
                Wxyz += is * js *
                        (fine.x3f(fk + 2, fjj, fii) * sarea_x3_[2][jj](fii) -
                         fine.x3f(fk, fjj, fii) * sarea_x3_[0][jj](fii));
              }
            }
            Real Sdx1 = SQR(pco->dx1f(fi) + pco->dx1f(fi + 1));
            Real Sdx2 = SQR(pco->GetEdge2Length(fk + 1, fj, fi + 1) +
                            pco->GetEdge2Length(fk + 1, fj + 1, fi + 1));
            Real Sdx3 = SQR(pco->GetEdge3Length(fk, fj + 1, fi + 1) +
                            pco->GetEdge3Length(fk + 1, fj + 1, fi + 1));
            Uxx *= 0.125;
            Vyy *= 0.125;
            Wzz *= 0.125;
            Uxyz *= 0.125 / (Sdx2 + Sdx3);
            Vxyz *= 0.125 / (Sdx1 + Sdx3);
            Wxyz *= 0.125 / (Sdx1 + Sdx2);
            fine.x1f(fk, fj, fi + 1) =
              (0.5 * (fine.x1f(fk, fj, fi) * sarea_x1_[0][0](fi) +
                      fine.x1f(fk, fj, fi + 2) * sarea_x1_[0][0](fi + 2)) +
               Uxx - Sdx3 * Vxyz - Sdx2 * Wxyz) /
              sarea_x1_[0][0](fi + 1);
            fine.x1f(fk, fj + 1, fi + 1) =
              (0.5 * (fine.x1f(fk, fj + 1, fi) * sarea_x1_[0][1](fi) +
                      fine.x1f(fk, fj + 1, fi + 2) * sarea_x1_[0][1](fi + 2)) +
               Uxx - Sdx3 * Vxyz + Sdx2 * Wxyz) /
              sarea_x1_[0][1](fi + 1);
            fine.x1f(fk + 1, fj, fi + 1) =
              (0.5 * (fine.x1f(fk + 1, fj, fi) * sarea_x1_[1][0](fi) +
                      fine.x1f(fk + 1, fj, fi + 2) * sarea_x1_[1][0](fi + 2)) +
               Uxx + Sdx3 * Vxyz - Sdx2 * Wxyz) /
              sarea_x1_[1][0](fi + 1);
            fine.x1f(fk + 1, fj + 1, fi + 1) =
              (0.5 *
                 (fine.x1f(fk + 1, fj + 1, fi) * sarea_x1_[1][1](fi) +
                  fine.x1f(fk + 1, fj + 1, fi + 2) * sarea_x1_[1][1](fi + 2)) +
               Uxx + Sdx3 * Vxyz + Sdx2 * Wxyz) /
              sarea_x1_[1][1](fi + 1);

            fine.x2f(fk, fj + 1, fi) =
              (0.5 * (fine.x2f(fk, fj, fi) * sarea_x2_[0][0](fi) +
                      fine.x2f(fk, fj + 2, fi) * sarea_x2_[0][2](fi)) +
               Vyy - Sdx3 * Uxyz - Sdx1 * Wxyz) /
              sarea_x2_[0][1](fi);
            fine.x2f(fk, fj + 1, fi + 1) =
              (0.5 * (fine.x2f(fk, fj, fi + 1) * sarea_x2_[0][0](fi + 1) +
                      fine.x2f(fk, fj + 2, fi + 1) * sarea_x2_[0][2](fi + 1)) +
               Vyy - Sdx3 * Uxyz + Sdx1 * Wxyz) /
              sarea_x2_[0][1](fi + 1);
            fine.x2f(fk + 1, fj + 1, fi) =
              (0.5 * (fine.x2f(fk + 1, fj, fi) * sarea_x2_[1][0](fi) +
                      fine.x2f(fk + 1, fj + 2, fi) * sarea_x2_[1][2](fi)) +
               Vyy + Sdx3 * Uxyz - Sdx1 * Wxyz) /
              sarea_x2_[1][1](fi);
            fine.x2f(fk + 1, fj + 1, fi + 1) =
              (0.5 *
                 (fine.x2f(fk + 1, fj, fi + 1) * sarea_x2_[1][0](fi + 1) +
                  fine.x2f(fk + 1, fj + 2, fi + 1) * sarea_x2_[1][2](fi + 1)) +
               Vyy + Sdx3 * Uxyz + Sdx1 * Wxyz) /
              sarea_x2_[1][1](fi + 1);

            fine.x3f(fk + 1, fj, fi) =
              (0.5 * (fine.x3f(fk + 2, fj, fi) * sarea_x3_[2][0](fi) +
                      fine.x3f(fk, fj, fi) * sarea_x3_[0][0](fi)) +
               Wzz - Sdx2 * Uxyz - Sdx1 * Vxyz) /
              sarea_x3_[1][0](fi);
            fine.x3f(fk + 1, fj, fi + 1) =
              (0.5 * (fine.x3f(fk + 2, fj, fi + 1) * sarea_x3_[2][0](fi + 1) +
                      fine.x3f(fk, fj, fi + 1) * sarea_x3_[0][0](fi + 1)) +
               Wzz - Sdx2 * Uxyz + Sdx1 * Vxyz) /
              sarea_x3_[1][0](fi + 1);
            fine.x3f(fk + 1, fj + 1, fi) =
              (0.5 * (fine.x3f(fk + 2, fj + 1, fi) * sarea_x3_[2][1](fi) +
                      fine.x3f(fk, fj + 1, fi) * sarea_x3_[0][1](fi)) +
               Wzz + Sdx2 * Uxyz - Sdx1 * Vxyz) /
              sarea_x3_[1][1](fi);
            fine.x3f(fk + 1, fj + 1, fi + 1) =
              (0.5 *
                 (fine.x3f(fk + 2, fj + 1, fi + 1) * sarea_x3_[2][1](fi + 1) +
                  fine.x3f(fk, fj + 1, fi + 1) * sarea_x3_[0][1](fi + 1)) +
               Wzz + Sdx2 * Uxyz + Sdx1 * Vxyz) /
              sarea_x3_[1][1](fi + 1);
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    if (uniform_cart_)
    {
      const Real q_A2_o_A1 = 0.25 * uc_h1_ / uc_h2_;
      const Real q_A1_o_A2 = 0.25 * uc_h2_ / uc_h1_;
      int fk               = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        for (int i = si; i <= ei; i++)
        {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real tmp1 =
            q_A2_o_A1 *
            (fine.x2f(fk, fj + 2, fi + 1) - fine.x2f(fk, fj, fi + 1) -
             fine.x2f(fk, fj + 2, fi) + fine.x2f(fk, fj, fi));
          Real tmp2 =
            q_A1_o_A2 *
            (fine.x1f(fk, fj, fi) - fine.x1f(fk, fj, fi + 2) -
             fine.x1f(fk, fj + 1, fi) + fine.x1f(fk, fj + 1, fi + 2));
          fine.x1f(fk, fj, fi + 1) =
            0.5 * (fine.x1f(fk, fj, fi) + fine.x1f(fk, fj, fi + 2)) + tmp1;
          fine.x1f(fk, fj + 1, fi + 1) =
            0.5 * (fine.x1f(fk, fj + 1, fi) + fine.x1f(fk, fj + 1, fi + 2)) +
            tmp1;
          fine.x2f(fk, fj + 1, fi) =
            0.5 * (fine.x2f(fk, fj, fi) + fine.x2f(fk, fj + 2, fi)) + tmp2;
          fine.x2f(fk, fj + 1, fi + 1) =
            0.5 * (fine.x2f(fk, fj, fi + 1) + fine.x2f(fk, fj + 2, fi + 1)) +
            tmp2;
        }
      }
    }
    else
    {
      int fk = pmb->ks;
      for (int j = sj; j <= ej; j++)
      {
        int fj = (j - pmb->cjs) * 2 + pmb->js;
        pco->Face1Area(fk, fj, fsi, fei + 1, sarea_x1_[0][0]);
        pco->Face1Area(fk, fj + 1, fsi, fei + 1, sarea_x1_[0][1]);
        pco->Face2Area(fk, fj, fsi, fei, sarea_x2_[0][0]);
        pco->Face2Area(fk, fj + 1, fsi, fei, sarea_x2_[0][1]);
        pco->Face2Area(fk, fj + 2, fsi, fei, sarea_x2_[0][2]);
        for (int i = si; i <= ei; i++)
        {
          int fi = (i - pmb->cis) * 2 + pmb->is;
          Real tmp1 =
            0.25 * (fine.x2f(fk, fj + 2, fi + 1) * sarea_x2_[0][2](fi + 1) -
                    fine.x2f(fk, fj, fi + 1) * sarea_x2_[0][0](fi + 1) -
                    fine.x2f(fk, fj + 2, fi) * sarea_x2_[0][2](fi) +
                    fine.x2f(fk, fj, fi) * sarea_x2_[0][0](fi));
          Real tmp2 =
            0.25 * (fine.x1f(fk, fj, fi) * sarea_x1_[0][0](fi) -
                    fine.x1f(fk, fj, fi + 2) * sarea_x1_[0][0](fi + 2) -
                    fine.x1f(fk, fj + 1, fi) * sarea_x1_[0][1](fi) +
                    fine.x1f(fk, fj + 1, fi + 2) * sarea_x1_[0][1](fi + 2));
          fine.x1f(fk, fj, fi + 1) =
            (0.5 * (fine.x1f(fk, fj, fi) * sarea_x1_[0][0](fi) +
                    fine.x1f(fk, fj, fi + 2) * sarea_x1_[0][0](fi + 2)) +
             tmp1) /
            sarea_x1_[0][0](fi + 1);
          fine.x1f(fk, fj + 1, fi + 1) =
            (0.5 * (fine.x1f(fk, fj + 1, fi) * sarea_x1_[0][1](fi) +
                    fine.x1f(fk, fj + 1, fi + 2) * sarea_x1_[0][1](fi + 2)) +
             tmp1) /
            sarea_x1_[0][1](fi + 1);
          fine.x2f(fk, fj + 1, fi) =
            (0.5 * (fine.x2f(fk, fj, fi) * sarea_x2_[0][0](fi) +
                    fine.x2f(fk, fj + 2, fi) * sarea_x2_[0][2](fi)) +
             tmp2) /
            sarea_x2_[0][1](fi);
          fine.x2f(fk, fj + 1, fi + 1) =
            (0.5 * (fine.x2f(fk, fj, fi + 1) * sarea_x2_[0][0](fi + 1) +
                    fine.x2f(fk, fj + 2, fi + 1) * sarea_x2_[0][2](fi + 1)) +
             tmp2) /
            sarea_x2_[0][1](fi + 1);
        }
      }
    }
  }
  else
  {
    if (uniform_cart_)
    {
      for (int i = si; i <= ei; i++)
      {
        int fi                 = (i - pmb->cis) * 2 + pmb->is;
        fine.x1f(0, 0, fi + 1) = fine.x1f(0, 0, fi);
      }
    }
    else
    {
      pco->Face1Area(0, 0, fsi, fei + 1, sarea_x1_[0][0]);
      for (int i = si; i <= ei; i++)
      {
        int fi                 = (i - pmb->cis) * 2 + pmb->is;
        Real ph                = sarea_x1_[0][0](fi) * fine.x1f(0, 0, fi);
        fine.x1f(0, 0, fi + 1) = ph / sarea_x1_[0][0](fi + 1);
      }
    }
  }
  return;
}
