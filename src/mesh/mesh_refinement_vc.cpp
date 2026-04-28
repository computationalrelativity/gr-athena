//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_refinement_vc.cpp
//  \brief Vertex-centered restrict and prolongate operators for mesh
//  refinement

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../utils/floating_point.hpp"
#include "../utils/interp_univariate.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

//----------------------------------------------------------------------------------------
//! \fn inline void MeshRefinement::RestrictVertexCenteredIndicialHelper(...)
//  \brief De-duplicate some indicial logic
inline void MeshRefinement::RestrictVertexCenteredIndicialHelper(int ix,
                                                                 int ix_cvs,
                                                                 int ix_cve,
                                                                 int ix_vs,
                                                                 int ix_ve,
                                                                 int& f_ix)
{
  // map for fine-index
  if (ix < ix_cvs)
  {
    f_ix = ix_vs - 2 * (ix_cvs - ix);
  }
  else if (ix > ix_cve)
  {
    f_ix = ix_ve + 2 * (ix - ix_cve);
  }
  else
  {  // map to interior+boundary nodes
    f_ix = 2 * (ix - ix_cvs) + ix_vs;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictVertexCenteredValues(const
//! AthenaArray<Real> &fine,
//                           AthenaArray<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int
//                           cek)
//  \brief restrict vertex centered values

void MeshRefinement::RestrictVertexCenteredValues(
  const AthenaArray<Real>& fine,
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
  MeshBlock* pmb = pmy_block_;

  // store the restricted data in the prolongation buffer for later use
  if (pmb->block_size.nx3 > 1)
  {
    int k, j, i;
    for (int n = sn; n <= en; ++n)
    {
      for (int ck = csk; ck <= cek; ck++)
      {
        // int k = (ck - pmb->ckvs)*2 + pmb->kvs;
        RestrictVertexCenteredIndicialHelper(
          ck, pmb->ckvs, pmb->ckve, pmb->kvs, pmb->kve, k);
        for (int cj = csj; cj <= cej; cj++)
        {
          // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
          RestrictVertexCenteredIndicialHelper(
            cj, pmb->cjvs, pmb->cjve, pmb->jvs, pmb->jve, j);
          for (int ci = csi; ci <= cei; ci++)
          {
            // int i = (ci - pmb->civs)*2 + pmb->ivs;
            RestrictVertexCenteredIndicialHelper(
              ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
            coarse(n, ck, cj, ci) = fine(n, k, j, i);
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    int k = pmb->kvs, ck = pmb->ckvs;
    int j, i;

    for (int n = sn; n <= en; ++n)
    {
      for (int cj = csj; cj <= cej; cj++)
      {
        // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
        RestrictVertexCenteredIndicialHelper(
          cj, pmb->cjvs, pmb->cjve, pmb->jvs, pmb->jve, j);
        for (int ci = csi; ci <= cei; ci++)
        {
          // int i = (ci - pmb->civs)*2 + pmb->ivs;
          RestrictVertexCenteredIndicialHelper(
            ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
          coarse(n, ck, cj, ci) = fine(n, k, j, i);
        }
      }
    }
  }
  else
  {
    int j = pmb->jvs, cj = pmb->cjvs, k = pmb->kvs, ck = pmb->ckvs;
    int i;
    for (int n = sn; n <= en; ++n)
    {
      for (int ci = csi; ci <= cei; ci++)
      {
        // int i = (ci - pmb->civs)*2 + pmb->ivs;
        RestrictVertexCenteredIndicialHelper(
          ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
        coarse(n, ck, cj, ci) = fine(n, k, j, i);
      }
    }
  }
}

void MeshRefinement::RestrictTwiceToBufferVertexCenteredValues(
  const AthenaArray<Real>& fine,
  Real* buf,
  int sn,
  int en,
  int csi,
  int cei,
  int csj,
  int cej,
  int csk,
  int cek,
  int& offset)
{
  MeshBlock* pmb = pmy_block_;
  // Coordinates *pco = pmb->pcoord;

  // store the restricted data within input buffer
  if (pmb->block_size.nx3 > 1)
  {  // 3D
    int k, j, i;
    for (int n = sn; n <= en; ++n)
    {
      for (int ck = csk; ck <= cek; ck += 2)
      {
        // int k = (ck - pmb->ckvs)*2 + pmb->kvs;
        RestrictVertexCenteredIndicialHelper(
          ck, pmb->ckvs, pmb->ckve, pmb->kvs, pmb->kve, k);
        for (int cj = csj; cj <= cej; cj += 2)
        {
          // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
          RestrictVertexCenteredIndicialHelper(
            cj, pmb->cjvs, pmb->cjve, pmb->jvs, pmb->jve, j);
          for (int ci = csi; ci <= cei; ci += 2)
          {
            // int i = (ci - pmb->civs)*2 + pmb->ivs;
            RestrictVertexCenteredIndicialHelper(
              ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
            buf[offset++] = fine(n, k, j, i);
          }
        }
      }
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    int k = pmb->kvs;
    int j, i;
    for (int n = sn; n <= en; ++n)
    {
      for (int cj = csj; cj <= cej; cj += 2)
      {
        // int j = (cj - pmb->cjvs)*2 + pmb->jvs;
        RestrictVertexCenteredIndicialHelper(
          cj, pmb->cjvs, pmb->cjve, pmb->jvs, pmb->jve, j);
        for (int ci = csi; ci <= cei; ci += 2)
        {
          // int i = (ci - pmb->civs)*2 + pmb->ivs;
          RestrictVertexCenteredIndicialHelper(
            ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
          buf[offset++] = fine(n, k, j, i);
        }
      }
    }
  }
  else
  {  // 1D
    int j = pmb->jvs, k = pmb->kvs;
    int i;
    for (int n = sn; n <= en; ++n)
    {
      for (int ci = csi; ci <= cei; ci += 2)
      {
        // int i = (ci - pmb->civs)*2 + pmb->ivs;
        RestrictVertexCenteredIndicialHelper(
          ci, pmb->civs, pmb->cive, pmb->ivs, pmb->ive, i);
        buf[offset++] = fine(n, k, j, i);
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn inline void MeshRefinement::ProlongateVertexCenteredIndicialHelper(...)
//  \brief De-duplicate some indicial logic
inline void MeshRefinement::ProlongateVertexCenteredIndicialHelper(int hs_sz,
                                                                   int ix,
                                                                   int ix_cvs,
                                                                   int ix_cve,
                                                                   int ix_cmp,
                                                                   int ix_vs,
                                                                   int ix_ve,
                                                                   int& f_ix,
                                                                   int& ix_b,
                                                                   int& ix_so,
                                                                   int& ix_eo,
                                                                   int& ix_l,
                                                                   int& ix_u)
{
  // map for fine-index
  if (ix < ix_cvs)
  {
    f_ix = ix_vs - 2 * (ix_cvs - ix);
  }
  else if (ix > ix_cve)
  {
    f_ix = ix_ve + 2 * (ix - ix_cve);
  }
  else
  {  // map to interior+boundary nodes
    f_ix = 2 * (ix - ix_cvs) + ix_vs;
  }

  // bias direction [nb. stencil still symmetric!]
  if (ix < ix_cmp)
  {
    ix_b  = 1;
    ix_so = 0;
    ix_eo = 1;
  }
  else if (ix > ix_cmp)
  {
    ix_b  = -1;
    ix_so = -1;
    ix_eo = 0;
  }
  else
  {
    // central node is unbiased, coincident, inject with no neighbors
    ix_so = ix_eo = 0;
    ix_b          = -1;
  }

  ix_l = ix - hs_sz + 1 - (1 - ix_b) / 2;
  ix_u = ix + hs_sz - (1 - ix_b) / 2;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateVertexCenteredValues(
//        const AthenaArray<Real> &coarse,AthenaArray<Real> &fine, int sn, int
//        en,, int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate vertex centered values;
//  Faster implementation, bias towards center by default

void MeshRefinement::ProlongateVertexCenteredValues(
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
  using InterpUniform::InterpolateLagrangeUniform;

  // // half number of ghosts
  // int const H_NCGHOST = NCGHOST / 2;

  // // maximum stencil size for interpolator
  // int const H_SZ = H_NCGHOST + 1;

  // ghost shift parameter
  int const H_NCGHOST = (2 * NCGHOST - NGHOST) / 2;

  // maximum stencil size for interpolator
  int const H_SZ = (2 * NCGHOST - NGHOST) / 2 + 1;

  MeshBlock* pmb = pmy_block_;

  if (pmb->pmy_mesh->ndim == 3)
  {
#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{ (si) };
    int const ei_inj{ (ei) };

    int const sj_inj{ (sj) };
    int const ej_inj{ (ej) };

    int const sk_inj{ (sk) };
    int const ek_inj{ (ek) };
#else
    int const eo_offset = -1;

    int const fis_inj{ (2 * (si - H_NCGHOST) + eo_offset) };
    int const fie_inj{ (2 * (ei - H_NCGHOST) + eo_offset) };

    int const fjs_inj{ (2 * (sj - H_NCGHOST) + eo_offset) };
    int const fje_inj{ (2 * (ej - H_NCGHOST) + eo_offset) };

    int const fks_inj{ (2 * (sk - H_NCGHOST) + eo_offset) };
    int const fke_inj{ (2 * (ek - H_NCGHOST) + eo_offset) };

    int const si_inj{ (fis_inj < 0) ? si + 1 : si };
    int const ei_inj{ (fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1
                                                                   : ei };

    int const sj_inj{ (fjs_inj < 0) ? sj + 1 : sj };
    int const ej_inj{ (fje_inj > 2 * NGHOST + pmb->block_size.nx2) ? ej - 1
                                                                   : ej };

    int const sk_inj{ (fks_inj < 0) ? sk + 1 : sk };
    int const ek_inj{ (fke_inj > 2 * NGHOST + pmb->block_size.nx3) ? ek - 1
                                                                   : ek };
#endif

    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{ (si > pmb->cimp) ? si - 1 : si };
    int const ei_prl{ (ei < pmb->cimp) ? ei + 1 : ei };
    int const sj_prl{ (sj > pmb->cjmp) ? sj - 1 : sj };
    int const ej_prl{ (ej < pmb->cjmp) ? ej + 1 : ej };
    int const sk_prl{ (sk > pmb->ckmp) ? sk - 1 : sk };
    int const ek_prl{ (ek < pmb->ckmp) ? ek + 1 : ek };

    // [running ix]: op

    for (int n = sn; n <= en; ++n)
    {
      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 3d
      for (int k = sk_prl; k < ek_prl; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_prl; j < ej_prl; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_prl; i < ei_prl; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_prl, fj_prl, fi_prl) = 0.;

            for (int dk = 0; dk < H_SZ; ++dk)
            {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dk - 1];

              for (int dj = 0; dj < H_SZ; ++dj)
              {
                int const cj_u = j + dj + 1;
                int const cj_l = j - dj;

                Real const lckj =
                  lck * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

                for (int di = 0; di < H_SZ; ++di)
                {
                  int const ci_u = i + di + 1;
                  int const ci_l = i - di;

                  Real const lckji =
                    lckj *
                    InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

                  Real const fc_uuu = coarse(n, ck_u, cj_u, ci_u);
                  Real const fc_lll = coarse(n, ck_l, cj_l, ci_l);

                  Real const fc_luu = coarse(n, ck_l, cj_u, ci_u);
                  Real const fc_ulu = coarse(n, ck_u, cj_l, ci_u);
                  Real const fc_uul = coarse(n, ck_u, cj_u, ci_l);

                  Real const fc_llu = coarse(n, ck_l, cj_l, ci_u);
                  Real const fc_ull = coarse(n, ck_u, cj_l, ci_l);
                  Real const fc_lul = coarse(n, ck_l, cj_u, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
#ifdef DBG_SYMMETRIZE_CHEAP
                  fine(n, fk_prl, fj_prl, fi_prl) +=
                    lckji * (FloatingPoint::sum_corners(fc_uuu,
                                                        fc_lll,
                                                        fc_uul,
                                                        fc_llu,
                                                        fc_lul,
                                                        fc_ulu,
                                                        fc_luu,
                                                        fc_ull));
#else
                  fine(n, fk_prl, fj_prl, fi_prl) +=
                    lckji * (FloatingPoint::sum_associative(fc_uuu,
                                                            fc_lll,
                                                            fc_uul,
                                                            fc_llu,
                                                            fc_lul,
                                                            fc_ulu,
                                                            fc_luu,
                                                            fc_ull));
#endif  // DBG_SYMMETRIZE_CHEAP
#else
                  fine(n, fk_prl, fj_prl, fi_prl) +=
                    lckji * (((fc_uuu + fc_lll) + (fc_uul + fc_llu)) +
                             ((fc_lul + fc_ulu) + (fc_luu + fc_ull)));
#endif  // DBG_SYMMETRIZE_P_OP
                }
              }
            }
          }
        }
      }  // (prl, prl, prl)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_inj; k <= ek_inj; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_prl; j < ej_prl; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_prl; i < ei_prl; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_inj, fj_prl, fi_prl) = 0.;

            for (int dj = 0; dj < H_SZ; ++dj)
            {
              int const cj_u = j + dj + 1;
              int const cj_l = j - dj;

              Real const lcj =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

              for (int di = 0; di < H_SZ; ++di)
              {
                int const ci_u = i + di + 1;
                int const ci_l = i - di;

                Real const lcji =
                  lcj * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

                Real const fc_cuu = coarse(n, k, cj_u, ci_u);
                Real const fc_cul = coarse(n, k, cj_u, ci_l);
                Real const fc_clu = coarse(n, k, cj_l, ci_u);
                Real const fc_cll = coarse(n, k, cj_l, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
#ifdef DBG_SYMMETRIZE_CHEAP
                fine(n, fk_inj, fj_prl, fi_prl) +=
                  lcji *
                  (FloatingPoint::sum_corners(fc_cuu, fc_cll, fc_clu, fc_cul));
#else
                fine(n, fk_inj, fj_prl, fi_prl) +=
                  lcji * (FloatingPoint::sum_associative(
                           fc_cuu, fc_cll, fc_clu, fc_cul));
#endif  // DBG_SYMMETRIZE_CHEAP
#else
                fine(n, fk_inj, fj_prl, fi_prl) +=
                  lcji * ((fc_cuu + fc_cll) + (fc_clu + fc_cul));
#endif  // DBG_SYMMETRIZE_P_OP
              }
            }
          }
        }
      }  // (inj, prl, prl)

      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_prl; k < ek_prl; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_inj; j <= ej_inj; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_prl; i < ei_prl; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_prl, fj_inj, fi_prl) = 0.;

            for (int dk = 0; dk < H_SZ; ++dk)
            {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dk - 1];

              for (int di = 0; di < H_SZ; ++di)
              {
                int const ci_u = i + di + 1;
                int const ci_l = i - di;

                Real const lcki =
                  lck * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

                Real const fc_ucu = coarse(n, ck_u, j, ci_u);
                Real const fc_ucl = coarse(n, ck_u, j, ci_l);
                Real const fc_lcu = coarse(n, ck_l, j, ci_u);
                Real const fc_lcl = coarse(n, ck_l, j, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
#ifdef DBG_SYMMETRIZE_CHEAP
                fine(n, fk_prl, fj_inj, fi_prl) +=
                  lcki *
                  (FloatingPoint::sum_corners(fc_lcu, fc_ucl, fc_ucu, fc_lcl));
#else
                fine(n, fk_prl, fj_inj, fi_prl) +=
                  lcki * (FloatingPoint::sum_associative(
                           fc_lcu, fc_ucl, fc_ucu, fc_lcl));
#endif  // DBG_SYMMETRIZE_CHEAP

#else
                fine(n, fk_prl, fj_inj, fi_prl) +=
                  lcki * ((fc_lcu + fc_ucl) + (fc_ucu + fc_lcl));
#endif  // DBG_SYMMETRIZE_P_OP
              }
            }
          }
        }
      }  // (prl, inj, prl)

      // [k, j, i]: interp. 2d & inject 1d
      for (int k = sk_prl; k < ek_prl; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_prl; j < ej_prl; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_inj; i <= ei_inj; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_prl, fj_prl, fi_inj) = 0.;

            for (int dk = 0; dk < H_SZ; ++dk)
            {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dk - 1];

              for (int dj = 0; dj < H_SZ; ++dj)
              {
                int const cj_u = j + dj + 1;
                int const cj_l = j - dj;

                Real const lckj =
                  lck * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

                Real const fc_uuc = coarse(n, ck_u, cj_u, i);
                Real const fc_ulc = coarse(n, ck_u, cj_l, i);
                Real const fc_luc = coarse(n, ck_l, cj_u, i);
                Real const fc_llc = coarse(n, ck_l, cj_l, i);

#ifdef DBG_SYMMETRIZE_P_OP
#ifdef DBG_SYMMETRIZE_CHEAP
                fine(n, fk_prl, fj_prl, fi_inj) +=
                  lckj *
                  (FloatingPoint::sum_corners(fc_uuc, fc_llc, fc_luc, fc_ulc));
#else
                fine(n, fk_prl, fj_prl, fi_inj) +=
                  lckj * (FloatingPoint::sum_associative(
                           fc_uuc, fc_llc, fc_luc, fc_ulc));
#endif  // DBG_SYMMETRIZE_CHEAP
#else
                fine(n, fk_prl, fj_prl, fi_inj) +=
                  lckj * ((fc_uuc + fc_llc) + (fc_luc + fc_ulc));
#endif  // DBG_SYMMETRIZE_P_OP
              }
            }
          }
        }
      }  // (prl, prl, inj)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_inj; k <= ek_inj; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_inj; j <= ej_inj; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_prl; i < ei_prl; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;

            fine(n, fk_inj, fj_inj, fi_prl) = 0.;

            for (int di = 0; di < H_SZ; ++di)
            {
              int const ci_u = i + di + 1;
              int const ci_l = i - di;

              Real const lci =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

              Real const fc_ccu = coarse(n, k, j, ci_u);
              Real const fc_ccl = coarse(n, k, j, ci_l);

              fine(n, fk_inj, fj_inj, fi_prl) += lci * (fc_ccl + fc_ccu);
            }
          }
        }
      }  // (inj, inj, prl)

      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_inj; k <= ek_inj; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_prl; j < ej_prl; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;

          for (int i = si_inj; i <= ei_inj; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_inj, fj_prl, fi_inj) = 0.;

            for (int dj = 0; dj < H_SZ; ++dj)
            {
              int const cj_u = j + dj + 1;
              int const cj_l = j - dj;

              Real const lcj =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

              Real const fc_cuc = coarse(n, k, cj_u, i);
              Real const fc_clc = coarse(n, k, cj_l, i);

              fine(n, fk_inj, fj_prl, fi_inj) += lcj * (fc_clc + fc_cuc);
            }
          }
        }
      }  // (inj, prl, inj)

      // [k, j, i]: interp. 1d & inject 2d
      for (int k = sk_prl; k < ek_prl; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;

        for (int j = sj_inj; j <= ej_inj; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_inj; i <= ei_inj; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_prl, fj_inj, fi_inj) = 0.;

            for (int dk = 0; dk < H_SZ; ++dk)
            {
              int const ck_u = k + dk + 1;
              int const ck_l = k - dk;

              Real const lck =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dk - 1];

              Real const fc_ucc = coarse(n, ck_u, j, i);
              Real const fc_lcc = coarse(n, ck_l, j, i);

              fine(n, fk_prl, fj_inj, fi_inj) += lck * (fc_lcc + fc_ucc);
            }
          }
        }
      }  // (prl, inj, inj)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [k, j, i]: inject 3d
      for (int k = sk_inj; k <= ek_inj; ++k)
      {
        int const fk_inj = 2 * (k - H_NCGHOST) + eo_offset;
        int const fk_prl = fk_inj + 1;
        (void)fk_prl;  // DR: why is this not used?

        for (int j = sj_inj; j <= ej_inj; ++j)
        {
          int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
          int const fj_prl = fj_inj + 1;
          (void)fj_prl;  // DR: why is this not used?

          for (int i = si_inj; i <= ei_inj; ++i)
          {
            int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
            int const fi_prl = fi_inj + 1;
            (void)fi_prl;  // DR: why is this not used?

            fine(n, fk_inj, fj_inj, fi_inj) = coarse(n, k, j, i);
          }
        }
      }  // (inj, inj, inj)
      //-----------------------------------------------------------------------

    }  // function component loop
  }
  else if (pmb->pmy_mesh->ndim == 2)
  {
#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{ (si) };
    int const sj_inj{ (sj) };

    int const ei_inj{ (ei) };
    int const ej_inj{ (ej) };
#else
    int const eo_offset = -1;

    int const fis_inj{ (2 * (si - H_NCGHOST) + eo_offset) };
    int const fie_inj{ (2 * (ei - H_NCGHOST) + eo_offset) };

    int const fjs_inj{ (2 * (sj - H_NCGHOST) + eo_offset) };
    int const fje_inj{ (2 * (ej - H_NCGHOST) + eo_offset) };

    int const si_inj{ (fis_inj < 0) ? si + 1 : si };
    int const ei_inj{ (fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1
                                                                   : ei };

    int const sj_inj{ (fjs_inj < 0) ? sj + 1 : sj };
    int const ej_inj{ (fje_inj > 2 * NGHOST + pmb->block_size.nx2) ? ej - 1
                                                                   : ej };
#endif

    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{ (si > pmb->cimp) ? si - 1 : si };
    int const ei_prl{ (ei < pmb->cimp) ? ei + 1 : ei };
    int const sj_prl{ (sj > pmb->cjmp) ? sj - 1 : sj };
    int const ej_prl{ (ej < pmb->cjmp) ? ej + 1 : ej };

    // [running ix]: op

    for (int n = sn; n <= en; ++n)
    {
      //-----------------------------------------------------------------------
      // [j, i]: interp. 2d
      for (int j = sj_prl; j < ej_prl; ++j)
      {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;

        for (int i = si_prl; i < ei_prl; ++i)
        {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;

          fine(n, 0, fj_prl, fi_prl) = 0.;

          // apply stencil via Cartesian product relation
          for (int dj = 0; dj < H_SZ; ++dj)
          {
            int const cj_u = j + dj + 1;
            int const cj_l = j - dj;

            Real const lcj =
              InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

            for (int di = 0; di < H_SZ; ++di)
            {
              int const ci_u = i + di + 1;
              int const ci_l = i - di;

              Real const lcji =
                lcj * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

              Real const fc_uu = coarse(n, 0, cj_u, ci_u);
              Real const fc_ul = coarse(n, 0, cj_u, ci_l);
              Real const fc_lu = coarse(n, 0, cj_l, ci_u);
              Real const fc_ll = coarse(n, 0, cj_l, ci_l);

#ifdef DBG_SYMMETRIZE_P_OP
#ifdef DBG_SYMMETRIZE_CHEAP
              fine(n, 0, fj_prl, fi_prl) +=
                lcji * FloatingPoint::sum_corners(fc_uu, fc_ll, fc_lu, fc_ul);
#else
              fine(n, 0, fj_prl, fi_prl) +=
                lcji *
                FloatingPoint::sum_associative(fc_uu, fc_ll, fc_lu, fc_ul);
#endif  // DBG_SYMMETRIZE_CHEAP
#else
              fine(n, 0, fj_prl, fi_prl) +=
                lcji * ((fc_uu + fc_ll) + (fc_lu + fc_ul));
#endif  // DBG_SYMMETRIZE_P_OP
            }
          }
        }
      }  // (prl, prl)
      //-----------------------------------------------------------------------

      //-----------------------------------------------------------------------
      // [j, i]: (interp. 1d, inject 1d)
      for (int j = sj_prl; j < ej_prl; ++j)
      {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;

        for (int i = si_inj; i <= ei_inj; ++i)
        {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;
          (void)fi_prl;  // DR: why is this not used?

          fine(n, 0, fj_prl, fi_inj) = 0.;

          for (int dj = 0; dj < H_SZ; ++dj)
          {
            int const cj_u = j + dj + 1;
            int const cj_l = j - dj;

            Real const lcj =
              InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

            Real const fc_uc = coarse(n, 0, cj_u, i);
            Real const fc_lc = coarse(n, 0, cj_l, i);

            fine(n, 0, fj_prl, fi_inj) += lcj * (fc_uc + fc_lc);
          }
        }
      }  // (prl, inj)

      // [j, i]: (inject 1d, interp. 1d)
      for (int j = sj_inj; j <= ej_inj; ++j)
      {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;
        (void)fj_prl;  // DR: why is this not used?

        for (int i = si_prl; i < ei_prl; ++i)
        {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;

          fine(n, 0, fj_inj, fi_prl) = 0.;

          for (int di = 0; di < H_SZ; ++di)
          {
            int const ci_u = i + di + 1;
            int const ci_l = i - di;

            Real const lci =
              InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

            Real const fc_cu = coarse(n, 0, j, ci_u);
            Real const fc_cl = coarse(n, 0, j, ci_l);

            fine(n, 0, fj_inj, fi_prl) += lci * (fc_cl + fc_cu);
          }
        }
      }  // (inj, prl)
      //-----------------------------------------------------------------------

      // [j, i]: inject 2d
      for (int j = sj_inj; j <= ej_inj; ++j)
      {
        int const fj_inj = 2 * (j - H_NCGHOST) + eo_offset;
        int const fj_prl = fj_inj + 1;
        (void)fj_prl;  // DR: why is this not used?

        for (int i = si_inj; i <= ei_inj; ++i)
        {
          int const fi_inj = 2 * (i - H_NCGHOST) + eo_offset;
          int const fi_prl = fi_inj + 1;
          (void)fi_prl;  // DR: why is this not used?

          // injected
          fine(n, 0, fj_inj, fi_inj) = coarse(n, 0, j, i);
        }
      }  // (inj, inj)
      //-----------------------------------------------------------------------

    }  // function component loop
    //-------------------------------------------------------------------------
  }
  else
  {
#if ISEVEN(NGHOST)
    int const eo_offset = 0;

    int const si_inj{ (si) };
    int const ei_inj{ (ei) };
#else
    int const eo_offset = -1;

    int const fis_inj{ (2 * (si - H_NCGHOST) + eo_offset) };
    int const fie_inj{ (2 * (ei - H_NCGHOST) + eo_offset) };

    int const si_inj{ (fis_inj < 0) ? si + 1 : si };
    int const ei_inj{ (fie_inj > 2 * NGHOST + pmb->block_size.nx1) ? ei - 1
                                                                   : ei };
#endif

    //-------------------------------------------------------------------------
    // bias offsets for prolongation (depends on location)
    int const si_prl{ (si > pmb->cimp) ? si - 1 : si };
    int const ei_prl{ (ei < pmb->cimp) ? ei + 1 : ei };

    for (int n = sn; n <= en; ++n)
    {
      for (int i = si_prl; i < ei_prl; ++i)
      {
        int const fi_inj      = 2 * (i - H_NCGHOST) + eo_offset;
        int const fi_prl      = fi_inj + 1;
        fine(n, 0, 0, fi_prl) = 0.;

        // apply stencil
        for (int di = 0; di < H_SZ; ++di)
        {
          int const ci_u = i + di + 1;
          int const ci_l = i - di;

          Real const lc =
            InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

          Real const fc_u = coarse(n, 0, 0, ci_u);
          Real const fc_l = coarse(n, 0, 0, ci_l);

          fine(n, 0, 0, fi_prl) += lc * (fc_l + fc_u);
        }
      }
      //-----------------------------------------------------------------------
    }  // function component loop

    // inject
    for (int n = sn; n <= en; ++n)
    {
      for (int i = si_inj; i <= ei_inj; ++i)
      {
        int const fi_inj      = 2 * (i - H_NCGHOST) + eo_offset;
        fine(n, 0, 0, fi_inj) = coarse(n, 0, 0, i);
      }
      //-----------------------------------------------------------------------
    }  // function component loop
  }

  return;
}
