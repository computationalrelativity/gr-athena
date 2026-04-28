//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file mesh_refinement_cx.cpp
//  \brief Cell-centered extended (CX) restrict and prolongate operators for
//         mesh refinement

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../utils/floating_point.hpp"
#include "../utils/interp_univariate.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

#if defined(DBG_CX_RESTRICT_TENSORPRODUCT) || \
  defined(DBG_CX_PROLONG_TENSORPRODUCT)

// ============================================================================
// Symmetrized tensor-product helpers for CX restriction / prolongation.
// Each helper performs a 3-pass (3D) or 2-pass (2D) 1D-sweep decomposition
// with a caller-specified axis ordering.  The main functions call the helper
// with every cyclic permutation of {0,1,2} (or {0,1} in 2D) and average the
// results, restoring the rotational symmetry that a single fixed ordering
// would break in floating-point arithmetic.
//
// Axis convention: dim 0 = i (stride 1 in the AthenaArray),
//                  dim 1 = j,  dim 2 = k.
// ============================================================================

// ----------------------------------------------------------------------------
// RestrictCX_TP_3D_ordered
//
// 3-pass restriction with a given axis ordering.
// axis_order[p] selects which spatial dimension is restricted in pass p.
//
// Output is written to 'result' in canonical [k][j][i] layout with
// result_strides = {1, nc[0], nc[0]*nc[1]}.
// ----------------------------------------------------------------------------
template <int H_SZ>
static void RestrictCX_TP_3D_ordered(const Real* fine_n,
                                     Real* result,
                                     const int fine_strides[3],
                                     const int axis_order[3],
                                     const int fs[3],
                                     const int nc[3],
                                     const int efs[3],
                                     const int nf_ext[3],
                                     Real* tmp1,
                                     Real* tmp2)
{
  using InterpUniform::InterpolateLagrangeUniform;
  const Real* rc = InterpolateLagrangeUniform<H_SZ>::coeff;

  const int a = axis_order[0], b = axis_order[1], c = axis_order[2];

  const int s_a = fine_strides[a];
  const int s_b = fine_strides[b];
  const int s_c = fine_strides[c];

  // tmp1 layout: [idx_c][idx_b][idx_a]  dims: nf_ext[c] x nf_ext[b] x nc[a]
  const int t1_sa = 1;
  const int t1_sb = nc[a];
  const int t1_sc = nf_ext[b] * nc[a];

  // tmp2 layout: [idx_c][idx_b][idx_a]  dims: nf_ext[c] x nc[b] x nc[a]
  const int t2_sa = 1;
  const int t2_sb = nc[a];
  const int t2_sc = nc[b] * nc[a];

  // result layout: canonical [k][j][i] = [idx_2][idx_1][idx_0]
  const int r_s0 = 1;
  const int r_s1 = nc[0];
  const int r_s2 = nc[0] * nc[1];
  int r_sa, r_sb, r_sc;
  {
    int r_strides[3] = { r_s0, r_s1, r_s2 };
    r_sa             = r_strides[a];
    r_sb             = r_strides[b];
    r_sc             = r_strides[c];
  }

  // ----- Pass 1: restrict along axis a (fine -> tmp1) ----------------------
  for (int ic = 0; ic < nf_ext[c]; ++ic)
  {
    const int val_c = efs[c] + ic;
    for (int ib = 0; ib < nf_ext[b]; ++ib)
    {
      const int val_b     = efs[b] + ib;
      const int t1_base   = ic * t1_sc + ib * t1_sb;
      const int fine_base = val_c * s_c + val_b * s_b;

      for (int ia = 0; ia < nc[a]; ++ia)
      {
        const int fi = fs[a] + 2 * ia;

        Real val = 0.0;
        for (int d = 0; d < H_SZ; ++d)
          val += rc[H_SZ - d - 1] * (fine_n[fine_base + (fi - d) * s_a] +
                                     fine_n[fine_base + (fi + d + 1) * s_a]);

        tmp1[t1_base + ia] = val;
      }
    }
  }

  // ----- Pass 2: restrict along axis b (tmp1 -> tmp2) ----------------------
  for (int ic = 0; ic < nf_ext[c]; ++ic)
  {
    for (int cb = 0; cb < nc[b]; ++cb)
    {
      const int fj  = fs[b] + 2 * cb;
      const int dst = ic * t2_sc + cb * t2_sb;

      for (int d = 0; d < H_SZ; ++d)
      {
        const Real w     = rc[H_SZ - d - 1];
        const int idx_bL = (fj - d) - efs[b];
        const int idx_bR = (fj + d + 1) - efs[b];
        const int srcL   = ic * t1_sc + idx_bL * t1_sb;
        const int srcR   = ic * t1_sc + idx_bR * t1_sb;

        if (d == 0)
        {
          _Pragma("omp simd") for (int ia = 0; ia < nc[a]; ++ia)
            tmp2[dst + ia] = w * (tmp1[srcL + ia] + tmp1[srcR + ia]);
        }
        else
        {
          _Pragma("omp simd") for (int ia = 0; ia < nc[a]; ++ia)
            tmp2[dst + ia] += w * (tmp1[srcL + ia] + tmp1[srcR + ia]);
        }
      }
    }
  }

  // ----- Pass 3: restrict along axis c (tmp2 -> result) --------------------
  for (int cc_ = 0; cc_ < nc[c]; ++cc_)
  {
    const int fk = fs[c] + 2 * cc_;

    for (int cb = 0; cb < nc[b]; ++cb)
    {
      const int r_base = cc_ * r_sc + cb * r_sb;

      for (int d = 0; d < H_SZ; ++d)
      {
        const Real w     = rc[H_SZ - d - 1];
        const int idx_cL = (fk - d) - efs[c];
        const int idx_cR = (fk + d + 1) - efs[c];
        const int srcL   = idx_cL * t2_sc + cb * t2_sb;
        const int srcR   = idx_cR * t2_sc + cb * t2_sb;

        if (d == 0)
        {
          for (int ia = 0; ia < nc[a]; ++ia)
            result[r_base + ia * r_sa] =
              w * (tmp2[srcL + ia] + tmp2[srcR + ia]);
        }
        else
        {
          for (int ia = 0; ia < nc[a]; ++ia)
            result[r_base + ia * r_sa] +=
              w * (tmp2[srcL + ia] + tmp2[srcR + ia]);
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// RestrictCX_TP_2D_ordered
//
// 2-pass restriction with a given axis ordering (2D, k trivial).
// axis_order[p] selects which of {0,1} is restricted in pass p.
// Output in canonical [j][i] layout.
// ----------------------------------------------------------------------------
template <int H_SZ>
static void RestrictCX_TP_2D_ordered(const Real* fine_n,
                                     Real* result,
                                     const int fine_strides[2],
                                     const int axis_order[2],
                                     const int fs[2],
                                     const int nc[2],
                                     const int efs[2],
                                     const int nf_ext[2],
                                     Real* tmp1)
{
  using InterpUniform::InterpolateLagrangeUniform;
  const Real* rc = InterpolateLagrangeUniform<H_SZ>::coeff;

  const int a = axis_order[0], b = axis_order[1];

  const int s_a = fine_strides[a];
  const int s_b = fine_strides[b];

  // tmp1 layout: [idx_b][idx_a]  dims: nf_ext[b] x nc[a]
  const int t1_sa = 1;
  const int t1_sb = nc[a];

  // result layout: canonical [j][i]
  const int r_s0 = 1;
  const int r_s1 = nc[0];
  int r_sa, r_sb;
  {
    int r_strides[2] = { r_s0, r_s1 };
    r_sa             = r_strides[a];
    r_sb             = r_strides[b];
  }

  // ----- Pass 1: restrict along axis a (fine -> tmp1) ----------------------
  for (int ib = 0; ib < nf_ext[b]; ++ib)
  {
    const int val_b   = efs[b] + ib;
    const int t1_base = ib * t1_sb;

    for (int ia = 0; ia < nc[a]; ++ia)
    {
      const int fi = fs[a] + 2 * ia;

      Real val = 0.0;
      for (int d = 0; d < H_SZ; ++d)
        val += rc[H_SZ - d - 1] * (fine_n[val_b * s_b + (fi - d) * s_a] +
                                   fine_n[val_b * s_b + (fi + d + 1) * s_a]);

      tmp1[t1_base + ia] = val;
    }
  }

  // ----- Pass 2: restrict along axis b (tmp1 -> result) --------------------
  for (int cb = 0; cb < nc[b]; ++cb)
  {
    const int fj     = fs[b] + 2 * cb;
    const int r_base = cb * r_sb;

    for (int d = 0; d < H_SZ; ++d)
    {
      const Real w     = rc[H_SZ - d - 1];
      const int idx_bL = (fj - d) - efs[b];
      const int idx_bR = (fj + d + 1) - efs[b];
      const int srcL   = idx_bL * t1_sb;
      const int srcR   = idx_bR * t1_sb;

      if (d == 0)
      {
        for (int ia = 0; ia < nc[a]; ++ia)
          result[r_base + ia * r_sa] = w * (tmp1[srcL + ia] + tmp1[srcR + ia]);
      }
      else
      {
        for (int ia = 0; ia < nc[a]; ++ia)
          result[r_base + ia * r_sa] +=
            w * (tmp1[srcL + ia] + tmp1[srcR + ia]);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// ProlongCX_TP_3D_ordered
//
// 3-pass prolongation with a given axis ordering.
// Output written to 'result' in canonical [fk][fj][fi] layout.
// ----------------------------------------------------------------------------
template <int H_SZ>
static void ProlongCX_TP_3D_ordered(const Real* coarse_n,
                                    Real* result,
                                    const int coarse_strides[3],
                                    const int axis_order[3],
                                    const int cs[3],
                                    const int nf[3],
                                    const bool f_isleft[3],
                                    Real* tmp1,
                                    Real* tmp2,
                                    const Real* coeff_R)
{
  using InterpUniform::InterpolateLagrangeUniformChildren;
  constexpr int S     = 2 * H_SZ + 1;
  const Real* coeff_L = InterpolateLagrangeUniformChildren<H_SZ>::coeff;

  const int a = axis_order[0], b = axis_order[1], c = axis_order[2];

  const int cs_a = coarse_strides[a];
  const int cs_b = coarse_strides[b];
  const int cs_c = coarse_strides[c];

  // Compute half-counts: nc[d] = (nf[d]+1)/2, but we derive from
  // the relationship: nf[d] = 2*nc_d or 2*nc_d+1 depending on parity.
  // The stencil for Pass p+1 reads coarse values [cs[d]-H_SZ ..
  // cs[d]+nc_d-1+H_SZ] relative to the coarse grid.  The "extended coarse"
  // range along each batch axis tracks these extra halo cells.
  //
  // For axis d processed in pass p, the # of output (fine) cells is nf[d].
  // For axis d NOT YET processed (batch), the needed coarse range is
  //   [cs[d] - H_SZ .. cs[d] + nc_d - 1 + H_SZ]  i.e. nc_d + 2*H_SZ = nc_d + S
  //   - 1.
  // nc_d for a batch axis: the number of distinct coarse cells that correspond
  // to nf[d] fine cells = (nf[d] + 1) / 2.  (integer division, works for
  // both even and odd nf[d].)

  const int nc_a = (nf[a] + 1) / 2;
  const int nc_b = (nf[b] + 1) / 2;
  const int nc_c = (nf[c] + 1) / 2;

  // Extended coarse counts (for batch axes of subsequent passes)
  const int ext_b = nc_b + S - 1;  // used by Pass 1 (batch for b)
  const int ext_c = nc_c + S - 1;  // used by Pass 1 & 2 (batch for c)

  // tmp1 layout: [idx_c_ext][idx_b_ext][idx_a_fine]
  //   dims: ext_c x ext_b x nf[a]
  const int t1_sa = 1;
  const int t1_sb = nf[a];
  const int t1_sc = ext_b * nf[a];

  // tmp2 layout: [idx_c_ext][idx_b_fine][idx_a_fine]
  //   dims: ext_c x nf[b] x nf[a]
  const int t2_sa = 1;
  const int t2_sb = nf[a];
  const int t2_sc = nf[b] * nf[a];

  // result layout: canonical [fk][fj][fi]
  const int r_s0 = 1;
  const int r_s1 = nf[0];
  const int r_s2 = nf[0] * nf[1];
  int r_sa, r_sb, r_sc;
  {
    int r_strides[3] = { r_s0, r_s1, r_s2 };
    r_sa             = r_strides[a];
    r_sb             = r_strides[b];
    r_sc             = r_strides[c];
  }

  // Extended coarse start indices (for batch axes)
  const int ecs_b = cs[b] - H_SZ;  // start of extended coarse range along b
  const int ecs_c = cs[c] - H_SZ;  // start of extended coarse range along c

  // ----- Pass 1: prolongate along axis a (coarse -> tmp1) ------------------
  for (int ic = 0; ic < ext_c; ++ic)
  {
    const int val_c = ecs_c + ic;
    for (int ib = 0; ib < ext_b; ++ib)
    {
      const int val_b       = ecs_b + ib;
      const int t1_base     = ic * t1_sc + ib * t1_sb;
      const int coarse_base = val_c * cs_c + val_b * cs_b;

      bool bl = f_isleft[a];
      int ci  = cs[a];

      for (int ia = 0; ia < nf[a]; ++ia)
      {
        const Real* w = bl ? coeff_L : coeff_R;

        Real val = 0.0;
        for (int s = 0; s < S; ++s)
          val += w[s] * coarse_n[coarse_base + (ci - H_SZ + s) * cs_a];

        tmp1[t1_base + ia] = val;

        bl = !bl;
        ci += bl;
      }
    }
  }

  // ----- Pass 2: prolongate along axis b (tmp1 -> tmp2) --------------------
  for (int ic = 0; ic < ext_c; ++ic)
  {
    bool bl = f_isleft[b];
    int cb  = cs[b];

    for (int ib = 0; ib < nf[b]; ++ib)
    {
      const Real* w = bl ? coeff_L : coeff_R;
      const int dst = ic * t2_sc + ib * t2_sb;

      for (int s = 0; s < S; ++s)
      {
        const Real coef = w[s];
        const int src_b = cb - H_SZ + s;
        const int src   = ic * t1_sc + (src_b - ecs_b) * t1_sb;

        if (s == 0)
        {
          _Pragma("omp simd") for (int ia = 0; ia < nf[a]; ++ia)
            tmp2[dst + ia] = coef * tmp1[src + ia];
        }
        else
        {
          _Pragma("omp simd") for (int ia = 0; ia < nf[a]; ++ia)
            tmp2[dst + ia] += coef * tmp1[src + ia];
        }
      }

      bl = !bl;
      cb += bl;
    }
  }

  // ----- Pass 3: prolongate along axis c (tmp2 -> result) ------------------
  {
    bool bl = f_isleft[c];
    int cc  = cs[c];

    for (int ic = 0; ic < nf[c]; ++ic)
    {
      const Real* w = bl ? coeff_L : coeff_R;

      for (int ib = 0; ib < nf[b]; ++ib)
      {
        const int r_base = ic * r_sc + ib * r_sb;

        for (int s = 0; s < S; ++s)
        {
          const Real coef = w[s];
          const int src_c = cc - H_SZ + s;
          const int src   = (src_c - ecs_c) * t2_sc + ib * t2_sb;

          if (s == 0)
          {
            for (int ia = 0; ia < nf[a]; ++ia)
              result[r_base + ia * r_sa] = coef * tmp2[src + ia];
          }
          else
          {
            for (int ia = 0; ia < nf[a]; ++ia)
              result[r_base + ia * r_sa] += coef * tmp2[src + ia];
          }
        }
      }

      bl = !bl;
      cc += bl;
    }
  }
}

// ----------------------------------------------------------------------------
// ProlongCX_TP_2D_ordered
//
// 2-pass prolongation with a given axis ordering (2D, k trivial).
// Output written to 'result' in canonical [fj][fi] layout.
// ----------------------------------------------------------------------------
template <int H_SZ>
static void ProlongCX_TP_2D_ordered(const Real* coarse_n,
                                    Real* result,
                                    const int coarse_strides[2],
                                    const int axis_order[2],
                                    const int cs[2],
                                    const int nf[2],
                                    const bool f_isleft[2],
                                    Real* tmp1,
                                    const Real* coeff_R)
{
  using InterpUniform::InterpolateLagrangeUniformChildren;
  constexpr int S     = 2 * H_SZ + 1;
  const Real* coeff_L = InterpolateLagrangeUniformChildren<H_SZ>::coeff;

  const int a = axis_order[0], b = axis_order[1];

  const int cs_a = coarse_strides[a];
  const int cs_b = coarse_strides[b];

  const int nc_a = (nf[a] + 1) / 2;
  const int nc_b = (nf[b] + 1) / 2;

  const int ext_b = nc_b + S - 1;

  // tmp1 layout: [idx_b_ext][idx_a_fine]  dims: ext_b x nf[a]
  const int t1_sa = 1;
  const int t1_sb = nf[a];

  // result layout: canonical [fj][fi]
  const int r_s0 = 1;
  const int r_s1 = nf[0];
  int r_sa, r_sb;
  {
    int r_strides[2] = { r_s0, r_s1 };
    r_sa             = r_strides[a];
    r_sb             = r_strides[b];
  }

  const int ecs_b = cs[b] - H_SZ;

  // ----- Pass 1: prolongate along axis a (coarse -> tmp1) ------------------
  for (int ib = 0; ib < ext_b; ++ib)
  {
    const int val_b       = ecs_b + ib;
    const int t1_base     = ib * t1_sb;
    const int coarse_base = val_b * cs_b;

    bool bl = f_isleft[a];
    int ci  = cs[a];

    for (int ia = 0; ia < nf[a]; ++ia)
    {
      const Real* w = bl ? coeff_L : coeff_R;

      Real val = 0.0;
      for (int s = 0; s < S; ++s)
        val += w[s] * coarse_n[coarse_base + (ci - H_SZ + s) * cs_a];

      tmp1[t1_base + ia] = val;

      bl = !bl;
      ci += bl;
    }
  }

  // ----- Pass 2: prolongate along axis b (tmp1 -> result) ------------------
  {
    bool bl = f_isleft[b];
    int cb  = cs[b];

    for (int ib = 0; ib < nf[b]; ++ib)
    {
      const Real* w    = bl ? coeff_L : coeff_R;
      const int r_base = ib * r_sb;

      for (int s = 0; s < S; ++s)
      {
        const Real coef = w[s];
        const int src_b = cb - H_SZ + s;
        const int src   = (src_b - ecs_b) * t1_sb;

        if (s == 0)
        {
          for (int ia = 0; ia < nf[a]; ++ia)
            result[r_base + ia * r_sa] = coef * tmp1[src + ia];
        }
        else
        {
          for (int ia = 0; ia < nf[a]; ++ia)
            result[r_base + ia * r_sa] += coef * tmp1[src + ia];
        }
      }

      bl = !bl;
      cb += bl;
    }
  }
}

#endif  // DBG_CX_RESTRICT_TENSORPRODUCT || DBG_CX_PROLONG_TENSORPRODUCT

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCellCenteredX<H_SZ>(...)
//  \brief restrict cell centered (extended) values
//  H_SZ is the interpolation stencil half-size (resultant degree = 2*H_SZ -
//  1). H_SZ = NGHOST for high-order, H_SZ = 1 for low-order (linear).

template <int H_SZ>
void MeshRefinement::RestrictCellCenteredX(const AthenaArray<Real>& fine,
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
  using InterpUniform::InterpolateLagrangeUniform;

  MeshBlock* pmb = pmy_block_;

  if (pmb->block_size.nx3 > 1)
  {  // 3D

#ifdef DBG_CX_RESTRICT_TENSORPRODUCT
     // -----------------------------------------------------------------------
    // Symmetrized tensor-product restriction (3D).
    // Runs 3 cyclic axis orderings {i,j,k}, {j,k,i}, {k,i,j} and averages
    // the results, restoring the rotational symmetry that a single fixed
    // ordering would break in floating-point arithmetic.
    // Cost: 3 * 3 * 2 * H_SZ = 18*H_SZ FMA per coarse cell (vs 8*H_SZ^3
    // fused).
    // -----------------------------------------------------------------------

    const int nc[3] = { cei - csi + 1, cej - csj + 1, cek - csk + 1 };

    const int fs[3] = { 2 * (csi - pmb->cx_cis) + pmb->cx_is,
                        2 * (csj - pmb->cx_cjs) + pmb->cx_js,
                        2 * (csk - pmb->cx_cks) + pmb->cx_ks };

    const int fe[3] = { 2 * (cei - pmb->cx_cis) + pmb->cx_is,
                        2 * (cej - pmb->cx_cjs) + pmb->cx_js,
                        2 * (cek - pmb->cx_cks) + pmb->cx_ks };

    int efs[3], nf_ext[3];
    for (int d = 0; d < 3; ++d)
    {
      efs[d]    = fs[d] - (H_SZ - 1);
      nf_ext[d] = (fe[d] + H_SZ) - efs[d] + 1;
    }

    const int fine_strides[3] = { 1,
                                  fine.GetDim1(),
                                  fine.GetDim1() * fine.GetDim2() };

    // Scratch sizing: max over all 3 orderings of {a,b,c}:
    //   tmp1_max = max_a(nc[a]) * max_{b,c}(nf_ext[b] * nf_ext[c])
    //   tmp2_max = max_{a,b}(nc[a]*nc[b]) * max_c(nf_ext[c])
    // Conservative: use product of all maxes.
    int max_nc = nc[0], max_nfe = nf_ext[0];
    for (int d = 1; d < 3; ++d)
    {
      if (nc[d] > max_nc)
        max_nc = nc[d];
      if (nf_ext[d] > max_nfe)
        max_nfe = nf_ext[d];
    }
    const int tmp1_sz = max_nc * max_nfe * max_nfe;
    const int tmp2_sz = max_nc * max_nc * max_nfe;
    const int res_sz  = nc[0] * nc[1] * nc[2];

    Real* tmp1       = cx_scratch1_.data();
    Real* tmp2       = cx_scratch2_.data();
    Real* result_buf = cx_scratch3_.data();
    Real* accum      = cx_scratch4_.data();

    static constexpr int orderings[3][3] = { { 0, 1, 2 },
                                             { 1, 2, 0 },
                                             { 2, 0, 1 } };

    for (int n = sn; n <= en; ++n)
    {
      const Real* fine_n = fine.data() + n * fine.GetStride4();
      for (int p = 0; p < 3; ++p)
      {
        RestrictCX_TP_3D_ordered<H_SZ>(fine_n,
                                       result_buf,
                                       fine_strides,
                                       orderings[p],
                                       fs,
                                       nc,
                                       efs,
                                       nf_ext,
                                       tmp1,
                                       tmp2);

        if (p == 0)
        {
          for (int idx = 0; idx < res_sz; ++idx)
            accum[idx] = result_buf[idx];
        }
        else
        {
          for (int idx = 0; idx < res_sz; ++idx)
            accum[idx] += result_buf[idx];
        }
      }

      for (int ck = csk; ck <= cek; ++ck)
        for (int cj = csj; cj <= cej; ++cj)
          for (int ci = csi; ci <= cei; ++ci)
            coarse(n, ck, cj, ci) =
              ONE_3RD * accum[(ck - csk) * nc[1] * nc[0] + (cj - csj) * nc[0] +
                              (ci - csi)];
    }

#else  // !DBG_CX_RESTRICT_TENSORPRODUCT

    for (int n = sn; n <= en; ++n)
      for (int cx_ck = csk; cx_ck <= cek; cx_ck++)
      {
        // left child idx on fine grid
        const int cx_fk = 2 * (cx_ck - pmb->cx_cks) + pmb->cx_ks;

        for (int cx_cj = csj; cx_cj <= cej; cx_cj++)
        {
          // left child idx on fine grid
          const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

          for (int cx_ci = csi; cx_ci <= cei; cx_ci++)
          {
            // left child idx on fine grid
            const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

            // use templated
            // ----------------------------------------------------
            coarse(n, cx_ck, cx_cj, cx_ci) = 0.0;

            for (int dk = 0; dk < H_SZ; ++dk)
            {
              int const cx_fk_l = cx_fk - dk;
              int const cx_fk_r = cx_fk + dk + 1;
              Real const lck =
                InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dk - 1];

              for (int dj = 0; dj < H_SZ; ++dj)
              {
                int const cx_fj_l = cx_fj - dj;
                int const cx_fj_r = cx_fj + dj + 1;
                Real const lckj =
                  lck * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

                for (int di = 0; di < H_SZ; ++di)
                {
                  int const cx_fi_l = cx_fi - di;
                  int const cx_fi_r = cx_fi + di + 1;

                  Real const lckji =
                    lckj *
                    InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

                  Real const f_rrr = fine(n, cx_fk_r, cx_fj_r, cx_fi_r);
                  Real const f_lrr = fine(n, cx_fk_l, cx_fj_r, cx_fi_r);
                  Real const f_rlr = fine(n, cx_fk_r, cx_fj_l, cx_fi_r);
                  Real const f_rrl = fine(n, cx_fk_r, cx_fj_r, cx_fi_l);

                  Real const f_llr = fine(n, cx_fk_l, cx_fj_l, cx_fi_r);
                  Real const f_rll = fine(n, cx_fk_r, cx_fj_l, cx_fi_l);
                  Real const f_lrl = fine(n, cx_fk_l, cx_fj_r, cx_fi_l);
                  Real const f_lll = fine(n, cx_fk_l, cx_fj_l, cx_fi_l);

                  coarse(n, cx_ck, cx_cj, cx_ci) +=
                    lckji *
#ifdef DBG_SYMMETRIZE_CHEAP
                    FloatingPoint::sum_corners(
#else
                    FloatingPoint::sum_associative(
#endif
                      f_rrr, f_lll, f_rrl, f_llr, f_lrl, f_rlr, f_lrr, f_rll);
                }
              }
            }
          }
        }
      }

#endif  // DBG_CX_RESTRICT_TENSORPRODUCT
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;

#ifdef DBG_CX_RESTRICT_TENSORPRODUCT
    // -----------------------------------------------------------------------
    // Symmetrized tensor-product restriction (2D).
    // Runs 2 axis orderings {i,j} and {j,i}, averages the results.
    // -----------------------------------------------------------------------
    const int nc[2] = { cei - csi + 1, cej - csj + 1 };

    const int fs[2] = { 2 * (csi - pmb->cx_cis) + pmb->cx_is,
                        2 * (csj - pmb->cx_cjs) + pmb->cx_js };

    const int fe[2] = { 2 * (cei - pmb->cx_cis) + pmb->cx_is,
                        2 * (cej - pmb->cx_cjs) + pmb->cx_js };

    int efs[2], nf_ext[2];
    for (int d = 0; d < 2; ++d)
    {
      efs[d]    = fs[d] - (H_SZ - 1);
      nf_ext[d] = (fe[d] + H_SZ) - efs[d] + 1;
    }

    // fine(n, cx_fk, j, i) - k is fixed; effective strides for i and j
    const int fine_strides_2d[2] = { 1, fine.GetDim1() };
    const int fine_k_offset      = cx_fk * fine.GetDim1() * fine.GetDim2();

    int max_nc2          = (nc[0] > nc[1]) ? nc[0] : nc[1];
    int max_nfe2         = (nf_ext[0] > nf_ext[1]) ? nf_ext[0] : nf_ext[1];
    const int tmp1_2d_sz = max_nc2 * max_nfe2;
    const int res_2d_sz  = nc[0] * nc[1];

    Real* tmp1_2d       = cx_scratch1_.data();
    Real* result_buf_2d = cx_scratch3_.data();
    Real* accum_2d      = cx_scratch4_.data();

    static constexpr int orderings_2d[2][2] = { { 0, 1 }, { 1, 0 } };

    for (int n = sn; n <= en; ++n)
    {
      const Real* fine_n = fine.data() + n * fine.GetStride4() + fine_k_offset;

      for (int p = 0; p < 2; ++p)
      {
        RestrictCX_TP_2D_ordered<H_SZ>(fine_n,
                                       result_buf_2d,
                                       fine_strides_2d,
                                       orderings_2d[p],
                                       fs,
                                       nc,
                                       efs,
                                       nf_ext,
                                       tmp1_2d);

        if (p == 0)
        {
          for (int idx = 0; idx < res_2d_sz; ++idx)
            accum_2d[idx] = result_buf_2d[idx];
        }
        else
        {
          for (int idx = 0; idx < res_2d_sz; ++idx)
            accum_2d[idx] += result_buf_2d[idx];
        }
      }

      for (int cj = csj; cj <= cej; ++cj)
        for (int ci = csi; ci <= cei; ++ci)
          coarse(n, cx_ck, cj, ci) =
            0.5 * accum_2d[(cj - csj) * nc[0] + (ci - csi)];
    }

#else  // !DBG_CX_RESTRICT_TENSORPRODUCT

    for (int n = sn; n <= en; ++n)
      for (int cx_cj = csj; cx_cj <= cej; cx_cj++)
      {
        // left child idx on fine grid
        const int cx_fj = 2 * (cx_cj - pmb->cx_cjs) + pmb->cx_js;

        for (int cx_ci = csi; cx_ci <= cei; cx_ci++)
        {
          // left child idx on fine grid
          const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

          // use templated
          // ------------------------------------------------------
          coarse(n, cx_ck, cx_cj, cx_ci) = 0.0;

          for (int dj = 0; dj < H_SZ; ++dj)
          {
            int const cx_fj_l = cx_fj - dj;
            int const cx_fj_r = cx_fj + dj + 1;
            Real const lcj =
              InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - dj - 1];

            for (int di = 0; di < H_SZ; ++di)
            {
              int const cx_fi_l = cx_fi - di;
              int const cx_fi_r = cx_fi + di + 1;

              Real const lcji =
                lcj * InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];

              Real const f_uu = fine(n, cx_fk, cx_fj_r, cx_fi_r);
              Real const f_ul = fine(n, cx_fk, cx_fj_r, cx_fi_l);
              Real const f_lu = fine(n, cx_fk, cx_fj_l, cx_fi_r);
              Real const f_ll = fine(n, cx_fk, cx_fj_l, cx_fi_l);

              coarse(n, cx_ck, cx_cj, cx_ci) += lcji *
#ifdef DBG_SYMMETRIZE_CHEAP
                                                FloatingPoint::sum_corners(
#else
                                                FloatingPoint::sum_associative(
#endif
                                                  f_uu, f_ll, f_lu, f_ul);
            }
          }
        }
      }

#endif  // DBG_CX_RESTRICT_TENSORPRODUCT
  }
  else
  {  // 1D
    const int cx_fj = pmb->js, cx_cj = pmb->cjs;
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;

    for (int n = sn; n <= en; ++n)
      for (int cx_ci = csi; cx_ci <= cei; cx_ci++)
      {
        // left child idx on fine grid
        const int cx_fi = 2 * (cx_ci - pmb->cx_cis) + pmb->cx_is;

        // use templated
        // --------------------------------------------------------
        coarse(n, cx_ck, cx_cj, cx_ci) = 0.0;

        for (int di = 0; di < H_SZ; ++di)
        {
          Real const lc =
            InterpolateLagrangeUniform<H_SZ>::coeff[H_SZ - di - 1];
          int const cx_fi_l = cx_fi - di;
          int const cx_fi_r = cx_fi + di + 1;

          coarse(n, cx_ck, cx_cj, cx_ci) +=
            lc *
            (fine(n, cx_fk, cx_fj, cx_fi_l) + fine(n, cx_fk, cx_fj, cx_fi_r));
        }
      }
  }
}

template void MeshRefinement::RestrictCellCenteredX<NGHOST>(
  const AthenaArray<Real>& fine,
  AthenaArray<Real>& coarse,
  int sn,
  int en,
  int csi,
  int cei,
  int csj,
  int cej,
  int csk,
  int cek);
template void MeshRefinement::RestrictCellCenteredX<1>(
  const AthenaArray<Real>& fine,
  AthenaArray<Real>& coarse,
  int sn,
  int en,
  int csi,
  int cei,
  int csj,
  int cej,
  int csk,
  int cek);

//----------------------------------------------------------------------------------------
// Restriction utilizing only physical data from fine to coarse grid
void MeshRefinement::RestrictCellCenteredXWithInteriorValues(
  const AthenaArray<Real>& fine,
  AthenaArray<Real>& coarse,
  int sn,
  int en)
{
  using namespace numprox::interpolation;

  MeshBlock* pmb   = pmy_block_;
  Coordinates* pco = pmb->pcoord;

  int si, sj, sk, ei, ej, ek;
  si = pmb->cx_cis;
  ei = pmb->cx_cie;
  sj = pmb->cx_cjs;
  ej = pmb->cx_cje;
  sk = pmb->cx_cks;
  ek = pmb->cx_cke;

  // BD: debug with LO
  // RestrictCellCenteredX<NGHOST>(
  //   fine, coarse, sn, en,
  //   si, ei,
  //   sj, ej,
  //   sk, ek);
  // return;

  const int Ns_x3 = pmb->block_size.nx3 - 1;  // # phys. nodes - 1
  const int Ns_x2 = pmb->block_size.nx2 - 1;  // # phys. nodes - 1
  const int Ns_x1 = pmb->block_size.nx1 - 1;

#ifdef Z4C_CX_NUM_RBC_INIT_LO
  if (Z4C_CX_NUM_RBC > 0)
  {
    RestrictCellCenteredX<1>(fine, coarse, sn, en, si, ei, sj, ej, sk, ek);
    return;
  }
#endif

  // RestrictCellCenteredX<1>(fine, coarse, sn, en,
  //                               si, ei, sj, ej, sk, ek);
  // return;

  // Floater-Hormann blending parameter controls the formal order of approx.
  // const int d = (NGHOST-1) * 2 + 1;

  AthenaArray<Real>& var_t = coarse;
  AthenaArray<Real>& var_s = const_cast<AthenaArray<Real>&>(fine);

  if (pmb->block_size.nx3 > 1)
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int ck=pmb->cx_cks; ck<=pmb->cx_cke; ++ck)
    {
      // left child idx on fundamental grid
      const int cx_fk = 2 * (ck - pmb->cx_cks) + pmb->cx_ks;

      Real* x3_s = &(pco->x3v(NGHOST));

      // coarse variable grids are constructed with CC ghost number
      const Real x3_t = pcoarsec->x3v(NCGHOST+(ck-NCGHOST_CX));

      for(int cj=pmb->cx_cjs; cj<=pmb->cx_cje; ++cj)
      {
        // left child idx on fundamental grid
        const int cx_fj = 2 * (cj - pmb->cx_cjs) + pmb->cx_js;

        Real* x2_s = &(pco->x2v(NGHOST));

        // coarse variable grids are constructed with CC ghost number
        const Real x2_t = pcoarsec->x2v(NCGHOST+(cj-NCGHOST_CX));

        for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
        {
          // left child idx on fundamental grid
          const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

          Real* x1_s = &(pco->x1v(NGHOST));
          Real* fcn_s = &(var_s(n,0,0,0));

          const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

          var_t(n,ck,cj,ci) = Floater_Hormann::interp_3d(
            x1_t, x2_t, x3_t,
            x1_s, x2_s, x3_s,
            fcn_s,
            Ns_x1, Ns_x2, Ns_x3, d, NGHOST);
        }
      }
    }
    */

    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));
    Real* x3_s = &(pco->x3v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));
    Real* x3_t = &(pcoarsec->x3v(NCGHOST+(pmb->cx_cks-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;
    const int Nt_x3 = pmb->block_size.nx3 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i3c = new interp_nd_weights_precomputed(
      x1_t, x2_t, x3_t,
      x1_s, x2_s, x3_s,
      Nt_x1, Nt_x2, Nt_x3,
      Ns_x1, Ns_x2, Ns_x3,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i3c->eval(fcn_t, fcn_s);
    }

    delete i3c;
    */

    for (int n = sn; n <= en; ++n)
    {
      const Real* const fcn_s = &(var_s(n, 0, 0, 0));
      Real* fcn_t             = &(var_t(n, 0, 0, 0));

      // ind_interior_r_op->eval(fcn_t, fcn_s);
      ind_interior_r_op->eval_opt_nn(fcn_t, fcn_s);
    }
  }
  else if (pmb->block_size.nx2 > 1)
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int cj=pmb->cx_cjs; cj<=pmb->cx_cje; ++cj)
    {
      // left child idx on fundamental grid
      const int cx_fj = 2 * (cj - pmb->cx_cjs) + pmb->cx_js;

      Real* x2_s = &(pco->x2v(NGHOST));

      // coarse variable grids are constructed with CC ghost number
      const Real x2_t = pcoarsec->x2v(NCGHOST+(cj-NCGHOST_CX));

      for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
      {
        // left child idx on fundamental grid
        const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

        Real* x1_s = &(pco->x1v(NGHOST));
        Real* fcn_s = &(var_s(n,0,0,0));

        const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

        var_t(n,0,cj,ci) = Floater_Hormann::interp_2d(
          x1_t, x2_t,
          x1_s, x2_s,
          fcn_s,
          Ns_x1, Ns_x2, d, NGHOST);
      }
    }
    */

    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x2_s = &(pco->x2v(NGHOST));

    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));
    Real* x2_t = &(pcoarsec->x2v(NCGHOST+(pmb->cx_cjs-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;
    const int Nt_x2 = pmb->block_size.nx2 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i2c = new interp_nd_weights_precomputed(
      x1_t, x2_t,
      x1_s, x2_s,
      Nt_x1, Nt_x2,
      Ns_x1, Ns_x2,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i2c->eval(fcn_t, fcn_s);
    }

    delete i2c;
    */

    for (int n = sn; n <= en; ++n)
    {
      const Real* const fcn_s = &(var_s(n, 0, 0, 0));
      Real* fcn_t             = &(var_t(n, 0, 0, 0));

      ind_interior_r_op->eval_opt_nn(fcn_t, fcn_s);
      // ind_interior_r_op->eval(fcn_t, fcn_s);

      // ind_interior_r_op->eval(fcn_t, fcn_s,
      //                         pmb->cx_cis-NCGHOST_CX,
      //                         pmb->cx_cie-NCGHOST_CX,
      //                         pmb->cx_cjs-NCGHOST_CX,
      //                         pmb->cx_cje-NCGHOST_CX);
    }
  }
  else
  {
    /*
    for(int n=sn; n<=en; ++n)
    for(int ci=pmb->cx_cis; ci<=pmb->cx_cie; ++ci)
    {
      // left child idx on fundamental grid
      const int cx_fi = 2 * (ci - pmb->cx_cis) + pmb->cx_is;

      Real* x1_s = &(pco->x1v(NGHOST));
      Real* fcn_s = &(var_s(n,0,0,0));

      // coarse variable grids are constructed with CC ghost number
      const Real x1_t = pcoarsec->x1v(NCGHOST+(ci-NCGHOST_CX));

      var_t(n,0,0,ci) = Floater_Hormann::interp_1d(
        x1_t, x1_s,
        fcn_s,
        Ns_x1, d, NGHOST);
    }
    */

    /*
    // grids
    Real* x1_s = &(pco->x1v(NGHOST));
    Real* x1_t = &(pcoarsec->x1v(NCGHOST+(pmb->cx_cis-NCGHOST_CX)));

    const int Nt_x1 = pmb->block_size.nx1 / 2 - 1;

    typedef Floater_Hormann::interp_nd_weights_precomputed<Real, Real>
      interp_nd_weights_precomputed;

    interp_nd_weights_precomputed * i1c = new interp_nd_weights_precomputed(
      x1_t, x1_s,
      Nt_x1, Ns_x1,
      d,
      NCGHOST_CX, NGHOST
    );

    for(int n=sn; n<=en; ++n)
    {
      const Real* const fcn_s = &(var_s(n,0,0,0));
      Real* fcn_t = &(var_t(n,0,0,0));

      i1c->eval(fcn_t, fcn_s);
    }


    delete i1c;
    */

    for (int n = sn; n <= en; ++n)
    {
      const Real* const fcn_s = &(var_s(n, 0, 0, 0));
      Real* fcn_t             = &(var_t(n, 0, 0, 0));

      // ind_interior_r_op->eval(fcn_t, fcn_s);
      ind_interior_r_op->eval_opt_nn(fcn_t, fcn_s);

      // ind_interior_r_op->eval(fcn_t, fcn_s,
      //                         pmb->cx_cis-NCGHOST_CX,
      //                         pmb->cx_cigs-NCGHOST_CX);
    }

    // {
    //   const Real* const fcn_s = &(var_s(n,0,0,0));

    //   for(int i=pmb->cx_is; i<=pmb->cx_igs; ++i)
    //   {
    //     var_t(n,0,0,i) = Floater_Hormann::interp_1d(
    //       pmb->pcoord->x1v(i),
    //       &(pmb->pcoord->x1v(NGHOST)),
    //       fcn_s,
    //       d+2, // stencil size
    //       d,   // order
    //       NGHOST
    //     );
    //   }

    // }
  }
}

void MeshRefinement::ProlongateCellCenteredXBCValues(
  const AthenaArray<Real>& coarse,
  AthenaArray<Real>& fine,
  int sn,
  int en,
  int csi,
  int cei,
  int csj,
  int cej,
  int csk,
  int cek)
{
  // here H_SZ * 2 is resultant interpolant degree.
  // For boundary use one node fewer for compatibility with
  // ProlongateCellCenteredXGhosts
  const int H_SZ = (2 * NCGHOST_CX - NGHOST) / 2;
  ProlongateCellCenteredX<H_SZ>(
    coarse, fine, sn, en, csi, cei, csj, cej, csk, cek);
}

void MeshRefinement::ProlongateCellCenteredXValues(
  const AthenaArray<Real>& coarse,
  AthenaArray<Real>& fine,
  int sn,
  int en,
  int csi,
  int cei,
  int csj,
  int cej,
  int csk,
  int cek)
{
  // here H_SZ * 2 is resultant interpolant degree
  const int H_SZ = (2 * NCGHOST_CX - NGHOST) / 2 + 1;
  ProlongateCellCenteredX<H_SZ>(
    coarse, fine, sn, en, csi, cei, csj, cej, csk, cek);
}

template <int H_SZ>
void MeshRefinement::ProlongateCellCenteredX(const AthenaArray<Real>& coarse,
                                             AthenaArray<Real>& fine,
                                             int sn,
                                             int en,
                                             int csi,
                                             int cei,
                                             int csj,
                                             int cej,
                                             int csk,
                                             int cek)
{
  using InterpUniform::InterpolateLagrangeUniformChildren;

  MeshBlock* pmb = pmy_block_;
  // here H_SZ * 2 is resultant interpolant degree

  if (pmb->block_size.nx3 > 1)
  {  // 3D

    // see 1d case
    int cx_fsi = 2 * (csi - pmb->cx_cis) + pmb->cx_is;
    int cx_fei = 2 * (cei - pmb->cx_cis) + pmb->cx_is + 1;

    int cx_fsj = 2 * (csj - pmb->cx_cjs) + pmb->cx_js;
    int cx_fej = 2 * (cej - pmb->cx_cjs) + pmb->cx_js + 1;

    int cx_fsk = 2 * (csk - pmb->cx_cks) + pmb->cx_ks;
    int cx_fek = 2 * (cek - pmb->cx_cks) + pmb->cx_ks + 1;

    cx_fsi = (cx_fsi >= 0) ? cx_fsi : 0;
    cx_fei = (cx_fei + 1 < pmb->ncells1) ? cx_fei : pmb->ncells1 - 1;

    cx_fsj = (cx_fsj >= 0) ? cx_fsj : 0;
    cx_fej = (cx_fej + 1 < pmb->ncells2) ? cx_fej : pmb->ncells2 - 1;

    cx_fsk = (cx_fsk >= 0) ? cx_fsk : 0;
    cx_fek = (cx_fek + 1 < pmb->ncells3) ? cx_fek : pmb->ncells3 - 1;

    const bool cx_f_isleft = !((cx_fsi + NGHOST) % 2);
    const bool cx_f_jsleft = !((cx_fsj + NGHOST) % 2);
    const bool cx_f_ksleft = !((cx_fsk + NGHOST) % 2);

#ifdef DBG_CX_PROLONG_TENSORPRODUCT
    // -----------------------------------------------------------------------
    // Symmetrized tensor-product prolongation (3D).
    // Runs 3 cyclic axis orderings {0,1,2}, {1,2,0}, {2,0,1} and averages
    // the results, restoring the rotational symmetry that a single fixed
    // ordering would break in floating-point arithmetic.
    // Cost: 3 * 3 * (2*H_SZ+1) sweeps vs (2*H_SZ+1)^3 fused.
    // -----------------------------------------------------------------------
    constexpr int S = 2 * H_SZ + 1;

    Real coeff_R[S];
    for (int s = 0; s < S; ++s)
      coeff_R[s] = InterpolateLagrangeUniformChildren<H_SZ>::coeff[S - 1 - s];

    const int nfi = cx_fei - cx_fsi + 1;
    const int nfj = cx_fej - cx_fsj + 1;
    const int nfk = cx_fek - cx_fsk + 1;

    const int nf_arr[3]    = { nfi, nfj, nfk };
    const int cs_arr[3]    = { csi, csj, csk };
    const bool f_isleft[3] = { cx_f_isleft, cx_f_jsleft, cx_f_ksleft };

    const int coarse_strides[3] = { 1,
                                    coarse.GetDim1(),
                                    coarse.GetDim1() * coarse.GetDim2() };

    // Scratch sizing: conservative max over all 3 orderings.
    // For ordering {a,b,c}:
    //   nc_d = (nf[d]+1)/2
    //   ext_d = nc_d + S - 1
    //   tmp1: nf[a] * ext_b * ext_c
    //   tmp2: nf[a] * nf[b] * ext_c
    int max_nf = nfi, max_ext = ((nfi + 1) / 2) + S - 1;
    for (int d = 1; d < 3; ++d)
    {
      if (nf_arr[d] > max_nf)
        max_nf = nf_arr[d];
      const int ext_d = ((nf_arr[d] + 1) / 2) + S - 1;
      if (ext_d > max_ext)
        max_ext = ext_d;
    }
    const int tmp1_sz = max_nf * max_ext * max_ext;
    const int tmp2_sz = max_nf * max_nf * max_ext;
    const int res_sz  = nfi * nfj * nfk;

    Real* tmp1       = cx_scratch1_.data();
    Real* tmp2       = cx_scratch2_.data();
    Real* result_buf = cx_scratch3_.data();
    Real* accum      = cx_scratch4_.data();

    static constexpr int orderings[3][3] = { { 0, 1, 2 },
                                             { 1, 2, 0 },
                                             { 2, 0, 1 } };

    for (int n = sn; n <= en; ++n)
    {
      const Real* coarse_n = coarse.data() + n * coarse.GetStride4();

      for (int p = 0; p < 3; ++p)
      {
        ProlongCX_TP_3D_ordered<H_SZ>(coarse_n,
                                      result_buf,
                                      coarse_strides,
                                      orderings[p],
                                      cs_arr,
                                      nf_arr,
                                      f_isleft,
                                      tmp1,
                                      tmp2,
                                      coeff_R);

        if (p == 0)
        {
          for (int idx = 0; idx < res_sz; ++idx)
            accum[idx] = result_buf[idx];
        }
        else
        {
          for (int idx = 0; idx < res_sz; ++idx)
            accum[idx] += result_buf[idx];
        }
      }

      for (int fk = cx_fsk; fk <= cx_fek; ++fk)
        for (int fj = cx_fsj; fj <= cx_fej; ++fj)
          for (int fi = cx_fsi; fi <= cx_fei; ++fi)
            fine(n, fk, fj, fi) =
              ONE_3RD * accum[(fk - cx_fsk) * nfj * nfi + (fj - cx_fsj) * nfi +
                              (fi - cx_fsi)];
    }

#else  // !DBG_CX_PROLONG_TENSORPRODUCT

    for (int n = sn; n <= en; ++n)
    {
      bool bl_k = cx_f_ksleft;
      int ph_k  = (bl_k) ? -1 : 1;

      for (int cx_fk = cx_fsk, cx_ck = csk; cx_fk <= cx_fek; ++cx_fk)
      {
        bool bl_j = cx_f_jsleft;
        int ph_j  = (bl_j) ? -1 : 1;

        for (int cx_fj = cx_fsj, cx_cj = csj; cx_fj <= cx_fej; ++cx_fj)
        {
          bool bl_i = cx_f_isleft;
          int ph_i  = (bl_i) ? -1 : 1;

          for (int cx_fi = cx_fsi, cx_ci = csi; cx_fi <= cx_fei; ++cx_fi)
          {
            Real fine_val = 0.0;

            for (int dk = 0; dk < 2 * H_SZ + 1; ++dk)
            {
              Real const l_k =
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[dk];
              const int cx_cwk = cx_ck + ph_k * (H_SZ - dk);

              for (int dj = 0; dj < 2 * H_SZ + 1; ++dj)
              {
                Real const l_kj =
                  l_k * InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj];
                const int cx_cwj = cx_cj + ph_j * (H_SZ - dj);

                for (int di = 0; di < 2 * H_SZ + 1; ++di)
                {
                  Real const l_kji =
                    l_kj * InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];
                  const int cx_cwi = cx_ci + ph_i * (H_SZ - di);

                  Real const fc_kji = coarse(n, cx_cwk, cx_cwj, cx_cwi);
                  fine_val += l_kji * fc_kji;
                }
              }
            }

            fine(n, cx_fk, cx_fj, cx_fi) = fine_val;

            bl_i = !bl_i;
            cx_ci += bl_i;
            ph_i = -ph_i;
          }

          bl_j = !bl_j;
          cx_cj += bl_j;
          ph_j = -ph_j;
        }

        bl_k = !bl_k;
        cx_ck += bl_k;
        ph_k = -ph_k;
      }
    }

#endif  // DBG_CX_PROLONG_TENSORPRODUCT
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D

    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;

    // see 1d case
    int cx_fsi = 2 * (csi - pmb->cx_cis) + pmb->cx_is;
    int cx_fei = 2 * (cei - pmb->cx_cis) + pmb->cx_is + 1;

    int cx_fsj = 2 * (csj - pmb->cx_cjs) + pmb->cx_js;
    int cx_fej = 2 * (cej - pmb->cx_cjs) + pmb->cx_js + 1;

    cx_fsi = (cx_fsi >= 0) ? cx_fsi : 0;
    cx_fei = (cx_fei + 1 < pmb->ncells1) ? cx_fei : pmb->ncells1 - 1;

    cx_fsj = (cx_fsj >= 0) ? cx_fsj : 0;
    cx_fej = (cx_fej + 1 < pmb->ncells2) ? cx_fej : pmb->ncells2 - 1;

    const bool cx_f_isleft = !((cx_fsi + NGHOST) % 2);
    const bool cx_f_jsleft = !((cx_fsj + NGHOST) % 2);

#ifdef DBG_CX_PROLONG_TENSORPRODUCT
    // -----------------------------------------------------------------------
    // Symmetrized tensor-product prolongation (2D).
    // Runs 2 axis orderings {0,1} and {1,0}, averages the results.
    // -----------------------------------------------------------------------
    constexpr int S = 2 * H_SZ + 1;

    Real coeff_R[S];
    for (int s = 0; s < S; ++s)
      coeff_R[s] = InterpolateLagrangeUniformChildren<H_SZ>::coeff[S - 1 - s];

    const int nfi = cx_fei - cx_fsi + 1;
    const int nfj = cx_fej - cx_fsj + 1;

    const int nf_arr[2]    = { nfi, nfj };
    const int cs_arr[2]    = { csi, csj };
    const bool f_isleft[2] = { cx_f_isleft, cx_f_jsleft };

    const int coarse_strides[2] = { 1, coarse.GetDim1() };
    const int coarse_k_offset   = cx_ck * coarse.GetDim1() * coarse.GetDim2();

    // Scratch sizing: max over both orderings.
    // For ordering {a,b}: tmp1 = nf[a] * ext_b, where ext_b = (nf[b]+1)/2+S-1
    int max_nf2  = (nfi > nfj) ? nfi : nfj;
    int max_ext2 = (((nfi + 1) / 2) + S - 1);
    {
      const int ext_j = ((nfj + 1) / 2) + S - 1;
      if (ext_j > max_ext2)
        max_ext2 = ext_j;
    }
    const int tmp1_2d_sz = max_nf2 * max_ext2;
    const int res_2d_sz  = nfi * nfj;

    Real* tmp1_2d       = cx_scratch1_.data();
    Real* result_buf_2d = cx_scratch3_.data();
    Real* accum_2d      = cx_scratch4_.data();

    static constexpr int orderings_2d[2][2] = { { 0, 1 }, { 1, 0 } };

    for (int n = sn; n <= en; ++n)
    {
      const Real* coarse_n =
        coarse.data() + n * coarse.GetStride4() + coarse_k_offset;

      for (int p = 0; p < 2; ++p)
      {
        ProlongCX_TP_2D_ordered<H_SZ>(coarse_n,
                                      result_buf_2d,
                                      coarse_strides,
                                      orderings_2d[p],
                                      cs_arr,
                                      nf_arr,
                                      f_isleft,
                                      tmp1_2d,
                                      coeff_R);

        if (p == 0)
        {
          for (int idx = 0; idx < res_2d_sz; ++idx)
            accum_2d[idx] = result_buf_2d[idx];
        }
        else
        {
          for (int idx = 0; idx < res_2d_sz; ++idx)
            accum_2d[idx] += result_buf_2d[idx];
        }
      }

      for (int fj = cx_fsj; fj <= cx_fej; ++fj)
        for (int fi = cx_fsi; fi <= cx_fei; ++fi)
          fine(n, cx_fk, fj, fi) =
            0.5 * accum_2d[(fj - cx_fsj) * nfi + (fi - cx_fsi)];
    }

#else  // !DBG_CX_PROLONG_TENSORPRODUCT

    for (int n = sn; n <= en; ++n)
    {
      bool bl_j = cx_f_jsleft;
      int ph_j  = (bl_j) ? -1 : 1;

      for (int cx_fj = cx_fsj, cx_cj = csj; cx_fj <= cx_fej; ++cx_fj)
      {
        bool bl_i = cx_f_isleft;
        int ph_i  = (bl_i) ? -1 : 1;

        for (int cx_fi = cx_fsi, cx_ci = csi; cx_fi <= cx_fei; ++cx_fi)
        {
          fine(n, cx_fk, cx_fj, cx_fi) = 0.0;

          for (int dj = 0; dj < 2 * H_SZ + 1; ++dj)
          {
            Real const l_j =
              InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj];
            const int cx_cwj = cx_cj + ph_j * (H_SZ - dj);

            for (int di = 0; di < 2 * H_SZ + 1; ++di)
            {
              Real const l_i =
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];

              const int cx_cwi = cx_ci + ph_i * (H_SZ - di);

              Real const fc_ji = coarse(n, cx_ck, cx_cwj, cx_cwi);
              fine(n, cx_fk, cx_fj, cx_fi) += l_j * l_i * fc_ji;
            }
          }

          bl_i = !bl_i;
          cx_ci += bl_i;
          ph_i = -ph_i;
        }

        bl_j = !bl_j;
        cx_cj += bl_j;
        ph_j = -ph_j;
      }
    }

#endif  // DBG_CX_PROLONG_TENSORPRODUCT
  }
  else
  {  // 1D
    const int cx_fj = pmb->cx_js, cx_cj = pmb->cx_cjs;
    const int cx_fk = pmb->cx_ks, cx_ck = pmb->cx_cks;

    // first child idx on fine grid
    int cx_fsi = 2 * (csi - pmb->cx_cis) + pmb->cx_is;
    // last child idx on fine grid (+1 for potential right child)
    int cx_fei = 2 * (cei - pmb->cx_cis) + pmb->cx_is + 1;

    // Adjust bounds to be physical
    cx_fsi = (cx_fsi >= 0) ? cx_fsi : 0;
    cx_fei = (cx_fei + 1 < pmb->ncells1) ? cx_fei : pmb->ncells1 - 1;

    // Does first fine index remain a left child after adjusting?
    //
    // NGHOST even:
    // Then !(cx_fsi % 2) is sufficient
    // NGHOST odd:
    // Then the above condition is flipped, this is equivalent to
    // !((cx_fsi + NGHOST) % 2)
    const bool cx_f_isleft = !((cx_fsi + NGHOST) % 2);

    for (int n = sn; n <= en; ++n)
    {
      bool bl_i = cx_f_isleft;  // flip rather than recompute above condition
      int ph_i  = (bl_i) ? -1 : 1;

      for (int cx_fi = cx_fsi, cx_ci = csi; cx_fi <= cx_fei; ++cx_fi)
      {
        fine(n, cx_fk, cx_fj, cx_fi) = 0.0;

        /*
        for (int di=0; di<2*H_SZ+1; ++di)
        {
          Real const l_i = InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];

          // cx_ci-H_SZ+di and cx_ci+H_SZ-di are
          // left and right coarse idxs resp.
          // const int cx_cwi = (bl_i) ? cx_ci-H_SZ+di : cx_ci+H_SZ-di;
          const int cx_cwi = cx_ci + ph_i * (H_SZ-di);

          Real const fc_i = coarse(n,cx_ck,cx_cj,cx_cwi);
          fine(n,cx_fk,cx_fj,cx_fi) += l_i * fc_i;
        }
        */

        for (int di = 0; di < H_SZ; ++di)
        {
          Real const l_li =
            InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];
          Real const l_ui =
            InterpolateLagrangeUniformChildren<H_SZ>::coeff[2 * H_SZ - di];

          // cx_ci-H_SZ+di and cx_ci+H_SZ-di are
          // left and right coarse idxs resp.
          // const int cx_cwi = (bl_i) ? cx_ci-H_SZ+di : cx_ci+H_SZ-di;
          const int cx_cwli = cx_ci + ph_i * (H_SZ - di);
          const int cx_cwui = cx_ci + ph_i * (H_SZ - (2 * H_SZ - di));

          Real const fc_li = coarse(n, cx_ck, cx_cj, cx_cwli);
          Real const fc_ui = coarse(n, cx_ck, cx_cj, cx_cwui);

          fine(n, cx_fk, cx_fj, cx_fi) += (l_li * fc_li + l_ui * fc_ui);
        }

        fine(n, cx_fk, cx_fj, cx_fi) +=
          coarse(n, cx_ck, cx_cj, cx_ci) *
          InterpolateLagrangeUniformChildren<H_SZ>::coeff[H_SZ];

        bl_i = !bl_i;   // toggle left-right as fine-values swept
        cx_ci += bl_i;  // incr. if next will be a left fine child
        ph_i = -ph_i;   // flip idx add/sub
      }
    }
  }
}

// explicit template instantiations for ProlongateCellCenteredX
template void MeshRefinement::ProlongateCellCenteredX<
  (2 * NCGHOST_CX - NGHOST) / 2>(const AthenaArray<Real>&,
                                 AthenaArray<Real>&,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int,
                                 int);
template void MeshRefinement::ProlongateCellCenteredX<
  (2 * NCGHOST_CX - NGHOST) / 2 + 1>(const AthenaArray<Real>&,
                                     AthenaArray<Real>&,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int,
                                     int);
