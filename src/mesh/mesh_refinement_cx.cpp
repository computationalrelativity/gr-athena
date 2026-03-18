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
  }
  else if (pmb->block_size.nx2 > 1)
  {  // 2D
    const int cx_fk = pmb->ks, cx_ck = pmb->cks;
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

              Real const f_uu = fine(n, 0, cx_fj_r, cx_fi_r);
              Real const f_ul = fine(n, 0, cx_fj_r, cx_fi_l);
              Real const f_lu = fine(n, 0, cx_fj_l, cx_fi_r);
              Real const f_ll = fine(n, 0, cx_fj_l, cx_fi_l);

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
            // fine(n,cx_fk,cx_fj,cx_fi) = 0.0;

            for (int dk = 0; dk < 2 * H_SZ + 1; ++dk)
            {
              Real const l_k =
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[dk];
              // const int cx_cwk = (bl_k) ? cx_ck-H_SZ+dk : cx_ck+H_SZ-dk;
              const int cx_cwk = cx_ck + ph_k * (H_SZ - dk);

              for (int dj = 0; dj < 2 * H_SZ + 1; ++dj)
              {
                Real const l_kj =
                  l_k * InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj];
                // const int cx_cwj = (bl_j) ? cx_cj-H_SZ+dj : cx_cj+H_SZ-dj;
                const int cx_cwj = cx_cj + ph_j * (H_SZ - dj);

                for (int di = 0; di < 2 * H_SZ + 1; ++di)
                {
                  Real const l_kji =
                    l_kj * InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];
                  // const int cx_cwi = (bl_i) ? cx_ci-H_SZ+di : cx_ci+H_SZ-di;
                  const int cx_cwi = cx_ci + ph_i * (H_SZ - di);

                  Real const fc_kji = coarse(n, cx_cwk, cx_cwj, cx_cwi);
                  // Real const fc_kji = cx_cwk+cx_cwj+cx_cwi;
                  // fine(n,cx_fk,cx_fj,cx_fi) += l_kji * fc_kji;
                  fine_val += l_kji * fc_kji;
                }
              }
            }

            fine(n, cx_fk, cx_fj, cx_fi) = fine_val;

            // Real fine_val = 0.0;
            // Real const l_c =
            // InterpolateLagrangeUniformChildren<H_SZ>::coeff[H_SZ];

            // for (int dk=0; dk<H_SZ; ++dk)
            // {
            //   Real const l_lk =
            //   InterpolateLagrangeUniformChildren<H_SZ>::coeff[dk]; Real
            //   const l_uk =
            //   InterpolateLagrangeUniformChildren<H_SZ>::coeff[2*H_SZ-dk];

            //   const int cx_cwlk = cx_ck + ph_k * (H_SZ-dk);
            //   const int cx_cwuk = cx_ck + ph_k * (H_SZ-(2*H_SZ-dk));

            //   for (int dj=0; dj<H_SZ; ++dj)
            //   {
            //     Real const l_lj =
            //     InterpolateLagrangeUniformChildren<H_SZ>::coeff[dj]; Real
            //     const l_uj =
            //     InterpolateLagrangeUniformChildren<H_SZ>::coeff[2*H_SZ-dj];

            //     const int cx_cwlj = cx_cj + ph_j * (H_SZ-dj);
            //     const int cx_cwuj = cx_cj + ph_j * (H_SZ-(2*H_SZ-dj));

            //     for (int di=0; di<H_SZ; ++di)
            //     {
            //       Real const l_li =
            //       InterpolateLagrangeUniformChildren<H_SZ>::coeff[di]; Real
            //       const l_ui =
            //       InterpolateLagrangeUniformChildren<H_SZ>::coeff[2*H_SZ-di];

            //       const int cx_cwli = cx_ci + ph_i * (H_SZ-di);
            //       const int cx_cwui = cx_ci + ph_i * (H_SZ-(2*H_SZ-di));

            //       Real const fc_uuu = coarse(n,cx_cwuk,cx_cwuj,cx_cwui);
            //       Real const fc_lll = coarse(n,cx_cwlk,cx_cwlj,cx_cwli);

            //       Real const fc_luu = coarse(n,cx_cwlk,cx_cwuj,cx_cwui);
            //       Real const fc_ulu = coarse(n,cx_cwuk,cx_cwlj,cx_cwui);
            //       Real const fc_uul = coarse(n,cx_cwuk,cx_cwuj,cx_cwli);

            //       Real const fc_llu = coarse(n,cx_cwlk,cx_cwlj,cx_cwui);
            //       Real const fc_ull = coarse(n,cx_cwuk,cx_cwlj,cx_cwli);
            //       Real const fc_lul = coarse(n,cx_cwlk,cx_cwuj,cx_cwli);

            //       fine_val += FloatingPoint::sum_associative(
            //         l_uk*l_uj*l_ui*fc_uuu,
            //         l_lk*l_lj*l_li*fc_lll,
            //         l_lk*l_uj*l_ui*fc_luu,
            //         l_uk*l_lj*l_ui*fc_ulu,
            //         l_uk*l_uj*l_li*fc_uul,
            //         l_lk*l_lj*l_ui*fc_llu,
            //         l_uk*l_lj*l_li*fc_ull,
            //         l_lk*l_uj*l_li*fc_lul
            //       );
            //     }
            //     Real const fc_uuc = coarse(n,cx_cwuk,cx_cwuj,cx_ci);
            //     Real const fc_llc = coarse(n,cx_cwlk,cx_cwlj,cx_ci);

            //     Real const fc_luc = coarse(n,cx_cwlk,cx_cwuj,cx_ci);
            //     Real const fc_ulc = coarse(n,cx_cwuk,cx_cwlj,cx_ci);

            //     fine_val += l_c * FloatingPoint::sum_associative(
            //       l_uk*l_uj*fc_uuc,
            //       l_lk*l_lj*fc_llc,
            //       l_lk*l_uj*fc_luc,
            //       l_uk*l_lj*fc_ulc
            //     );

            //   }
            // }

            // fine(n,cx_fk,cx_fj,cx_fi) = fine_val;

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
            // const int cx_cwj = (bl_j) ? cx_cj-H_SZ+dj : cx_cj+H_SZ-dj;
            const int cx_cwj = cx_cj + ph_j * (H_SZ - dj);

            for (int di = 0; di < 2 * H_SZ + 1; ++di)
            {
              Real const l_i =
                InterpolateLagrangeUniformChildren<H_SZ>::coeff[di];

              // const int cx_cwi = (bl_i) ? cx_ci-H_SZ+di : cx_ci+H_SZ-di;
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
