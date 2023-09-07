// headers ====================================================================
// c / c++
// #include <algorithm>

// Athena++ headers
#include "floating_point.hpp"
#include "interp_intergrid.hpp"
// ============================================================================

// ============================================================================
namespace InterpIntergrid {

// ctor / dtor ----------------------------------------------------------------
template <typename dtype, int H_SZ>
InterpIntergrid<dtype, H_SZ>::InterpIntergrid(
  const int ndim,
  const int *N,
  const dtype *rds,
  const int NG_CC,
  const int NG_VC)
  :
  dim   (ndim),
  NG_CC (NG_CC),
  NG_VC (NG_VC),
  dg    (NG_VC-NG_CC),
  dc    (std::min(NG_VC,NG_CC)-H_SZ+1),
  dv    (std::min(NG_VC,NG_CC)-H_SZ)
{

  this->N = new int[dim];
  this->ncells = new int[dim];
  this->nverts = new int[dim];

  this->strides_cc = new int[dim];
  this->strides_vc = new int[dim];

  this->rds = new dtype[dim];

  for(int i=0; i<dim; ++i)
  {
    this->rds[i] = rds[i];
    this->N[i] = N[i];
    this->ncells[i] = N[i] + 2 * NG_CC;
    this->nverts[i] = N[i] + 2 * NG_CC + 1;

    if(i==0)
    {
      this->strides_cc[i] = 1;
      this->strides_vc[i] = 1;
    }
    else
    {
      this->strides_cc[i] = this->strides_cc[i-1] * ncells[i-1];
      this->strides_vc[i] = this->strides_vc[i-1] * nverts[i-1];
    }
  }

  // populate maximal idx where interp applies (tar. idx)

  // max # of nodes we can extend into ghost layer
  // const int ng_e_vc2cc = NG_VC-H_SZ+1;
  // const int ng_e_cc2vc = NG_CC-H_SZ;

  const int ng_e_vc2cc = dc;
  const int ng_e_cc2vc = dv;

  switch (ndim)
  {
    case 3:
      cc_kl = NG_CC-ng_e_vc2cc;
      // cc_ku = ncells[2]-NG_CC-1+ng_e_vc2cc;
      cc_ku = N[2]+NG_CC-1+ng_e_vc2cc;

      vc_kl = NG_VC-ng_e_cc2vc;
      // vc_ku = nverts[2]-NG_VC-1+ng_e_cc2vc;
      vc_ku = N[2]+NG_VC+ng_e_cc2vc;
    case 2:
      cc_jl = NG_CC-ng_e_vc2cc;
      // cc_ju = ncells[1]-NG_CC-1+ng_e_vc2cc;
      cc_ju = N[1]+NG_CC-1+ng_e_vc2cc;

      vc_jl = NG_VC-ng_e_cc2vc;
      // vc_ju = nverts[1]-NG_VC-1+ng_e_cc2vc;
      vc_ju = N[1]+NG_VC+ng_e_cc2vc;
    default:
      cc_il = NG_CC-ng_e_vc2cc;
      // cc_iu = ncells[0]-NG_CC-1+ng_e_vc2cc;
      cc_iu = N[0]+NG_CC-1+ng_e_vc2cc;

      vc_il = NG_VC-ng_e_cc2vc;
      // vc_iu = nverts[0]-NG_VC-1+ng_e_cc2vc;
      vc_iu = N[0]+NG_VC+ng_e_cc2vc;
  }
}

template <typename dtype, int H_SZ>
InterpIntergrid<dtype, H_SZ>::~InterpIntergrid()
{
  delete[] rds;
  delete[] N;
  delete[] ncells;
  delete[] nverts;
  delete[] strides_cc;
  delete[] strides_vc;
}

// interfaces of specializations ----------------------------------------------
template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::VC2CC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int cc_k,
  const int cc_j)
{
  VC2CC(tar.array(), src.array(), cc_k, cc_j);
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::CC2VC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int vc_k,
  const int vc_j)
{
  CC2VC(tar.array(), src.array(), vc_k, vc_j);
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::VC2CC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int cc_k,
  const int cc_j)
{
  const int nu = src.GetDim4()-1;

  switch (dim)
  {
    case 3:
    {
      const int vc_k = cc_k + dg;  // offset by difference in ghosts
      const int vc_j = cc_j + dg;

      for (int n=0; n<=nu; ++n)
      for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
      {
        tar(n,cc_i) = 0.;

        const int vc_i = cc_i + dg;

        for (int dk=0; dk<H_SZ; ++dk)
        {
          const dtype lc_k = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-dk-1];

          const int vk_l = vc_k - dk;
          const int vk_r = vc_k + dk + 1;

          for (int dj=0; dj<H_SZ; ++dj)
          {
            const dtype lc_kj = lc_k * InterpolateLagrangeUniform_opt<
              H_SZ
            >::coeff[H_SZ-dj-1];

            const int vj_l = vc_j - dj;
            const int vj_r = vc_j + dj + 1;

            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_kji = lc_kj * InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int vi_l = vc_i - di;
              const int vi_r = vc_i + di + 1;

              const dtype s_rrr = src(n, vk_r, vj_r, vi_r);
              const dtype s_lrr = src(n, vk_l, vj_r, vi_r);
              const dtype s_rlr = src(n, vk_r, vj_l, vi_r);
              const dtype s_rrl = src(n, vk_r, vj_r, vi_l);

              const dtype s_llr = src(n, vk_l, vj_l, vi_r);
              const dtype s_rll = src(n, vk_r, vj_l, vi_l);
              const dtype s_lrl = src(n, vk_l, vj_r, vi_l);
              const dtype s_lll = src(n, vk_l, vj_l, vi_l);


              tar(n,cc_i) += lc_kji * FloatingPoint::sum_associative(
                  s_rrr, s_lll, s_rrl, s_llr,
                  s_lrl, s_rlr, s_lrr, s_rll
              );
            }
          }

        }
      }
      break;
    }
    case 2:
    {
      const int vc_j = cc_j + dg;  // offset by difference in ghosts

      for (int n=0; n<=nu; ++n)
      for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
      {
        tar(n,cc_i) = 0.;

        const int vc_i = cc_i + dg;

        for (int dj=0; dj<H_SZ; ++dj)
        {
          const dtype lc_j = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-dj-1];

          const int vj_l = vc_j - dj;
          const int vj_r = vc_j + dj + 1;

          for (int di=0; di<H_SZ; ++di)
          {
            const dtype lc_ji = lc_j * InterpolateLagrangeUniform_opt<
              H_SZ
            >::coeff[H_SZ-di-1];

            const int vi_l = vc_i - di;
            const int vi_r = vc_i + di + 1;

            const dtype s_uu = src(n, 0, vj_r, vi_r);
            const dtype s_ul = src(n, 0, vj_r, vi_l);
            const dtype s_lu = src(n, 0, vj_l, vi_r);
            const dtype s_ll = src(n, 0, vj_l, vi_l);

            tar(n,cc_i) += lc_ji * FloatingPoint::sum_associative(
              s_uu, s_ll, s_lu, s_ul
            );
          }
        }
      }
      break;
    }
    default:
    {
      for (int n=0; n<=nu; ++n)
      for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
      {
        tar(n,cc_i) = 0.;

        const int vc_i = cc_i + dg;  // offset by difference in ghosts

        for (int di=0; di<H_SZ; ++di)
        {
          const dtype lc_i = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-di-1];

          const int vc_l = vc_i - di;
          const int vc_r = vc_i + di + 1;

          tar(n,cc_i) += lc_i * (
            src(n,0,0,vc_l) + src(n,0,0,vc_r)
          );
        }
      }
    }
  }
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::CC2VC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int vc_k,
  const int vc_j)
{
  const int nu = src.GetDim4()-1;

  switch (dim)
  {
    case 3:
    {
      const int cc_k = vc_k - dg - 1;  // offset by difference in ghosts
      const int cc_j = vc_j - dg - 1;

      for (int n=0; n<=nu; ++n)
      for (int vc_i=vc_il; vc_i<=vc_iu; ++vc_i)
      {
        tar(n,vc_i) = 0.;

        const int cc_i = vc_i - dg - 1;

        for (int dk=0; dk<H_SZ; ++dk)
        {
          const dtype lc_k = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-dk-1];

          const int ck_l = cc_k - dk;
          const int ck_r = cc_k + dk + 1;

          for (int dj=0; dj<H_SZ; ++dj)
          {
            const dtype lc_kj = lc_k * InterpolateLagrangeUniform_opt<
              H_SZ
            >::coeff[H_SZ-dj-1];

            const int cj_l = cc_j - dj;
            const int cj_r = cc_j + dj + 1;


            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_kji = lc_kj * InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int ci_l = cc_i - di;
              const int ci_r = cc_i + di + 1;

              const dtype s_rrr = src(n, ck_r, cj_r, ci_r);
              const dtype s_lrr = src(n, ck_l, cj_r, ci_r);
              const dtype s_rlr = src(n, ck_r, cj_l, ci_r);
              const dtype s_rrl = src(n, ck_r, cj_r, ci_l);

              const dtype s_llr = src(n, ck_l, cj_l, ci_r);
              const dtype s_rll = src(n, ck_r, cj_l, ci_l);
              const dtype s_lrl = src(n, ck_l, cj_r, ci_l);
              const dtype s_lll = src(n, ck_l, cj_l, ci_l);

              tar(n,vc_i) += lc_kji * FloatingPoint::sum_associative(
                  s_rrr, s_lll, s_rrl, s_llr,
                  s_lrl, s_rlr, s_lrr, s_rll
              );
            }
          }

        }
      }
      break;
    }
    case 2:
    {
      const int cc_j = vc_j - dg - 1;  // offset by difference in ghosts

      for (int n=0; n<=nu; ++n)
      for (int vc_i=vc_il; vc_i<=vc_iu; ++vc_i)
      {
        tar(n,vc_i) = 0.;

        const int cc_i = vc_i - dg - 1;

        for (int dj=0; dj<H_SZ; ++dj)
        {
          const dtype lc_j = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-dj-1];

          const int cj_l = cc_j - dj;
          const int cj_r = cc_j + dj + 1;


          for (int di=0; di<H_SZ; ++di)
          {
            const dtype lc_ji = lc_j * InterpolateLagrangeUniform_opt<
              H_SZ
            >::coeff[H_SZ-di-1];

            const int ci_l = cc_i - di;
            const int ci_r = cc_i + di + 1;

            const dtype s_uu = src(n, 0, cj_r, ci_r);
            const dtype s_ul = src(n, 0, cj_r, ci_l);
            const dtype s_lu = src(n, 0, cj_l, ci_r);
            const dtype s_ll = src(n, 0, cj_l, ci_l);

            tar(n,vc_i) += lc_ji * FloatingPoint::sum_associative(
              s_uu, s_ll, s_lu, s_ul
            );
          }
        }
      }
      break;
    }
    default:
    {
      for (int n=0; n<=nu; ++n)
      for (int vc_i=vc_il; vc_i<=vc_iu; ++vc_i)
      {
        tar(n,vc_i) = 0.;

        const int cc_i = vc_i - dg - 1;  // offset by difference in ghosts

        for (int di=0; di<H_SZ; ++di)
        {
          const dtype lc_i = InterpolateLagrangeUniform_opt<
            H_SZ
          >::coeff[H_SZ-di-1];

          const int cc_l = cc_i - di;
          const int cc_r = cc_i + di + 1;

          tar(n,vc_i) += lc_i * (
            src(n,0,0,cc_l) + src(n,0,0,cc_r)
          );
        }
      }
    }

  }
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::VC2FC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int dir,
  const int tr_k,
  const int tr_j)
{
  VC2FC(tar.array(), src.array(),
        dir,
        tr_k, tr_j);
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::CC2FC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int dir,
  const int tr_k,
  const int tr_j)
{
  CC2FC(tar.array(), src.array(),
        dir,
        tr_k, tr_j);
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::VC2FC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int dir,
  const int tr_k,
  const int tr_j)
{
  const int tr_il = (dir == 0) ? vc_il : cc_il;
  const int tr_iu = (dir == 0) ? vc_iu : cc_iu;

  VC2FC(
    tar, src,
    dir,
    tr_k, tr_j,
    tr_il, tr_iu
  );
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::CC2FC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int dir,
  const int tr_k,
  const int tr_j)
{
  const int tr_il = (dir == 0) ? vc_il : cc_il;
  const int tr_iu = (dir == 0) ? vc_iu : cc_iu;

  CC2FC(
    tar, src,
    dir,
    tr_k, tr_j,
    tr_il, tr_iu
  );
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::VC2FC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int dir,
  const int tr_k,
  const int tr_j,
  const int tr_il,
  const int tr_iu)
{
  VC2FC(tar.array(), src.array(),
        dir,
        tr_k, tr_j, tr_il, tr_iu);
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::CC2FC(
  AthenaTensor<       dtype, TSYM, DIM, NVAL> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL> & src,
  const int dir,
  const int tr_k,
  const int tr_j,
  const int tr_il,
  const int tr_iu)
{
  CC2FC(tar.array(), src.array(),
        dir,
        tr_k, tr_j, tr_il, tr_iu);
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::VC2FC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int dir,
  const int tr_k,
  const int tr_j,
  const int tr_il,
  const int tr_iu)
{
  const int nu = src.GetDim4()-1;

  switch (dim)
  {
    case 3:
    {
      switch (dir)
      {
        case 2:
        {
          // (VC,VC,VC) -> (VC,CC,CC)
          const int vc_j = tr_j + dg;  // offset by difference in ghosts

          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;

            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dj-1];

              const int vj_l = vc_j - dj;
              const int vj_r = vc_j + dj + 1;

              for (int di=0; di<H_SZ; ++di)
              {
                const dtype lc_ji = lc_j * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-di-1];

                const int vi_l = vc_i - di;
                const int vi_r = vc_i + di + 1;

                const dtype s_uu = src(n, tr_k, vj_r, vi_r);
                const dtype s_ul = src(n, tr_k, vj_r, vi_l);
                const dtype s_lu = src(n, tr_k, vj_l, vi_r);
                const dtype s_ll = src(n, tr_k, vj_l, vi_l);

                tar(n,cc_i) += lc_ji * FloatingPoint::sum_associative(
                  s_uu, s_ll, s_lu, s_ul
                );
              }
            }
          }
          break;
        }
        case 1:
        {
          // (VC,VC,VC) -> (CC,VC,CC)
          const int vc_k = tr_k + dg;  // offset by difference in ghosts

          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dk-1];

              const int vk_l = vc_k - dk;
              const int vk_r = vc_k + dk + 1;

              for (int di=0; di<H_SZ; ++di)
              {
                const dtype lc_ki = lc_k * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-di-1];

                const int vi_l = vc_i - di;
                const int vi_r = vc_i + di + 1;

                const dtype s_uu = src(n, vk_r, tr_j, vi_r);
                const dtype s_ul = src(n, vk_r, tr_j, vi_l);
                const dtype s_lu = src(n, vk_l, tr_j, vi_r);
                const dtype s_ll = src(n, vk_l, tr_j, vi_l);

                tar(n,cc_i) += lc_ki * FloatingPoint::sum_associative(
                  s_uu, s_ll, s_lu, s_ul
                );
              }
            }
          }
          break;
        }
        default:
        {
          // (VC,VC,VC) -> (CC,CC,VC)
          const int vc_k = tr_k + dg;  // offset by difference in ghosts

          for (int n=0; n<=nu; ++n)
          for (int vc_i=tr_il; vc_i<=tr_iu; ++vc_i)
          {
            tar(n,vc_i) = 0.;

            const int vc_j = tr_j + dg;

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dk-1];

              const int vk_l = vc_k - dk;
              const int vk_r = vc_k + dk + 1;

              for (int dj=0; dj<H_SZ; ++dj)
              {
                const dtype lc_kj = lc_k * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-dj-1];

                const int vj_l = vc_j - dj;
                const int vj_r = vc_j + dj + 1;

                const dtype s_uu = src(n, vk_r, vj_r, vc_i);
                const dtype s_ul = src(n, vk_r, vj_l, vc_i);
                const dtype s_lu = src(n, vk_l, vj_r, vc_i);
                const dtype s_ll = src(n, vk_l, vj_l, vc_i);

                tar(n,vc_i) += lc_kj * FloatingPoint::sum_associative(
                  s_uu, s_ll, s_lu, s_ul
                );
              }
            }
          }

        }
      }
      break;
    }
    case 2:
    {
      switch (dir)
      {
        case 1:
        {
          // (VC,VC) -> (VC,CC)
          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_i = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int vc_l = vc_i - di;
              const int vc_r = vc_i + di + 1;

              tar(n,cc_i) += lc_i * (
                src(n,0,tr_j,vc_l) + src(n,0,tr_j,vc_r)
              );
            }
          }
          break;
        }
        default:
        {
          // (VC,VC) -> (CC,VC)
          for (int n=0; n<=nu; ++n)
          for (int vc_i=tr_il; vc_i<=tr_iu; ++vc_i)
          {
            tar(n,vc_i) = 0.;

            const int vc_j = tr_j + dg;
            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dj-1];

              const int vj_l = vc_j - dj;
              const int vj_r = vc_j + dj + 1;

              tar(n,vc_i) += lc_j * (
                src(n,0,vj_l,vc_i) + src(n,0,vj_r,vc_i)
              );
            }
          }
        }
      }
      break;
    }
    default:
    {
      switch (dir)
      {
        // trivial copy in this case
        default:
        {
          for (int n=0; n<=nu; ++n)
          for (int tr_i=tr_il; tr_i<=tr_iu; ++tr_i)
          {
            tar(n,tr_i) = src(n,0,0,tr_i);
          }

        }
      }
    }
  }
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::CC2FC(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int dir,
  const int tr_k,
  const int tr_j,
  const int tr_il,
  const int tr_iu)
{
  const int nu = src.GetDim4()-1;

  switch (dim)
  {
    case 3:
    {
      switch (dir)
      {
        case 2:
        {
          // (CC,CC,CC) -> (VC,CC,CC)
          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int cc_k = tr_k - dg - 1;  // offset by difference in ghosts

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dk-1];

              const int ck_l = cc_k - dk;
              const int ck_r = cc_k + dk + 1;

              tar(n,cc_i) += lc_k * (
                src(n,ck_l,tr_j,cc_i) + src(n,ck_r,tr_j,cc_i)
              );
            }
          }
          break;
        }
        case 1:
        {
          // (CC,CC,CC) -> (CC,VC,CC)
          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int cc_j = tr_j - dg - 1;  // offset by difference in ghosts

            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dj-1];

              const int cj_l = cc_j - dj;
              const int cj_r = cc_j + dj + 1;

              tar(n,cc_i) += lc_j * (
                src(n,tr_k,cj_l,cc_i) + src(n,tr_k,cj_r,cc_i)
              );
            }
          }
          break;
        }
        default:
        {
          // (CC,CC,CC) -> (CC,CC,VC)
          for (int n=0; n<=nu; ++n)
          for (int vc_i=tr_il; vc_i<=tr_iu; ++vc_i)
          {
            tar(n,vc_i) = 0.;

            const int cc_i = vc_i - dg - 1;  // offset by difference in ghosts

            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_i = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int ci_l = cc_i - di;
              const int ci_r = cc_i + di + 1;

              tar(n,vc_i) += lc_i * (
                src(n,tr_k,tr_j,ci_l) + src(n,tr_k,tr_j,ci_r)
              );
            }
          }
        }
      }
      break;
    }
    case 2:
    {
      switch (dir)
      {
        case 1:
        {
          // (CC,CC) -> (VC,CC)
          for (int n=0; n<=nu; ++n)
          for (int cc_i=tr_il; cc_i<=tr_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int cc_j = tr_j - dg - 1;  // offset by difference in ghosts

            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dj-1];

              const int cj_l = cc_j - dj;
              const int cj_r = cc_j + dj + 1;

              tar(n,cc_i) += lc_j * (
                src(n,0,cj_l,cc_i) + src(n,0,cj_r,cc_i)
              );
            }
          }
          break;
        }
        default:
        {
          // (CC,CC) -> (CC,VC)
          for (int n=0; n<=nu; ++n)
          for (int vc_i=tr_il; vc_i<=tr_iu; ++vc_i)
          {
            tar(n,vc_i) = 0.;

            const int cc_i = vc_i - dg - 1;  // offset by difference in ghosts

            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_i = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int ci_l = cc_i - di;
              const int ci_r = cc_i + di + 1;

              tar(n,vc_i) += lc_i * (
                src(n,0,tr_j,ci_l) + src(n,0,tr_j,ci_r)
              );
            }
          }
        }
      }
      break;
    }
    default:
    {
      switch (dir)
      {
        default:
        {
          for (int n=0; n<=nu; ++n)
          for (int vc_i=tr_il; vc_i<=tr_iu; ++vc_i)
          {
            tar(n,vc_i) = 0.;

            const int cc_i = vc_i - dg - 1;  // offset by difference in ghosts

            for (int di=0; di<H_SZ; ++di)
            {
              const dtype lc_i = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-di-1];

              const int cc_l = cc_i - di;
              const int cc_r = cc_i + di + 1;

              tar(n,vc_i) += lc_i * (
                src(n,0,0,cc_l) + src(n,0,0,cc_r)
              );
            }
          }
        }
      }
    }
  }
}

template <typename dtype, int H_SZ>
template <TensorSymm TSYM, int DIM, int NVAL>
void InterpIntergrid<dtype, H_SZ>::VC2CC_D1(
  AthenaTensor<       dtype, TSYM, DIM, NVAL+1> & tar,
  const  AthenaTensor<dtype, TSYM, DIM, NVAL  > & src,
  const int dir,
  const int cc_k,
  const int cc_j)
{
  // valence changes with deriv, so need to reslice
  const int ix_slice = dir * (tar.ndof() / DIM);
  AthenaArray<dtype> tar_arr;
  tar_arr.InitWithShallowSlice(tar.array(), ix_slice, 1);

  VC2CC_D1(tar_arr, src.array(), dir, cc_k, cc_j);
}

template <typename dtype, int H_SZ>
void InterpIntergrid<dtype, H_SZ>::VC2CC_D1(
  AthenaArray<       dtype> & tar,
  const  AthenaArray<dtype> & src,
  const int dir,
  const int cc_k,
  const int cc_j)
{
  const int nu = src.GetDim4()-1;

  switch (dim)
  {
    case 3:
    {
      const int vc_k = cc_k + dg;  // offset by difference in ghosts
      const int vc_j = cc_j + dg;

      switch (dir)
      {
        case 2:
        {
          for (int n=0; n<=nu; ++n)
          for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateVC2DerCC_rev<
                1, H_SZ
              >::coeff[H_SZ-dk-1];

              const int vk_l = vc_k - dk;
              const int vk_r = vc_k + dk + 1;

              for (int dj=0; dj<H_SZ; ++dj)
              {
                const dtype lc_kj = lc_k * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-dj-1];

                const int vj_l = vc_j - dj;
                const int vj_r = vc_j + dj + 1;

                for (int di=0; di<H_SZ; ++di)
                {
                  const dtype lc_kji = lc_kj * InterpolateLagrangeUniform_opt<
                    H_SZ
                  >::coeff[H_SZ-di-1];

                  const int vi_l = vc_i - di;
                  const int vi_r = vc_i + di + 1;

                  const dtype s_rrr = src(n, vk_r, vj_r, vi_r);
                  const dtype s_lrr = src(n, vk_l, vj_r, vi_r);
                  const dtype s_rlr = src(n, vk_r, vj_l, vi_r);
                  const dtype s_rrl = src(n, vk_r, vj_r, vi_l);

                  const dtype s_llr = src(n, vk_l, vj_l, vi_r);
                  const dtype s_rll = src(n, vk_r, vj_l, vi_l);
                  const dtype s_lrl = src(n, vk_l, vj_r, vi_l);
                  const dtype s_lll = src(n, vk_l, vj_l, vi_l);

                  tar(n,cc_i) += lc_kji * FloatingPoint::sum_associative(
                    s_rrr, -s_lrr,
                    s_rrl, -s_lrl,
                    s_rll, -s_lll,
                    s_rlr, -s_llr
                  );

                }
              }

            }

            tar(n,cc_i) = tar(n,cc_i) * rds[2];
          }
          break;
        }
        case 1:
        {
          for (int n=0; n<=nu; ++n)
          for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dk-1];

              const int vk_l = vc_k - dk;
              const int vk_r = vc_k + dk + 1;

              for (int dj=0; dj<H_SZ; ++dj)
              {
                const dtype lc_kj = lc_k * InterpolateVC2DerCC_rev<
                  1, H_SZ
                >::coeff[H_SZ-dj-1];

                const int vj_l = vc_j - dj;
                const int vj_r = vc_j + dj + 1;

                for (int di=0; di<H_SZ; ++di)
                {
                  const dtype lc_kji = lc_kj * InterpolateLagrangeUniform_opt<
                    H_SZ
                  >::coeff[H_SZ-di-1];

                  const int vi_l = vc_i - di;
                  const int vi_r = vc_i + di + 1;

                  const dtype s_rrr = src(n, vk_r, vj_r, vi_r);
                  const dtype s_lrr = src(n, vk_l, vj_r, vi_r);
                  const dtype s_rlr = src(n, vk_r, vj_l, vi_r);
                  const dtype s_rrl = src(n, vk_r, vj_r, vi_l);

                  const dtype s_llr = src(n, vk_l, vj_l, vi_r);
                  const dtype s_rll = src(n, vk_r, vj_l, vi_l);
                  const dtype s_lrl = src(n, vk_l, vj_r, vi_l);
                  const dtype s_lll = src(n, vk_l, vj_l, vi_l);

                  tar(n,cc_i) += lc_kji * FloatingPoint::sum_associative(
                    s_lrl, -s_lll,
                    s_lrr, -s_llr,
                    s_rrr, -s_rlr,
                    s_rrl, -s_rll
                  );

                }
              }

            }

            tar(n,cc_i) = tar(n,cc_i) * rds[1];
          }
          break;
        }
        default:
        {
          for (int n=0; n<=nu; ++n)
          for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int dk=0; dk<H_SZ; ++dk)
            {
              const dtype lc_k = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dk-1];

              const int vk_l = vc_k - dk;
              const int vk_r = vc_k + dk + 1;

              for (int dj=0; dj<H_SZ; ++dj)
              {
                const dtype lc_kj = lc_k * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-dj-1];

                const int vj_l = vc_j - dj;
                const int vj_r = vc_j + dj + 1;

                for (int di=0; di<H_SZ; ++di)
                {
                  const dtype lc_kji = lc_kj * InterpolateVC2DerCC_rev<
                    1, H_SZ
                  >::coeff[H_SZ-di-1];

                  const int vi_l = vc_i - di;
                  const int vi_r = vc_i + di + 1;

                  const dtype s_rrr = src(n, vk_r, vj_r, vi_r);
                  const dtype s_lrr = src(n, vk_l, vj_r, vi_r);
                  const dtype s_rlr = src(n, vk_r, vj_l, vi_r);
                  const dtype s_rrl = src(n, vk_r, vj_r, vi_l);

                  const dtype s_llr = src(n, vk_l, vj_l, vi_r);
                  const dtype s_rll = src(n, vk_r, vj_l, vi_l);
                  const dtype s_lrl = src(n, vk_l, vj_r, vi_l);
                  const dtype s_lll = src(n, vk_l, vj_l, vi_l);

                  tar(n,cc_i) += lc_kji * FloatingPoint::sum_associative(
                    s_llr, -s_lll,
                    s_lrr, -s_lrl,
                    s_rrr, -s_rrl,
                    s_rlr, -s_rll
                  );

                }
              }

            }

            tar(n,cc_i) = tar(n,cc_i) * rds[0];
          }
        }
      }
      break;
    }
    case 2:
    {
      const int vc_j = cc_j + dg;  // offset by difference in ghosts

      switch (dir)
      {
        case 1:
        {
          for (int n=0; n<=nu; ++n)
          for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateVC2DerCC_rev<
                1, H_SZ
              >::coeff[H_SZ-dj-1];

              const int vj_l = vc_j - dj;
              const int vj_r = vc_j + dj + 1;

              for (int di=0; di<H_SZ; ++di)
              {
                const dtype lc_ji = lc_j * InterpolateLagrangeUniform_opt<
                  H_SZ
                >::coeff[H_SZ-di-1];

                const int vi_l = vc_i - di;
                const int vi_r = vc_i + di + 1;

                const dtype s_uu = src(n, 0, vj_r, vi_r);
                const dtype s_ul = src(n, 0, vj_r, vi_l);
                const dtype s_lu = src(n, 0, vj_l, vi_r);
                const dtype s_ll = src(n, 0, vj_l, vi_l);

                tar(n,cc_i) += lc_ji * FloatingPoint::sum_associative(
                  s_uu, -s_lu,
                  s_ul, -s_ll
                );
              }
            }

            tar(n,cc_i) = tar(n,cc_i) * rds[1];
          }
          break;
        }
        default:
        {
          for (int n=0; n<=nu; ++n)
          for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
          {
            tar(n,cc_i) = 0.;

            const int vc_i = cc_i + dg;  // offset by difference in ghosts

            for (int dj=0; dj<H_SZ; ++dj)
            {
              const dtype lc_j = InterpolateLagrangeUniform_opt<
                H_SZ
              >::coeff[H_SZ-dj-1];

              const int vj_l = vc_j - dj;
              const int vj_r = vc_j + dj + 1;

              for (int di=0; di<H_SZ; ++di)
              {
                const dtype lc_ji = lc_j * InterpolateVC2DerCC_rev<
                  1, H_SZ
                >::coeff[H_SZ-di-1];

                const int vi_l = vc_i - di;
                const int vi_r = vc_i + di + 1;

                const dtype s_uu = src(n, 0, vj_r, vi_r);
                const dtype s_ul = src(n, 0, vj_r, vi_l);
                const dtype s_lu = src(n, 0, vj_l, vi_r);
                const dtype s_ll = src(n, 0, vj_l, vi_l);

                tar(n,cc_i) += lc_ji * FloatingPoint::sum_associative(
                  s_lu, -s_ll,
                  s_uu, -s_ul
                );
              }
            }

            tar(n,cc_i) = tar(n,cc_i) * rds[0];
          }
        }
      }
      break;
    }
    default:
    {
      for (int n=0; n<=nu; ++n)
      for (int cc_i=cc_il; cc_i<=cc_iu; ++cc_i)
      {
        tar(n,cc_i) = 0.;

        const int vc_i = cc_i + dg;  // offset by difference in ghosts

        for (int di=0; di<H_SZ; ++di)
        {
          const dtype lc_i = InterpolateVC2DerCC_rev<
            1, H_SZ
          >::coeff[H_SZ-di-1];

          const int vc_l = vc_i - di;
          const int vc_r = vc_i + di + 1;

          tar(n,cc_i) += lc_i * (
            -src(n,0,0,vc_l) + src(n,0,0,vc_r)
          );
        }

        tar(n,cc_i) = tar(n,cc_i) * rds[0];
      }
    }
  }

}

} // namespace InterpIntergrid
// ============================================================================
