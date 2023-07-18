#ifndef INTERP_BARYCENTRIC_HPP_
#define INTERP_BARYCENTRIC_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file interp_univariate.hpp
//  \brief Collection of univariate interpolators

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// Construct Floater-Hormann weights on an input grid
//
// Parameters:
// -----------
// gr_s: Source grid with N + 1 entries
// k:    Weight index values (0,... , N)
// d:    FH blending parameter
template <typename T>
static T weight_FH(T * gr_s,
                   int N,
                   int k, int d)
{
  const int il = std::max(k-d, 0);
  const int iu = std::min(k, N-d);

  T wei = 0.;

  for(int i=il; i<=iu; ++i)
  {
    T wei_i = std::pow(-1, i);

    for(int s=i; s<=i+d; ++s)
    {
      T den = gr_s[k] - gr_s[s];
      den = den * (s != k) + (s == k);
      wei_i = wei_i / den;
    }

    wei += wei_i;
  }

  return wei;
}

// One-dimensional Floater-Hormann interpolation
//
// Parameters:
// -----------
// gr_t:  Target base-point to interpolate to
// gr_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// N:     Grid size parameter
// d:     FH blending parameter
// ng=0:  Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that gr_s is passed with first entry at gr_s[ng]
template<typename Tg, typename Tf>
static Tf interp_1d_FH(Tg gr_t, Tg * gr_s,
                       Tf * fcn_s,
                       int N, int d, int ng=0)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  for(int k=0; k<=N; ++k)
  {
    const Tg w_k  = weight_FH<Tg>(gr_s, N, k, d);
    const Tg rdx  = 1. / (gr_t - gr_s[k]);
    const Tg rw_k = w_k * rdx;

    num += rw_k * fcn_s[k+ng];
    den += rw_k;
  }

  return static_cast<Tf>(num / den);
}

// Two-dimensional Floater-Hormann interpolation
// Note:
//   Target node is assumed distinct from fundamental (source) nodes.
template<typename Tg, typename Tf>
static Tf interp_2d_FH(Tg x1_t, Tg x2_t,
                       Tg * x1_s, Tg * x2_s,
                       Tf * fcn_s,
                       int N_x1, int N_x2, int d, int ng=0)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight_FH<Tg>(x1_s, N_x1, k1, d);

  for(int k2=0; k2<=N_x2; ++k2)
  {
    const Tg w_k2  = weight_FH<Tg>(x2_s, N_x2, k2, d);
    const Tg rdx2  = 1. / (x2_t - x2_s[k2]);
    const Tg rw_k2 = w_k2 * rdx2;

    for(int k1=0; k1<=N_x1; ++k1)
    {
      const Tg w_k1  = arr_w_k1[k1];
      const Tg rdx1  = 1. / (x1_t - x1_s[k1]);
      const Tg rw_k1 = w_k1 * rdx1;

      const int ix = (k1+ng) + (N_x1+1+2*ng) * (k2+ng);

      num += rw_k1 * rw_k2 * fcn_s[ix];
      den += rw_k1 * rw_k2;
    }
  }

  delete[] arr_w_k1;

  return static_cast<Tf>(num / den);
}

// Three-dimensional Floater-Hormann interpolation
template<typename Tg, typename Tf>
static Tf interp_3d_FH(Tg x1_t, Tg x2_t, Tg x3_t,
                       Tg * x1_s, Tg * x2_s, Tg * x3_s,
                       Tf * fcn_s,
                       int N_x1, int N_x2, int N_x3, int d, int ng=0)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight_FH<Tg>(x1_s, N_x1, k1, d);

  Tg * arr_w_k2 = new Tg[N_x2 + 1];
  for(int k2=0; k2<=N_x2; ++k2)
    arr_w_k2[k2] = weight_FH<Tg>(x2_s, N_x2, k2, d);

  for(int k3=0; k3<=N_x3; ++k3)
  {
    const Tg w_k3  = weight_FH<Tg>(x3_s, N_x3, k3, d);
    const Tg rdx3  = 1. / (x3_t - x3_s[k3]);
    const Tg rw_k3 = w_k3 * rdx3;

    for(int k2=0; k2<=N_x2; ++k2)
    {
      const Tg w_k2  = arr_w_k2[k2];
      const Tg rdx2  = 1. / (x2_t - x2_s[k2]);
      const Tg rw_k2 = w_k2 * rdx2;

      for(int k1=0; k1<=N_x1; ++k1)
      {
        const Tg w_k1  = arr_w_k1[k1];
        const Tg rdx1  = 1. / (x1_t - x1_s[k1]);
        const Tg rw_k1 = w_k1 * rdx1;

        const int ix = (
          (k1+ng) + (N_x1+1+2*ng) * ((k2+ng) + (N_x2+1+2*ng) * (k3+ng))
        );

        num += rw_k1 * rw_k2 * rw_k3 * fcn_s[ix];
        den += rw_k1 * rw_k2 * rw_k3;
      }
    }
  }

  delete[] arr_w_k1;
  delete[] arr_w_k2;

  return static_cast<Tf>(num / den);
}

// Construct generalized Floater-Hormann weights on an input grid
// Note:
//   Target node is assumed distinct from fundamental (source) nodes.
// Ref(s): ...
template <typename T>
static T weight_GFH(T gr_t, T * gr_s, int N,
                    int k, int d, double gam)
{
  const int il = std::max(k-d, 0);
  const int iu = std::min(k, N-d);

  T wei = 0.;

  for(int i=il; i<=iu; ++i)
  {
    T wei_i = std::pow(-1, i * gam);

    for(int s=i; s<=i+d; ++s)
    {
      T den = (
        (gr_s[k] - gr_s[s]) * std::pow(gr_t - gr_s[s], gam-1.)
      );
      den = den * (s != k) + (s == k);
      wei_i = wei_i / den;
    }

    wei += wei_i;
  }

  return wei;
}

// One-dimensional generalized Floater-Hormann interpolation
// Note:
//   Target node is assumed distinct from fundamental (source) nodes.
template<typename Tg, typename Tf>
static Tf interp_1d_GFH(Tg gr_t, Tg * gr_s, Tf * fcn_s,
                        int N, int d, double gam, int ng=0)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  for(int k=0; k<=N; ++k)
  {
    const Tg w_k = weight_GFH<Tg>(gr_t, gr_s, N, k, d, gam);
    const Tg rdx  = 1. / (gr_t - gr_s[k]);
    const Tg rw_k = w_k * std::pow(rdx, gam);

    num += rw_k * fcn_s[k+ng];
    den += rw_k;
  }

  return static_cast<Tf>(num / den);
}

#endif