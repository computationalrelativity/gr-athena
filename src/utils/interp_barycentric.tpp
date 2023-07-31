// headers ====================================================================
// c / c++
#include <algorithm>

// numprox
#include "interp_barycentric.hpp"
// ============================================================================

// BD: does rewriting sums as simultaneously approaching center mitigate
//     asymetry at all?
// BD: is compensated summation here worth it?

namespace numprox { namespace interpolation {

// ============================================================================
namespace Floater_Hormann {
// ============================================================================

template <typename T>
static inline T weight(const T * const gr_s,
                       int Ns,
                       int k, int d)
{
  const int il = std::max(k-d, 0);
  const int iu = std::min(k, Ns-d);

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

template<typename Tg, typename Tf>
static Tf interp_1d(Tg x1_t, Tg * gr_s,
                    Tf * fcn_s,
                    int Ns, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns+1];
  for(int k1=0; k1<=Ns; ++k1)
  {
    const Tg w_k1 = weight<Tg>(gr_s, Ns, k1, d);
    const Tg rdx1 = 1. / (x1_t - gr_s[k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }
  // --------------------------------------------------------------------------

  for(int k1=0; k1<=Ns; ++k1)
  {
    num += rw_k1[k1] * fcn_s[k1+ng];
    den += rw_k1[k1];
  }

  delete[] rw_k1;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns_x1+1];
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_s, Ns_x1, k1, d);
    const Tg rdx1 = 1. / (x1_t - x1_s[k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }

  Tg * rw_k2 = new Tg[Ns_x2+1];
  for(int k2=0; k2<=Ns_x2; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_s, Ns_x2, k2, d);
    const Tg rdx2 = 1. / (x2_t - x2_s[k2]);
    rw_k2[k2] = w_k2 * rdx2;
  }
  // --------------------------------------------------------------------------

  for(int k2=0; k2<=Ns_x2; ++k2)
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const int ix = (k1+ng) + (Ns_x1+1+2*ng) * (k2+ng);

    num += rw_k1[k1]*rw_k2[k2]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2];
  }

  delete[] rw_k1;
  delete[] rw_k2;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int Ns_x3, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns_x1+1];
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_s, Ns_x1, k1, d);
    const Tg rdx1 = 1. / (x1_t - x1_s[k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }

  Tg * rw_k2 = new Tg[Ns_x2+1];
  for(int k2=0; k2<=Ns_x2; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_s, Ns_x2, k2, d);
    const Tg rdx2 = 1. / (x2_t - x2_s[k2]);
    rw_k2[k2] = w_k2 * rdx2;
  }

  Tg * rw_k3 = new Tg[Ns_x3+1];
  for(int k3=0; k3<=Ns_x3; ++k3)
  {
    const Tg w_k3 = weight<Tg>(x3_s, Ns_x3, k3, d);
    const Tg rdx3 = 1. / (x3_t - x3_s[k3]);
    rw_k3[k3] = w_k3 * rdx3;
  }
  // --------------------------------------------------------------------------

  for(int k3=0; k3<=Ns_x3; ++k3)
  for(int k2=0; k2<=Ns_x2; ++k2)
  for(int k1=0; k1<=Ns_x1; ++k1)
  {

    const int ix = (
      (k1+ng) + (Ns_x1+1+2*ng) * ((k2+ng) + (Ns_x2+1+2*ng) * (k3+ng))
    );

    num += rw_k1[k1]*rw_k2[k2]*rw_k3[k3]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2]*rw_k3[k3];
  }

  delete[] rw_k1;
  delete[] rw_k2;
  delete[] rw_k3;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_1d(Tg x1_t, Tg * x1_s,
                       Tf * fcn_s,
                       int Ns, int d, int W, int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(&x1_s[ix_il], W-1, k1, d);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }
  // --------------------------------------------------------------------------

  for(int k1=0; k1<=W-1; ++k1)
  {
    num += rw_k1[k1] * fcn_s[k1+ng+ix_il];
    den += rw_k1[k1];
  }

  delete[] rw_k1;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_2d(Tg x1_t, Tg x2_t,
                       Tg * x1_s, Tg * x2_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2,
                       int d, int W, int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;
  size_t ix_jn, ix_jl, ix_ju;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns_x1,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_jn,
    &ix_jl,
    &ix_ju,
    x2_t,
    x2_s,
    Ns_x2,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(&x1_s[ix_il], W-1, k1, d);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }

  Tg * rw_k2 = new Tg[W];
  for(int k2=0; k2<=W-1; ++k2)
  {
    const Tg w_k2 = weight<Tg>(&x2_s[ix_jl], W-1, k2, d);
    const Tg rdx2 = 1. / (x2_t - x2_s[ix_jl+k2]);
    rw_k2[k2] = w_k2 * rdx2;
  }
  // --------------------------------------------------------------------------

  for(int k2=0; k2<=W-1; ++k2)
  for(int k1=0; k1<=W-1; ++k1)
  {
    const int ix = (k1+ng+ix_il) + (Ns_x1+1+2*ng) * (k2+ng+ix_jl);

    num += rw_k1[k1] * rw_k2[k2] * fcn_s[ix];
    den += rw_k1[k1] * rw_k2[k2];
  }

  delete[] rw_k1;
  delete[] rw_k2;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                       Tg * x1_s, Tg * x2_s, Tg * x3_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2, int Ns_x3,
                       int d, int W, int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;
  size_t ix_jn, ix_jl, ix_ju;
  size_t ix_kn, ix_kl, ix_ku;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns_x1,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_jn,
    &ix_jl,
    &ix_ju,
    x2_t,
    x2_s,
    Ns_x2,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_kn,
    &ix_kl,
    &ix_ku,
    x3_t,
    x3_s,
    Ns_x3,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(&x1_s[ix_il], W-1, k1, d);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * rdx1;
  }

  Tg * rw_k2 = new Tg[W];
  for(int k2=0; k2<=W-1; ++k2)
  {
    const Tg w_k2 = weight<Tg>(&x2_s[ix_jl], W-1, k2, d);
    const Tg rdx2 = 1. / (x2_t - x2_s[ix_jl+k2]);
    rw_k2[k2] = w_k2 * rdx2;
  }

  Tg * rw_k3 = new Tg[W];
  for(int k3=0; k3<=W-1; ++k3)
  {
    const Tg w_k3 = weight<Tg>(&x3_s[ix_kl], W-1, k3, d);
    const Tg rdx3 = 1. / (x3_t - x3_s[ix_kl+k3]);
    rw_k3[k3] = w_k3 * rdx3;
  }
  // --------------------------------------------------------------------------

  for(int k3=0; k3<=W-1; ++k3)
  for(int k2=0; k2<=W-1; ++k2)
  for(int k1=0; k1<=W-1; ++k1)
  {
    const int ix = (
      (k1+ng+ix_il) + (Ns_x1+1+2*ng) *
                      ((k2+ng+ix_jl) +
                       (Ns_x2+1+2*ng) * (k3+ng+ix_kl))
    );

    num += rw_k1[k1]*rw_k2[k2]*rw_k3[k3]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2]*rw_k3[k3];
  }

  delete[] rw_k1;
  delete[] rw_k2;
  delete[] rw_k3;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static void D1(
  const Tg * const x1_s,
  const Tf * const fcn_s,
  Tf * d1_fcn_t,
  const int Ns,
  const int d,
  const int ng)
{
  // construct weights
  Tg * wei = new Tg[Ns+1];
  Tg * r1d_x1_wei = new Tg[(Ns+1)*(Ns+1)];

  for(int k=0; k<=Ns; ++k)
    wei[k] = weight(x1_s, Ns, k, d);

  for(int k=0; k<=Ns; ++k)
  for(int i=0; i<=Ns; ++i)
  {
    const int ix = k+(Ns+1)*i;
    // optimize out branch (in sum i != j)
    const Tg den = (i != k) * (x1_s[k] - x1_s[i]) + (i == k);

    r1d_x1_wei[ix] = -wei[k] / wei[i] * 1. / den;
  }

  // compute derivatives
  numprox::interpolation::impl::_D1(fcn_s,
                                    r1d_x1_wei,
                                    d1_fcn_t,
                                    Ns,
                                    ng);

  delete[] wei;
  delete[] r1d_x1_wei;
}

template<typename Tg, typename Tf>
static void D1(
  const Tg * const x1_t,
  const Tg * const x1_s,
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  Tf * d1_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int d,
  const int ng_t,
  const int ng_s)
{
  // construct weights
  Tg * wei = new Tg[Ns_x1+1];
  Tg * s1d_x1_wei = new Tg[Nt_x1+1];
  Tg * r2d_x1_wei = new Tg[(Nt_x1+1)*(Ns_x1+1)];

  for(int k=0; k<=Ns_x1; ++k)
    wei[k] = weight(x1_s, Ns_x1, k, d);

  for(int i=0; i<=Nt_x1; ++i)
  {
    s1d_x1_wei[i] = 0;

    for(int k=0; k<=Ns_x1; ++k)
    {
      // optimize out branch (in sum i != j)
      const Tg rden = 1. / (x1_t[i] - x1_s[k]);

      s1d_x1_wei[i] += wei[k] * rden;

      const int ix = k + (Ns_x1+1)*i;
      r2d_x1_wei[ix] = wei[k] * rden * (-rden);
    }

    s1d_x1_wei[i] = 1. / s1d_x1_wei[i];
  }

  for(int i=0; i<=Nt_x1; ++i)
  for(int k=0; k<=Ns_x1; ++k)
  {
    const int ix = k + (Ns_x1+1)*i;
    r2d_x1_wei[ix] = r2d_x1_wei[ix] * s1d_x1_wei[i];
  }

  numprox::interpolation::impl::_D1(
    fcn_t,
    fcn_s,
    r2d_x1_wei,
    d1_fcn_t,
    Nt_x1,
    Ns_x1,
    ng_t,
    ng_s);

  delete[] wei;
  delete[] s1d_x1_wei;
  delete[] r2d_x1_wei;
}

template<typename Tg, typename Tf>
static void D2(
  const Tg * const x1_s,
  const Tf * const fcn_s,
  const Tf * const d1_fcn_s,
  Tf * d2_fcn_t,
  const int Ns,
  const int d,
  const int ng)
{
  // construct weights
  Tg * wei = new Tg[Ns+1];
  Tg * r1d_x1 = new Tg[(Ns+1)*(Ns+1)];
  Tg * r2d_x1_wei = new Tg[(Ns+1)*(Ns+1)];

  for(int k=0; k<=Ns; ++k)
    wei[k] = weight(x1_s, Ns, k, d);

  for(int k=0; k<=Ns; ++k)
  for(int i=0; i<=Ns; ++i)
  {
    const int ix = k+(Ns+1)*i;
    // optimize out branch (in sum i != j)
    const Tg den = (i != k) * (x1_s[k] - x1_s[i]) + (i == k);

    r1d_x1[ix] = 1. / den * (i != k);
    r2d_x1_wei[ix] = -2 * wei[k] / wei[i] * r1d_x1[ix];
  }

  // compute derivatives
  numprox::interpolation::impl::_D2(fcn_s,
                                    d1_fcn_s,
                                    r1d_x1,
                                    r2d_x1_wei,
                                    d2_fcn_t,
                                    Ns,
                                    ng);

  delete[] wei;
  delete[] r1d_x1;
  delete[] r2d_x1_wei;
}

template<typename Tg, typename Tf>
static void D2(
  const Tg * const x1_t,
  const Tg * const x1_s,
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  const Tf * const d1_fcn_t,
  const Tf * const d1_fcn_s,
  Tf * d2_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int d,
  const int ng_t,
  const int ng_s)
{
  // construct weights
  Tg * wei = new Tg[Ns_x1+1];
  Tg * s1d_x1_wei = new Tg[Nt_x1+1];
  Tg * r1d_x1 = new Tg[(Nt_x1+1)*(Ns_x1+1)];
  Tg * r2d_x1_wei = new Tg[(Nt_x1+1)*(Ns_x1+1)];

  for(int k=0; k<=Ns_x1; ++k)
    wei[k] = weight(x1_s, Ns_x1, k, d);

  for(int i=0; i<=Nt_x1; ++i)
  {
    s1d_x1_wei[i] = 0;

    for(int k=0; k<=Ns_x1; ++k)
    {
      const int ix = k + (Ns_x1+1)*i;

      // optimize out branch (in sum i != j)
      const Tg rden = 1. / (x1_t[i] - x1_s[k]);

      s1d_x1_wei[i] += wei[k] * rden;

      r1d_x1[ix] = -rden;
      r2d_x1_wei[ix] = wei[k] * rden * r1d_x1[ix];
    }

    s1d_x1_wei[i] = 1. / s1d_x1_wei[i];
  }

  for(int i=0; i<=Nt_x1; ++i)
  for(int k=0; k<=Ns_x1; ++k)
  {
    const int ix = k + (Ns_x1+1)*i;
    r2d_x1_wei[ix] = 2 * r2d_x1_wei[ix] * s1d_x1_wei[i];
  }

  // compute derivatives
  numprox::interpolation::impl::_D2(fcn_t,
                                    fcn_s,
                                    d1_fcn_t,
                                    r1d_x1,
                                    r2d_x1_wei,
                                    d2_fcn_t,
                                    Nt_x1,
                                    Ns_x1,
                                    ng_t,
                                    ng_s);

  delete[] wei;
  delete[] s1d_x1_wei;
  delete[] r1d_x1;
  delete[] r2d_x1_wei;
}

// classes --------------------------------------------------------------------

// ctor (1d) ------------------------------------------------------------------

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(
  Tg* x1_t, Tg* x1_s,
  const int Nt_x1, const int Ns_x1,
  const int d, const int ng_t, const int ng_s)
  : ndim {1}
  , Nt_x1 {Nt_x1}
  , Nt_x2 {1}
  , Nt_x3 {1}
  , Ns_x1 {Ns_x1}
  , Ns_x2 {1}
  , Ns_x3 {1}
  , d {d}
  , W {d+1}
  , ng_t {ng_t}
  , ng_s {ng_s}
  , rwei_x1    {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
  , rwei_nn_x1 {new Tg[W*(Nt_x1+1)]}
  , ix_nn_x1   {new size_t[Nt_x1+1]}
{
  precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
  precompute_nn_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_nn_x1, ix_nn_x1);
}

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x1_s,
                             const int Nt_x1, const int Ns_x1,
                             const int d,
                             const int ng)
  : interp_nd(x1_t, x1_s, Nt_x1, Ns_x1, d, ng, ng)
{ }

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x1_s,
                             const int Nt_x1, const int Ns_x1,
                             const int d)
  : interp_nd(x1_t, x1_s, Nt_x1, Ns_x1, d, 0, 0)
{ }

// ctor (2d) ------------------------------------------------------------------

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t,
                             Tg* x1_s, Tg* x2_s,
                             const int Nt_x1, const int Nt_x2,
                             const int Ns_x1, const int Ns_x2,
                             const int d,
                             const int ng_t,
                             const int ng_s)
  : ndim {2}
  , Nt_x1 {Nt_x1}
  , Nt_x2 {Nt_x2}
  , Nt_x3 {1}
  , Ns_x1 {Ns_x1}
  , Ns_x2 {Ns_x2}
  , Ns_x3 {1}
  , d {d}
  , W {d+1}
  , ng_t {ng_t}
  , ng_s {ng_s}
  , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
  , rwei_x2 {new Tg[(Ns_x2+1)*(Nt_x2+1)]}
  , rwei_nn_x1 {new Tg[W*(Nt_x1+1)]}
  , rwei_nn_x2 {new Tg[W*(Nt_x2+1)]}
  , ix_nn_x1   {new size_t[Nt_x1+1]}
  , ix_nn_x2   {new size_t[Nt_x2+1]}
{
  precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
  precompute_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_x2);

  precompute_nn_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_nn_x1, ix_nn_x1);
  precompute_nn_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_nn_x2, ix_nn_x2);
}

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t,
                             Tg* x1_s, Tg* x2_s,
                             const int Nt_x1, const int Nt_x2,
                             const int Ns_x1, const int Ns_x2,
                             const int d,
                             const int ng)
  : interp_nd(x1_t, x2_t,
              x1_s, x2_s,
              Nt_x1, Nt_x2,
              Ns_x1, Ns_x2,
              d, ng, ng)
{ }

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t,
                             Tg* x1_s, Tg* x2_s,
                             const int Nt_x1, const int Nt_x2,
                             const int Ns_x1, const int Ns_x2,
                             const int d)
  : interp_nd(x1_t, x2_t,
              x1_s, x2_s,
              Nt_x1, Nt_x2,
              Ns_x1, Ns_x2,
              d, 0, 0)
{ }

// ctor (3d) ------------------------------------------------------------------

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
                             Tg* x1_s, Tg* x2_s, Tg* x3_s,
                             const int Nt_x1,
                             const int Nt_x2,
                             const int Nt_x3,
                             const int Ns_x1,
                             const int Ns_x2,
                             const int Ns_x3,
                             const int d,
                             const int ng_t,
                             const int ng_s)
  : ndim {3}
  , Nt_x1 {Nt_x1}
  , Nt_x2 {Nt_x2}
  , Nt_x3 {Nt_x3}
  , Ns_x1 {Ns_x1}
  , Ns_x2 {Ns_x2}
  , Ns_x3 {Ns_x3}
  , d {d}
  , W {d+1}
  , ng_t {ng_t}
  , ng_s {ng_s}
  , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
  , rwei_x2 {new Tg[(Ns_x2+1)*(Nt_x2+1)]}
  , rwei_x3 {new Tg[(Ns_x3+1)*(Nt_x3+1)]}
  , rwei_nn_x1 {new Tg[W*(Nt_x1+1)]}
  , rwei_nn_x2 {new Tg[W*(Nt_x2+1)]}
  , rwei_nn_x3 {new Tg[W*(Nt_x3+1)]}
  , ix_nn_x1   {new size_t[Nt_x1+1]}
  , ix_nn_x2   {new size_t[Nt_x2+1]}
  , ix_nn_x3   {new size_t[Nt_x3+1]}
{
  precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
  precompute_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_x2);
  precompute_weights(x3_t, x3_s, Nt_x3, Ns_x3, rwei_x3);

  precompute_nn_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_nn_x1, ix_nn_x1);
  precompute_nn_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_nn_x2, ix_nn_x2);
  precompute_nn_weights(x3_t, x3_s, Nt_x3, Ns_x3, rwei_nn_x3, ix_nn_x3);
}

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
                             Tg* x1_s, Tg* x2_s, Tg* x3_s,
                             const int Nt_x1,
                             const int Nt_x2,
                             const int Nt_x3,
                             const int Ns_x1,
                             const int Ns_x2,
                             const int Ns_x3,
                             const int d,
                             const int ng)
  : interp_nd(x1_t, x2_t, x3_t,
              x1_s, x2_s, x3_s,
              Nt_x1, Nt_x2, Nt_x3,
              Ns_x1, Ns_x2, Ns_x3,
              d, ng, ng)
{ }

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
                             Tg* x1_s, Tg* x2_s, Tg* x3_s,
                             const int Nt_x1,
                             const int Nt_x2,
                             const int Nt_x3,
                             const int Ns_x1,
                             const int Ns_x2,
                             const int Ns_x3,
                             const int d)
  : interp_nd(x1_t, x2_t, x3_t,
              x1_s, x2_s, x3_s,
              Nt_x1, Nt_x2, Nt_x3,
              Ns_x1, Ns_x2, Ns_x3,
              d, 0, 0)
{ }


// dtor -----------------------------------------------------------------------

template <typename Tg, typename Tf>
interp_nd<Tg, Tf>::~interp_nd()
{
  switch (ndim)
  {
    case 3:
      delete[] rwei_x3;
      delete[] rwei_x2;
      delete[] rwei_x1;
      delete[] rwei_nn_x3;
      delete[] rwei_nn_x2;
      delete[] rwei_nn_x1;
      delete[] ix_nn_x3;
      delete[] ix_nn_x2;
      delete[] ix_nn_x1;
      break;
    case 2:
      delete[] rwei_x2;
      delete[] rwei_x1;
      delete[] rwei_nn_x2;
      delete[] rwei_nn_x1;
      delete[] ix_nn_x2;
      delete[] ix_nn_x1;
      break;
    default:
      delete[] rwei_x1;
      delete[] rwei_nn_x1;
      delete[] ix_nn_x1;
  }

}

// public method details ------------------------------------------------------

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval(Tf* fcn_t, const Tf* const fcn_s)
{
  switch (ndim)
  {
    case 3:
      eval(fcn_t, fcn_s, 0, Nt_x1, 0, Nt_x2, 0, Nt_x3);
      break;
    case 2:
      eval(fcn_t, fcn_s, 0, Nt_x1, 0, Nt_x2);
      break;
    default:  // ndim == 1
      eval(fcn_t, fcn_s, 0, Nt_x1);
      break;

  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval(Tf* fcn_t, const Tf* const fcn_s,
                                    const int il_t, const int iu_t,
                                    const int jl_t, const int ju_t,
                                    const int kl_t, const int ku_t)
{
  for(int i3=kl_t; i3<=ku_t; ++i3)
  for(int i2=jl_t; i2<=ju_t; ++i2)
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    // if((NGHOST <= i1) && (i1 <= Nt_x1 - NGHOST + 1) &&
    //    (NGHOST <= i2) && (i2 <= Nt_x2 - NGHOST + 1) &&
    //    (NGHOST <= i3) && (i3 <= Nt_x3 - NGHOST + 1))
    //   continue;

    Tr num = 0, den = 0;

    for(int k3=0; k3<=Ns_x3; ++k3)
    for(int k2=0; k2<=Ns_x2; ++k2)
    for(int k1=0; k1<=Ns_x1; ++k1)
    {
      // k1 is fastest running index
      const int ix_k1 = k1 + (Ns_x1+1)*i1;
      const int ix_k2 = k2 + (Ns_x2+1)*i2;
      const int ix_k3 = k3 + (Ns_x3+1)*i3;

      const int fs_ix = (
        (k1+ng_s) +
        (Ns_x1+1+2*ng_s) * ((k2+ng_s) + (Ns_x2+1+2*ng_s) * (k3+ng_s))
      );

      const Tg prwei_k1k2k3 = (
        rwei_x1[ix_k1] *
        rwei_x2[ix_k2] *
        rwei_x3[ix_k3]
      );

      num += prwei_k1k2k3 * fcn_s[fs_ix];
      den += prwei_k1k2k3;
    }

    const int ft_ix = (
      (i1+ng_t) +
      (Nt_x1+1+2*ng_t) * ((i2+ng_t) + (Nt_x2+1+2*ng_t) * (i3+ng_t))
    );

    fcn_t[ft_ix] = num / den;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval(Tf* fcn_t, const Tf* const fcn_s,
                                    const int il_t, const int iu_t,
                                    const int jl_t, const int ju_t)
{
  for(int i2=jl_t; i2<=ju_t; ++i2)
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    Tr num = 0, den = 0;

    for(int k2=0; k2<=Ns_x2; ++k2)
    for(int k1=0; k1<=Ns_x1; ++k1)
    {
      // k1 is fastest running index
      const int ix_k1 = k1 + (Ns_x1+1)*i1;
      const int ix_k2 = k2 + (Ns_x2+1)*i2;

      const int fs_ix = (k1+ng_s) + (Ns_x1+1+2*ng_s) * (k2+ng_s);

      const Tg prwei_k1k2 = rwei_x1[ix_k1] * rwei_x2[ix_k2];

      num += prwei_k1k2 * fcn_s[fs_ix];
      den += prwei_k1k2;
    }

    const int ft_ix = (i1+ng_t) + (Nt_x1+1+2*ng_t) * (i2+ng_t);
    fcn_t[ft_ix] = num / den;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval(Tf* fcn_t, const Tf* const fcn_s,
                                    const int il_t, const int iu_t)
{
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    Tr num = 0, den = 0;

    for(int k1=0; k1<=Ns_x1; ++k1)
    {
      // k1 is fastest running index
      const int ix = k1 + (Ns_x1+1)*i1;
      num += rwei_x1[ix] * fcn_s[k1+ng_s];
      den += rwei_x1[ix];
    }

    fcn_t[i1+ng_t] = num / den;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval_nn(Tf* fcn_t, const Tf* const fcn_s)
{
  switch (ndim)
  {
    case 3:
      eval_nn(fcn_t, fcn_s, 0, Nt_x1, 0, Nt_x2, 0, Nt_x3);
      break;
    case 2:
      eval_nn(fcn_t, fcn_s, 0, Nt_x1, 0, Nt_x2);
      break;
    default:  // ndim == 1
      eval_nn(fcn_t, fcn_s, 0, Nt_x1);
      break;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                                       const int il_t, const int iu_t,
                                       const int jl_t, const int ju_t,
                                       const int kl_t, const int ku_t)
{
  for(int i3=kl_t; i3<=ku_t; ++i3)
  for(int i2=jl_t; i2<=ju_t; ++i2)
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    Tr num = 0, den = 0;

    for(int k3=0; k3<=W-1; ++k3)
    for(int k2=0; k2<=W-1; ++k2)
    for(int k1=0; k1<=W-1; ++k1)
    {
      // k1 is fastest running index
      const int ix_k1 = k1 + W*i1;
      const int ix_k2 = k2 + W*i2;
      const int ix_k3 = k3 + W*i3;

      const int fs_ix = (
        (k1+ng_s+ix_nn_x1[i1]) +
        (Ns_x1+1+2*ng_s) * ((k2+ng_s+ix_nn_x2[i2]) +
                            (Ns_x2+1+2*ng_s) *
                            (k3+ng_s+ix_nn_x3[i3]))
      );

      const Tg prwei_k1k2k3 = (
        rwei_nn_x1[ix_k1] *
        rwei_nn_x2[ix_k2] *
        rwei_nn_x3[ix_k3]
      );

      num += prwei_k1k2k3 * fcn_s[fs_ix];
      den += prwei_k1k2k3;
    }

    const int ft_ix = (
        (i1+ng_t) +
        (Nt_x1+1+2*ng_t) * ((i2+ng_t) + (Nt_x2+1+2*ng_t) * (i3+ng_t))
    );

    fcn_t[ft_ix] = num / den;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                                       const int il_t, const int iu_t,
                                       const int jl_t, const int ju_t)
{
  for(int i2=jl_t; i2<=ju_t; ++i2)
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    Tr num = 0, den = 0;

    for(int k2=0; k2<=W-1; ++k2)
    for(int k1=0; k1<=W-1; ++k1)
    {
      // k1 is fastest running index
      const int ix_k1 = k1 + W*i1;
      const int ix_k2 = k2 + W*i2;

      const int fs_ix = (
        (k1+ng_s+ix_nn_x1[i1]) +
        (Ns_x1+1+2*ng_s) * (k2+ng_s+ix_nn_x2[i2])
      );

      const Tg prwei_k1k2 = rwei_nn_x1[ix_k1]*rwei_nn_x2[ix_k2];

      num += prwei_k1k2 * fcn_s[fs_ix];
      den += prwei_k1k2;
    }

    const int ft_ix = ((i1+ng_t) +
      (Nt_x1+1+2*ng_t) * (i2+ng_t)
    );
    fcn_t[ft_ix] = num / den;
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                                       const int il_t, const int iu_t)
{
  for(int i1=il_t; i1<=iu_t; ++i1)
  {
    Tr num = 0, den = 0;

    for(int k1=0; k1<=W-1; ++k1)
    {
      // k1 is fastest running index
      const int ix = k1 + W*i1;
      num += rwei_nn_x1[ix] * fcn_s[k1+ng_s+ix_nn_x1[i1]];
      den += rwei_nn_x1[ix];
    }

    fcn_t[i1+ng_t] = num / den;
  }
}

// private method details -----------------------------------------------------

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::precompute_weights(
      Tg* x_t, Tg* x_s,
      const int Nt, const int Ns,
      Tg* rwei)
{
  for(int k=0; k<=Ns; ++k)
  {
    const Tg w_k = weight(x_s, Ns, k, d);

    for(int i=0; i<=Nt; ++i)
    {
      const Tg rdx = 1. / (x_t[i] - x_s[k]);
      const int ix = k + (Ns+1)*i;
      rwei[ix] = w_k * rdx;
    }
  }
}

template <typename Tg, typename Tf>
inline void interp_nd<Tg, Tf>::precompute_nn_weights(
      Tg* x_t, Tg* x_s,
      const int Nt, const int Ns,
      Tg* rwei_nn, size_t* ix_nn)
{
  for(int i=0; i<=Nt; ++i)
  {
    size_t ix_n, ix_l, ix_u;

    numprox::interpolation::shared::idx_range_nearest_point(
      &ix_n,
      &ix_l,
      &ix_u,
      x_t[i],
      x_s,
      Ns,
      W,
      ne
    );
    ix_nn[i] = ix_l;

    for(int k=0; k<=W-1; ++k)
    {
      const Tg w_k = weight(&x_s[ix_l], W-1, k, d);
      const Tg rdx = 1. / (x_t[i] - x_s[ix_l+k]);
      const int ix = k + W*i;  // k is fastest idx
      rwei_nn[ix] = w_k * rdx;
    }
  }
}

// ============================================================================
}
// ============================================================================

// ============================================================================
namespace Floater_Hormann_generalized {
// ============================================================================

template <typename T>
static inline T weight(const T gr_t, const T * const gr_s, int Ns,
                       int k, int d, double gam)
{
  const int il = std::max(k-d, 0);
  const int iu = std::min(k, Ns-d);

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

template<typename Tg, typename Tf>
static Tf interp_1d(Tg gr_t, Tg * gr_s, Tf * fcn_s,
                    int Ns, int d, double gam, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns+1];
  for(int k1=0; k1<=Ns; ++k1)
  {
    const Tg w_k1 = weight<Tg>(gr_t, gr_s, Ns, k1, d, gam);
    const Tg rdx1 = 1. / (gr_t - gr_s[k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }
  // --------------------------------------------------------------------------

  for(int k1=0; k1<=Ns; ++k1)
  {
    num += rw_k1[k1] * fcn_s[k1+ng];
    den += rw_k1[k1];
  }

  delete[] rw_k1;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int d, double gam, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns_x1+1];
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_t, x1_s, Ns_x1, k1, d, gam);
    const Tg rdx1 = 1. / (x1_t - x1_s[k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }

  Tg * rw_k2 = new Tg[Ns_x2+1];
  for(int k2=0; k2<=Ns_x2; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_t, x2_s, Ns_x2, k2, d, gam);
    const Tg rdx2 = 1. / (x2_t - x2_s[k2]);
    rw_k2[k2] = w_k2 * std::pow(rdx2, gam);
  }
  // --------------------------------------------------------------------------

  for(int k2=0; k2<=Ns_x2; ++k2)
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const int ix = (k1+ng) + (Ns_x1+1+2*ng) * (k2+ng);

    num += rw_k1[k1]*rw_k2[k2]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2];
  }

  delete[] rw_k1;
  delete[] rw_k2;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int Ns_x3, int d, double gam,
                    int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[Ns_x1+1];
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_t, x1_s, Ns_x1, k1, d, gam);
    const Tg rdx1 = 1. / (x1_t - x1_s[k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }

  Tg * rw_k2 = new Tg[Ns_x2+1];
  for(int k2=0; k2<=Ns_x2; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_t, x2_s, Ns_x2, k2, d, gam);
    const Tg rdx2 = 1. / (x2_t - x2_s[k2]);
    rw_k2[k2] = w_k2 * std::pow(rdx2, gam);
  }

  Tg * rw_k3 = new Tg[Ns_x3+1];
  for(int k3=0; k3<=Ns_x3; ++k3)
  {
    const Tg w_k3 = weight<Tg>(x3_t, x3_s, Ns_x3, k3, d, gam);
    const Tg rdx3 = 1. / (x3_t - x3_s[k3]);
    rw_k3[k3] = w_k3 * std::pow(rdx3, gam);
  }
  // --------------------------------------------------------------------------

  for(int k3=0; k3<=Ns_x3; ++k3)
  for(int k2=0; k2<=Ns_x2; ++k2)
  for(int k1=0; k1<=Ns_x1; ++k1)
  {
    const int ix = (
      (k1+ng) + (Ns_x1+1+2*ng) * ((k2+ng) + (Ns_x2+1+2*ng) * (k3+ng))
    );

    num += rw_k1[k1]*rw_k2[k2]*rw_k3[k3]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2]*rw_k3[k3];
  }

  delete[] rw_k1;
  delete[] rw_k2;
  delete[] rw_k3;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_1d(Tg x1_t, Tg * x1_s,
                       Tf * fcn_s,
                       int Ns, int d, double gam,
                       int W, int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_t, &x1_s[ix_il], W-1, k1, d, gam);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }
  // --------------------------------------------------------------------------

  for(int k1=0; k1<=W-1; ++k1)
  {
    num += rw_k1[k1] * fcn_s[k1+ng+ix_il];
    den += rw_k1[k1];
  }

  delete[] rw_k1;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_2d(Tg x1_t, Tg x2_t,
                       Tg * x1_s, Tg * x2_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2,
                       int d, double gam, int W,
                       int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;
  size_t ix_jn, ix_jl, ix_ju;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns_x1,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_jn,
    &ix_jl,
    &ix_ju,
    x2_t,
    x2_s,
    Ns_x2,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_t, &x1_s[ix_il], W-1, k1, d, gam);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }

  Tg * rw_k2 = new Tg[W];
  for(int k2=0; k2<=W-1; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_t, &x2_s[ix_jl], W-1, k2, d, gam);
    const Tg rdx2 = 1. / (x2_t - x2_s[ix_jl+k2]);
    rw_k2[k2] = w_k2 * std::pow(rdx2, gam);
  }
  // --------------------------------------------------------------------------

  for(int k2=0; k2<=W-1; ++k2)
  for(int k1=0; k1<=W-1; ++k1)
  {
    const int ix = (k1+ng+ix_il) + (Ns_x1+1+2*ng) * (k2+ng+ix_jl);

    num += rw_k1[k1] * rw_k2[k2] * fcn_s[ix];
    den += rw_k1[k1] * rw_k2[k2];
  }

  delete[] rw_k1;
  delete[] rw_k2;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_nn_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                       Tg * x1_s, Tg * x2_s, Tg * x3_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2, int Ns_x3,
                       int d, double gam, int W,
                       int ng, int ne)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  size_t ix_in, ix_il, ix_iu;
  size_t ix_jn, ix_jl, ix_ju;
  size_t ix_kn, ix_kl, ix_ku;

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_in,
    &ix_il,
    &ix_iu,
    x1_t,
    x1_s,
    Ns_x1,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_jn,
    &ix_jl,
    &ix_ju,
    x2_t,
    x2_s,
    Ns_x2,
    W,
    ne
  );

  numprox::interpolation::shared::idx_range_nearest_point(
    &ix_kn,
    &ix_kl,
    &ix_ku,
    x3_t,
    x3_s,
    Ns_x3,
    W,
    ne
  );

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc --------------------------------
  Tg * rw_k1 = new Tg[W];
  for(int k1=0; k1<=W-1; ++k1)
  {
    const Tg w_k1 = weight<Tg>(x1_t, &x1_s[ix_il], W-1, k1, d, gam);
    const Tg rdx1 = 1. / (x1_t - x1_s[ix_il+k1]);
    rw_k1[k1] = w_k1 * std::pow(rdx1, gam);
  }

  Tg * rw_k2 = new Tg[W];
  for(int k2=0; k2<=W-1; ++k2)
  {
    const Tg w_k2 = weight<Tg>(x2_t, &x2_s[ix_jl], W-1, k2, d, gam);
    const Tg rdx2 = 1. / (x2_t - x2_s[ix_jl+k2]);
    rw_k2[k2] = w_k2 * std::pow(rdx2, gam);
  }

  Tg * rw_k3 = new Tg[W];
  for(int k3=0; k3<=W-1; ++k3)
  {
    const Tg w_k3 = weight<Tg>(x3_t, &x3_s[ix_kl], W-1, k3, d, gam);
    const Tg rdx3 = 1. / (x3_t - x3_s[ix_kl+k3]);
    rw_k3[k3] = w_k3 * std::pow(rdx3, gam);
  }
  // --------------------------------------------------------------------------

  for(int k3=0; k3<=W-1; ++k3)
  for(int k2=0; k2<=W-1; ++k2)
  for(int k1=0; k1<=W-1; ++k1)
  {
    const int ix = (
      (k1+ng+ix_il) + (Ns_x1+1+2*ng) *
                      ((k2+ng+ix_jl) +
                       (Ns_x2+1+2*ng) * (k3+ng+ix_kl))
    );

    num += rw_k1[k1]*rw_k2[k2]*rw_k3[k3]*fcn_s[ix];
    den += rw_k1[k1]*rw_k2[k2]*rw_k3[k3];
  }

  delete[] rw_k1;
  delete[] rw_k2;
  delete[] rw_k3;

  return static_cast<Tf>(num / den);
}

// ============================================================================
}
// ============================================================================

// ============================================================================
namespace shared {
// ============================================================================

template<typename Tg>
inline void idx_range_nearest_point(
  size_t* ix_np,
  size_t* ix_il,
  size_t* ix_iu,
  const Tg x_t,
  const Tg* x_s,
  const int Ns,
  const int width,
  const int ne_)
{
  const size_t ne = static_cast<size_t>(ne_);
  size_t N = Ns+1;

  // find idx of nearest x_s to x_t
  size_t i_ub = std::upper_bound(x_s, x_s + N, x_t) - x_s;

  i_ub = (i_ub < ne) ? ne : i_ub;
  i_ub = (i_ub > N-ne-1) ? N-ne-1 : i_ub;

  i_ub = (i_ub > ne) ? (
    (std::abs(x_s[i_ub  ] - x_t) >
     std::abs(x_s[i_ub-1] - x_t)) ? i_ub-1 : i_ub
  ) : i_ub;

  size_t il = i_ub, iu = i_ub;

  size_t num_W = width;

  while (num_W > 1)
  {
    if (il > static_cast<size_t>(ne))
    {
      if (iu < static_cast<size_t>(N-1-ne))
      {
        if (std::abs(x_s[il-1] - x_s[i_ub]) <=
            std::abs(x_s[iu+1] - x_s[i_ub]))
        {
          il--; num_W--;
          continue;
        }
      }
    }
    else
    {
      iu++; num_W--;
      continue;
    }

    if (iu < static_cast<size_t>(N-1-ne))
    {
      if (il > static_cast<size_t>(ne))
      {
        if (std::abs(x_s[iu+1] - x_s[i_ub]) <=
            std::abs(x_s[il-1] - x_s[i_ub]))
        {
          iu++; num_W--;
          continue;
        }
      }
    }
    else
    {
      il--; num_W--;
      continue;
    }


    num_W--;
  }

  // populate
  *ix_np = i_ub;
  *ix_il = il;
  *ix_iu = iu;
}

// ============================================================================
}
// ============================================================================

// ============================================================================
namespace impl {
// ============================================================================

// Used with D1
template<typename Tg, typename Tf>
static inline void _D1(
  const Tf * const fcn_s,
  const Tg * const r1d_x1_wei,
  Tf * d1_fcn_t,
  const int Ns,
  const int ng)
{

  // compute derivative
  for(int i=0; i<=Ns; ++i)
  {
    d1_fcn_t[i+ng] = 0;
    for(int k=0; k<=Ns; ++k)
    {
      const int ix = k + (Ns+1)*i;
      d1_fcn_t[i+ng] += r1d_x1_wei[ix] * (fcn_s[k+ng] - fcn_s[i+ng]);
    }
  }
}

template<typename Tg, typename Tf>
static inline void _D1(
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  const Tg * const r2d_x1_wei,
  Tf * d1_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int ng_t,
  const int ng_s)
{
  // compute derivative
  for(int i=0; i<=Nt_x1; ++i)
  {
    d1_fcn_t[i+ng_t] = 0;
    for(int k=0; k<=Ns_x1; ++k)
    {
      const int ix = k + (Ns_x1+1)*i;
      d1_fcn_t[i+ng_t] += r2d_x1_wei[ix] * (fcn_s[k+ng_s] - fcn_t[i+ng_t]);
    }
  }
}

// Used with D2
template<typename Tg, typename Tf>
static inline void _D2(
  const Tf * const fcn_s,
  const Tf * const d1_fcn_s,
  const Tg * const r1d_x1,
  const Tg * const r2d_x1_wei,
  Tf * d2_fcn_t,
  const int Ns,
  const int ng)
{
  // compute derivative
  for(int i=0; i<=Ns; ++i)
  {
    d2_fcn_t[i+ng] = 0;
    for(int k=0; k<=Ns; ++k)
    {
      const int ix = k + (Ns+1)*i;
      d2_fcn_t[i+ng] += r2d_x1_wei[ix] * (
        (fcn_s[k+ng] - fcn_s[i+ng]) * r1d_x1[ix] -
        d1_fcn_s[i+ng]
      );
    }
  }
}

template<typename Tg, typename Tf>
static inline void _D2(
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  const Tf * const d1_fcn_t,
  const Tg * const r1d_x1,
  const Tg * const r2d_x1_wei,
  Tf * d2_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int ng_t,
  const int ng_s)
{
  // compute derivative
  for(int i=0; i<=Nt_x1; ++i)
  {
    d2_fcn_t[i+ng_t] = 0;
    for(int k=0; k<=Ns_x1; ++k)
    {
      const int ix = k + (Ns_x1+1)*i;
      d2_fcn_t[i+ng_t] += r2d_x1_wei[ix] * (
        (fcn_s[k+ng_s] - fcn_t[i+ng_t]) * r1d_x1[ix] -
        d1_fcn_t[i+ng_t]
      );
    }
  }
}

// ============================================================================
}
// ============================================================================

// ============================================================================
}}
// ============================================================================
// :D