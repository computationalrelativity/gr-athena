
// headers ====================================================================
#include "interp_barycentric.hpp"
// ============================================================================

// BD: is compensated summation here worth it?

namespace numprox { namespace interpolation {

// ============================================================================
namespace Floater_Hormann {
// ============================================================================

template <typename T>
static inline T weight(T * gr_s,
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

template<typename Tg, typename Tf>
static Tf interp_1d(Tg x1_t, Tg * gr_s,
                    Tf * fcn_s,
                    int N, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  for(int k=0; k<=N; ++k)
  {
    const Tg w_k  = weight<Tg>(gr_s, N, k, d);
    const Tg rdx  = 1. / (x1_t - gr_s[k]);
    const Tg rw_k = w_k * rdx;

    num += rw_k * fcn_s[k+ng];
    den += rw_k;
  }

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight<Tg>(x1_s, N_x1, k1, d);

  for(int k2=0; k2<=N_x2; ++k2)
  {
    const Tg w_k2  = weight<Tg>(x2_s, N_x2, k2, d);
    const Tg rdx2  = 1. / (x2_t - x2_s[k2]);
    const Tg rw_k2 = w_k2 * rdx2;

    for(int k1=0; k1<=N_x1; ++k1)
    {
      const Tg w_k1  = arr_w_k1[k1];  // lifted
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

template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int N_x3, int d, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight<Tg>(x1_s, N_x1, k1, d);

  Tg * arr_w_k2 = new Tg[N_x2 + 1];
  for(int k2=0; k2<=N_x2; ++k2)
    arr_w_k2[k2] = weight<Tg>(x2_s, N_x2, k2, d);

  for(int k3=0; k3<=N_x3; ++k3)
  {
    const Tg w_k3  = weight<Tg>(x3_s, N_x3, k3, d);
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

// ============================================================================
}
// ============================================================================

// ============================================================================
namespace Floater_Hormann_generalized {
// ============================================================================

template <typename T>
static inline T weight(T gr_t, T * gr_s, int N,
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

template<typename Tg, typename Tf>
static Tf interp_1d(Tg gr_t, Tg * gr_s, Tf * fcn_s,
                    int N, int d, double gam, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  for(int k=0; k<=N; ++k)
  {
    const Tg w_k = weight<Tg>(gr_t, gr_s, N, k, d, gam);
    const Tg rdx  = 1. / (gr_t - gr_s[k]);
    const Tg rw_k = w_k * std::pow(rdx, gam);

    num += rw_k * fcn_s[k+ng];
    den += rw_k;
  }

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int d, double gam, int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight<Tg>(x1_t, x1_s, N_x1, k1, d, gam);

  for(int k2=0; k2<=N_x2; ++k2)
  {
    const Tg w_k2  = weight<Tg>(x2_t, x2_s, N_x2, k2, d, gam);
    const Tg rdx2  = 1. / (x2_t - x2_s[k2]);
    const Tg rw_k2 = w_k2 * std::pow(rdx2, gam);

    for(int k1=0; k1<=N_x1; ++k1)
    {
      const Tg w_k1  = arr_w_k1[k1];
      const Tg rdx1  = 1. / (x1_t - x1_s[k1]);
      const Tg rw_k1 = w_k1 * std::pow(rdx1, gam);

      const int ix = (k1+ng) + (N_x1+1+2*ng) * (k2+ng);

      num += rw_k1 * rw_k2 * fcn_s[ix];
      den += rw_k1 * rw_k2;
    }
  }

  delete[] arr_w_k1;

  return static_cast<Tf>(num / den);
}

template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int N_x3, int d, double gam,
                    int ng)
{
  // weight & function return common value
  typedef typename std::common_type<Tg, Tf>::type Tr;

  Tr num = 0, den = 0;

  // loop-lift for speed at the cost of malloc
  Tg * arr_w_k1 = new Tg[N_x1 + 1];
  for(int k1=0; k1<=N_x1; ++k1)
    arr_w_k1[k1] = weight<Tg>(x1_t, x1_s, N_x1, k1, d, gam);

  Tg * arr_w_k2 = new Tg[N_x2 + 1];
  for(int k2=0; k2<=N_x2; ++k2)
    arr_w_k2[k2] = weight<Tg>(x2_t, x2_s, N_x2, k2, d, gam);

  for(int k3=0; k3<=N_x3; ++k3)
  {
    const Tg w_k3  = weight<Tg>(x3_t, x3_s, N_x3, k3, d, gam);
    const Tg rdx3  = 1. / (x3_t - x3_s[k3]);
    const Tg rw_k3 = w_k3 * std::pow(rdx3, gam);

    for(int k2=0; k2<=N_x2; ++k2)
    {
      const Tg w_k2  = arr_w_k2[k2];
      const Tg rdx2  = 1. / (x2_t - x2_s[k2]);
      const Tg rw_k2 = w_k2 * std::pow(rdx2, gam);

      for(int k1=0; k1<=N_x1; ++k1)
      {
        const Tg w_k1  = arr_w_k1[k1];
        const Tg rdx1  = 1. / (x1_t - x1_s[k1]);
        const Tg rw_k1 = w_k1 * std::pow(rdx1, gam);

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
// ============================================================================
}
// ============================================================================

// ============================================================================
}}
// ============================================================================
// :D