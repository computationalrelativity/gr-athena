#ifndef NUMPROX_INTERP_BARYCENTRIC_HPP
#define NUMPROX_INTERP_BARYCENTRIC_HPP

// headers ====================================================================
// c / c++
#include <cmath>
#include <type_traits>

// numprox
// #include "../constants.hpp"
// ============================================================================

// ============================================================================
namespace numprox { namespace interpolation {
// ============================================================================

// ============================================================================
namespace Floater_Hormann {
// ============================================================================

// Construct Floater-Hormann weights on an input grid
//
// Parameters:
// -----------
// gr_s: Source grid with N + 1 entries
// Ns:    Source grid size parameter
// k:    Weight index values (0,... , N)
// d:    FH blending parameter
template <typename T>
static inline T weight(T * gr_s,
                       int Ns,
                       int k, int d);

// Direct interpolation based on one-dimensional Floater-Hormann barycentric
// form.
//
// Parameters:
// -----------
// x1_t:  Target base-point to interpolate to
// x1_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// Ns:    Source grid size parameter
// d:     FH blending parameter
// ng=0:  Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that x1_s is passed with first entry at x1_s[ng]
template<typename Tg, typename Tf>
static Tf interp_1d(Tg x1_t, Tg * x1_s,
                    Tf * fcn_s,
                    int Ns, int d, int ng=0);

// As above, targets 2d
template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int d, int ng=0);

// As above, targets 3d
template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int Ns_x3, int d, int ng=0);

// Utilize precomputed weights
template <typename Tg, typename Tf>
class interp_1d_cached
{
  public:
    // evaluate interpolant
    inline void eval(Tf* fcn_t, const Tf* const fcn_s)
    {
      for(int i1=0; i1<=Nt_x1; ++i1)
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

  private:
    // compute and store weights together with reciprocal factors
    inline void precompute_weights(
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

  private:
    const int Nt_x1;
    const int Ns_x1;
    const int d;
    const int ng_t;
    const int ng_s;

    Tg* rwei_x1;

    // weight & function of common value
    typedef typename std::common_type<Tg, Tf>::type Tr;

  public:
    // ctor -------------------------------------------------------------------
    interp_1d_cached(Tg* x1_t, Tg* x1_s,
                     const int Nt_x1, const int Ns_x1,
                     const int d,
                     const int ng_t,
                     const int ng_s)
                     : Nt_x1 {Nt_x1}
                     , Ns_x1 {Ns_x1}
                     , d {d}
                     , ng_t {ng_t}
                     , ng_s {ng_s}
                     , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
    {
      precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
    }
    interp_1d_cached(Tg* x1_t, Tg* x1_s,
                     const int Nt_x1, const int Ns_x1,
                     const int d,
                     const int ng)
                     : interp_1d_cached(x1_t, x1_s,
                                        Nt_x1, Ns_x1, d, ng, ng)
    { }
    interp_1d_cached(Tg* x1_t, Tg* x1_s,
                     const int Nt_x1, const int Ns_x1,
                     const int d)
                     : interp_1d_cached(x1_t, x1_s,
                                        Nt_x1, Ns_x1, d, 0, 0)
    { }

    // dtor -------------------------------------------------------------------
    ~interp_1d_cached()
    {
      delete[] rwei_x1;
    }

};

// Support up to 3d interpolation with weight precomputation
template <typename Tg, typename Tf>
class interp_nd_weights_precomputed
{
  public:
    // evaluate interpolant
    inline void eval(Tf* fcn_t, const Tf* const fcn_s)
    {
      if (ndim == 3)
      {
        for(int i3=0; i3<=Nt_x3; ++i3)
        for(int i2=0; i2<=Nt_x2; ++i2)
        for(int i1=0; i1<=Nt_x1; ++i1)
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
      else if (ndim == 2)
      {
        for(int i2=0; i2<=Nt_x2; ++i2)
        for(int i1=0; i1<=Nt_x1; ++i1)
        {
          // if((NGHOST <= i1) && (i1 <= Nt_x1 - NGHOST + 1) &&
          //    (NGHOST <= i2) && (i2 <= Nt_x2 - NGHOST + 1))
          //   continue;

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

        /*
        // single-dir test ----------------------------------------------------
        // fbuf_2_x1x2
        // fbuf_2_x2x1
        for(int i2=0; i2<=Ns_x2; ++i2)
        for(int i1=0; i1<=Nt_x1; ++i1)
        {
          Tr num = 0, den = 0;

          for(int k1=0; k1<=Ns_x1; ++k1)
          {
            // k1 is fastest running index
            const int ix_k1 = k1 + (Ns_x1+1)*i1;
            const int fs_ix = (k1+ng_s) + (Ns_x1+1+2*ng_s) * (i2+ng_s);

            num += rwei_x1[ix_k1] * fcn_s[fs_ix];
            den += rwei_x1[ix_k1];
          }

          const int ft_ix = i1 + (Nt_x1+1) * i2;
          fbuf_2_x1x2[ft_ix] = num / den;
        }

        for(int i2=0; i2<=Nt_x2; ++i2)
        for(int i1=0; i1<=Nt_x1; ++i1)
        {
          Tr num = 0, den = 0;

          for(int k2=0; k2<=Ns_x2; ++k2)
          {
            const int ix_k2 = k2 + (Ns_x2+1)*i2;
            const int fs_ix = i1 + (Nt_x1+1)*k2;

            num += rwei_x2[ix_k2] * fbuf_2_x1x2[fs_ix];
            den += rwei_x2[ix_k2];
          }

          // const int ft_ix = i1 + (Nt_x1+1) * i2;
          const int ft_ix = (i1+ng_t) + (Nt_x1+1+2*ng_t) * (i2+ng_t);
          fcn_t[ft_ix] = num / den;
        }
        // --------------------------------------------------------------------
        */
      }
      else
      {
        for(int i1=0; i1<=Nt_x1; ++i1)
        {
          // if((NGHOST <= i1) && (i1 <= Nt_x1 - NGHOST + 1))
          //   continue;

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

    }

  private:
    // compute and store weights together with reciprocal factors
    inline void precompute_weights(
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

  private:
    const int ndim;

    const int Nt_x1;
    const int Nt_x2;
    const int Nt_x3;

    const int Ns_x1;
    const int Ns_x2;
    const int Ns_x3;

    const int d;
    const int ng_t;
    const int ng_s;

    Tg* rwei_x1;
    Tg* rwei_x2;
    Tg* rwei_x3;

    // buffers for uni-dir iter test ----------------------
    // Tf* fbuf_2_x1x2;
    // Tf* fbuf_2_x2x1;
    // ----------------------------------------------------

    // weight & function of common value
    typedef typename std::common_type<Tg, Tf>::type Tr;

  public:
    // ctor (1d) --------------------------------------------------------------
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x1_s,
                                  const int Nt_x1, const int Ns_x1,
                                  const int d,
                                  const int ng_t,
                                  const int ng_s)
                                  : ndim {1}
                                  , Nt_x1 {Nt_x1}
                                  , Nt_x2 {1}
                                  , Nt_x3 {1}
                                  , Ns_x1 {Ns_x1}
                                  , Ns_x2 {1}
                                  , Ns_x3 {1}
                                  , d {d}
                                  , ng_t {ng_t}
                                  , ng_s {ng_s}
                                  , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
    {
      precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
    }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x1_s,
                                  const int Nt_x1, const int Ns_x1,
                                  const int d,
                                  const int ng)
      : interp_nd_weights_precomputed(x1_t, x1_s,
                                      Nt_x1, Ns_x1, d, ng, ng)
    { }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x1_s,
                                  const int Nt_x1, const int Ns_x1,
                                  const int d)
      : interp_nd_weights_precomputed(x1_t, x1_s,
                                      Nt_x1, Ns_x1, d, 0, 0)
    { }

    // ctor (2d) --------------------------------------------------------------
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t,
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
                                  , ng_t {ng_t}
                                  , ng_s {ng_s}
                                  , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
                                  , rwei_x2 {new Tg[(Ns_x2+1)*(Nt_x2+1)]}
                                  // test single dir iter ---------------------
                                  // , fbuf_2_x1x2 {new Tg[(Nt_x1+1)*(Ns_x2+1)]}
                                  // , fbuf_2_x2x1 {new Tg[(Ns_x1+1)*(Nt_x2+1)]}
                                  // ------------------------------------------
    {
      precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
      precompute_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_x2);
    }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t,
                                  Tg* x1_s, Tg* x2_s,
                                  const int Nt_x1, const int Nt_x2,
                                  const int Ns_x1, const int Ns_x2,
                                  const int d,
                                  const int ng)
      : interp_nd_weights_precomputed(x1_t, x2_t,
                                      x1_s, x2_s,
                                      Nt_x1, Nt_x2,
                                      Ns_x1, Ns_x2,
                                      d, ng, ng)
    { }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t,
                                  Tg* x1_s, Tg* x2_s,
                                  const int Nt_x1, const int Nt_x2,
                                  const int Ns_x1, const int Ns_x2,
                                  const int d)
      : interp_nd_weights_precomputed(x1_t, x2_t,
                                      x1_s, x2_s,
                                      Nt_x1, Nt_x2,
                                      Ns_x1, Ns_x2,
                                      d, 0, 0)
    { }

    // ctor (3d) --------------------------------------------------------------
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t, Tg* x3_t,
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
                                  , ng_t {ng_t}
                                  , ng_s {ng_s}
                                  , rwei_x1 {new Tg[(Ns_x1+1)*(Nt_x1+1)]}
                                  , rwei_x2 {new Tg[(Ns_x2+1)*(Nt_x2+1)]}
                                  , rwei_x3 {new Tg[(Ns_x3+1)*(Nt_x3+1)]}
    {
      precompute_weights(x1_t, x1_s, Nt_x1, Ns_x1, rwei_x1);
      precompute_weights(x2_t, x2_s, Nt_x2, Ns_x2, rwei_x2);
      precompute_weights(x3_t, x3_s, Nt_x3, Ns_x3, rwei_x3);
    }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t, Tg* x3_t,
                                  Tg* x1_s, Tg* x2_s, Tg* x3_s,
                                  const int Nt_x1,
                                  const int Nt_x2,
                                  const int Nt_x3,
                                  const int Ns_x1,
                                  const int Ns_x2,
                                  const int Ns_x3,
                                  const int d,
                                  const int ng)
      : interp_nd_weights_precomputed(x1_t, x2_t, x3_t,
                                      x1_s, x2_s, x3_s,
                                      Nt_x1, Nt_x2, Nt_x3,
                                      Ns_x1, Ns_x2, Ns_x3,
                                      d, ng, ng)
    { }
    interp_nd_weights_precomputed(Tg* x1_t, Tg* x2_t, Tg* x3_t,
                                  Tg* x1_s, Tg* x2_s, Tg* x3_s,
                                  const int Nt_x1,
                                  const int Nt_x2,
                                  const int Nt_x3,
                                  const int Ns_x1,
                                  const int Ns_x2,
                                  const int Ns_x3,
                                  const int d)
      : interp_nd_weights_precomputed(x1_t, x2_t, x3_t,
                                      x1_s, x2_s, x3_s,
                                      Nt_x1, Nt_x2, Nt_x3,
                                      Ns_x1, Ns_x2, Ns_x3,
                                      d, 0, 0)
    { }


    // dtor -------------------------------------------------------------------
    ~interp_nd_weights_precomputed()
    {
      switch (ndim)
      {
        case 3:
          delete[] rwei_x3;
          delete[] rwei_x2;
          delete[] rwei_x1;
          break;
        case 2:
          delete[] rwei_x2;
          delete[] rwei_x1;
          // test single dir iter -------------------------
          // delete[] fbuf_2_x1x2;
          // delete[] fbuf_2_x2x1;
          // ----------------------------------------------

          break;
        default:
          delete[] rwei_x1;
      }

    }

};


// ============================================================================
}
// ============================================================================

// ============================================================================
namespace Floater_Hormann_generalized {
// ============================================================================

// Construct generalized Floater-Hormann weights on an input grid
//
// Parameters:
// -----------
// gr_t:  Target base-point to interpolate to
// gr_s:  Source grid with Ns + 1 entries
// Ns:    Source grid size parameter
// d:     FH blending parameter
// gam:   Rational power for barycentric form (reduced to FH for gam=1)
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that gr_s is passed with first entry at gr_s[ng]
template <typename T>
static inline T weight(T gr_t, T * gr_s, int Ns,
                       int k, int d, double gam);

// Direct interpolation based on one-dimensional generalized Floater-Hormann
// barycentric form.
//
// Parameters:
// -----------
// gr_t:  Target base-point to interpolate to
// gr_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// N:     Grid size parameter
// d:     FH blending parameter
// gam:   Rational power for barycentric form (reduced to FH for gam=1)
// ng=0:  Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that gr_s is passed with first entry at gr_s[ng]
template<typename Tg, typename Tf>
static Tf interp_1d(Tg gr_t, Tg * gr_s, Tf * fcn_s,
                    int N, int d, double gam, int ng=0);
// As above, targets 2d
template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int d, double gam, int ng=0);
// As above, targets 3d
template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int N_x3, int d, double gam,
                    int ng=0);

// ============================================================================
}
// ============================================================================

}}


// ============================================================================

// implementation details (for templates) =====================================
#include "interp_barycentric.tpp"
// ============================================================================

#endif // NUMPROX_INTERP_BARYCENTRIC_HPP
// :D