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
// k:    Weight index values (0,... , N)
// d:    FH blending parameter
template <typename T>
static inline T weight(T * gr_s,
                       int N,
                       int k, int d);

// Direct interpolation based on one-dimensional Floater-Hormann barycentric
// form.
//
// Parameters:
// -----------
// x1_t:  Target base-point to interpolate to
// x1_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// N:     Grid size parameter
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
                    int N, int d, int ng=0);

// As above, targets 2d
template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int d, int ng=0);

// As above, targets 3d
template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int N_x3, int d, int ng=0);

// Utilize precomputed weights
template <typename Tg, typename Tf>
class interp_1d_cached
{
  public:
    inline void eval(Tf* fcn_t, const Tf* const fcn_s)
    {
      for(int i1=0; i1<=Nt_x1; ++i1)
      {
        Tr num = 0, den = 0;

        for(int k1=0; k1<=Ns_x1; ++k1)
        {
          // k1 is fastest running index
          const int ix = k1 + (Ns_x1+1)*i1;
          num += rwei_x1[ix] * fcn_s[k1];
          den += rwei_x1[ix];
        }

        fcn_t[i1+ng] = num / den;
      }

    }

  private:
    Tg* rwei_x1;
    const int Nt_x1;
    const int Ns_x1;
    const int d;
    const int ng;
    // weight & function of common value
    typedef typename std::common_type<Tg, Tf>::type Tr;

  public:
    // ctor -------------------------------------------------------------------
    interp_1d_cached(Tg* x1_t, Tg* x1_s,
                     const int Nt_x1, const int Ns_x1,
                     const int d,
                     const int ng)
                     : Nt_x1{Nt_x1}
                     , Ns_x1{Ns_x1}
                     , d {d}
                     , ng {ng}
    {
      // compute and store weights together with reciprocal factors
      rwei_x1 = new Tg[(Ns_x1+1)*(Nt_x1+1)];

      for(int k1=0; k1<=Ns_x1; ++k1)
      {
        const Tg w_k1 = weight(x1_s, Ns_x1, k1, d);

        for(int i1=0; i1<=Nt_x1; ++i1)
        {
          const Tg rdx1 = 1. / (x1_t[i1] - x1_s[k1]);
          const int ix = k1 + (Ns_x1+1)*i1;
          rwei_x1[ix] = w_k1 * rdx1;
        }
      }
    }
    interp_1d_cached(Tg* x1_t, Tg* x1_s,
                     const int Nt_x1, const int Ns_x1,
                     const int d)
                     : interp_1d_cached(x1_t, x1_s,
                                        Nt_x1, Ns_x1, d, 0)
    { }

    // dtor -------------------------------------------------------------------
    ~interp_1d_cached()
    {
      delete[] rwei_x1;
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
// gr_s:  Source grid with N + 1 entries
// N:     Grid size parameter
// d:     FH blending parameter
// gam:   Rational power for barycentric form (reduced to FH for gam=1)
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that gr_s is passed with first entry at gr_s[ng]
template <typename T>
static inline T weight(T gr_t, T * gr_s, int N,
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