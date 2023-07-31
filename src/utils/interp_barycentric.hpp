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

// References:
// -----------
// Floater, Michael S., and Kai Hormann. 2007.
//
// ‘Barycentric Rational Interpolation with No Poles and High Rates of
// Approximation’. Numerische Mathematik 107 (2): 315–31.
// https://doi.org/10.1007/s00211-007-0093-y.
//
//
// Berrut, Jean-Paul, Richard Baltensperger, and Hans D. Mittelmann. 2005.
//
// ‘Recent Developments in Barycentric Rational Interpolation’.
// In Trends and Applications in Constructive Approximation, edited by
// Detlef H. Mache, József Szabados, and Marcel G. de Bruin, 27–51.
// ISNM International Series of Numerical Mathematics. Basel: Birkhäuser.
// https://doi.org/10.1007/3-7643-7356-3_3.
//
// Themistoclakis, Woula, and Marc Van Barel. 2023.
//
// ‘A Generalization of Floater--Hormann Interpolants’. arXiv.
// https://doi.org/10.48550/arXiv.2307.05345.

// ============================================================================
namespace Floater_Hormann {
// ============================================================================

// Construct Floater-Hormann weights on an input grid
//
// Parameters:
// -----------
// gr_s: Source grid with N + 1 entries
// Ns:   Source grid size parameter
// k:    Weight index values (0,... , N)
// d:    FH blending parameter
template <typename T>
static inline T weight(const T * const gr_s,
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

// Targets 2d see interp_1d
template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int d, int ng=0);

// Targets 3d see interp_1d
template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int Ns_x1, int Ns_x2, int Ns_x3, int d, int ng=0);

// Direct interpolation based on one-dimensional Floater-Hormann barycentric
// form. Control on total number of nearest-neighbour points involved.
//
// Parameters:
// -----------
// x1_t:  Target base-point to interpolate to
// x1_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// Ns:    Source grid size parameter
// d:     FH blending parameter
// W:     Number of points involved in stencil; nearest points to x_t selected
// ng=0:  Optionally control skipping of ghost-zone (i.e. control strides)
// ne=0:  Optionally control skipping of an interior edge zone
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that x1_s is passed with first entry at x1_s[ng]
template<typename Tg, typename Tf>
static Tf interp_nn_1d(Tg x1_t, Tg * x1_s,
                       Tf * fcn_s,
                       int Ns,
                       int d, int W, int ng=0, int ne=0);

// Targets 2d see interp_nn_1d
template<typename Tg, typename Tf>
static Tf interp_nn_2d(Tg x1_t, Tg x2_t,
                       Tg * x1_s, Tg * x2_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2,
                       int d, int W, int ng=0, int ne=0);

// Targets 3d see interp_nn_1d
template<typename Tg, typename Tf>
static Tf interp_nn_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                       Tg * x1_s, Tg * x2_s, Tg * x3_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2, int Ns_x3,
                       int d, int W, int ng=0, int ne=0);

// Direct differentiation (degree 1) based on one-dimensional Floater-Hormann
// barycentric form. Derivatives computed on source grid nodes.
//
// Parameters:
// -----------
// x1_s:     Source grid with N + 1 entries
// fcn_s:    Source function samples with N + 1 entries
// d1_fcn_t: Output with N + 1 entries
// Ns:       Source grid size parameter
// d:        FH blending parameter
// ng=0:     Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Weights are computed internally
//   - For ng!=0 it is assumed that x1_s is passed with first entry at x1_s[ng]
template<typename Tg, typename Tf>
static void D1(
  const Tg * const x1_s,
  const Tf * const fcn_s,
  Tf * d1_fcn_t,
  const int Ns,
  const int d,
  const int ng=0);

// Direct differentiation (degree 1) based on one-dimensional Floater-Hormann
// barycentric form. Derivatives computed on target grid nodes that must be
// distinct from source nodes.
//
// Parameters:
// -----------
// x1_t:     Target base-points with Nt + 1 entries
// x1_s:     Source grid with Ns + 1 entries
// fcn_t:    Source function samples with Nt + 1 entries
// fcn_s:    Source function samples with Ns + 1 entries
// d1_fcn_t: Output with Nt + 1 entries
// Nt:       Target grid size parameter
// Ns:       Source grid size parameter
// d:        FH blending parameter
// ng_t=0:   Optionally control skipping of ghost-zone (i.e. control strides)
// ng_s=0:   Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Weights are computed internally
//   - For ng_s!=0 it is assumed that x1_s is passed with first entry at
//     x1_s[ng_s] and similarly for ng_t.
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
  const int ng_t=0,
  const int ng_s=0);

// Direct differentiation (degree 2) based on one-dimensional Floater-Hormann
// barycentric form. Derivatives computed on source grid nodes.
//
// Parameters:
// -----------
// x1_s:     Source grid with N + 1 entries
// fcn_s:    Source function samples with N + 1 entries
// d1_fcn_s: Source function derivative samples with N + 1 entries
// d2_fcn_t: Output with N + 1 entries
// Ns:       Source grid size parameter
// d:        FH blending parameter
// ng=0:     Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Weights are computed internally
//   - For ng!=0 it is assumed that x1_s is passed with first entry at x1_s[ng]
template<typename Tg, typename Tf>
static void D2(
  const Tg * const x1_s,
  const Tf * const fcn_s,
  const Tf * const d1_fcn_s,
  Tf * d2_fcn_t,
  const int Ns,
  const int d,
  const int ng=0);

// Direct differentiation (degree 2) based on one-dimensional Floater-Hormann
// barycentric form. Derivatives computed on target grid nodes that must be
// distinct from source nodes.
//
// Parameters:
// -----------
// x1_t:     Target base-points with Nt + 1 entries
// x1_s:     Source grid with Ns + 1 entries
// fcn_t:    Source function samples with Nt + 1 entries
// fcn_s:    Source function samples with Ns + 1 entries
// d1_fcn_t: Source function derivative samples with Nt + 1 entries
// d1_fcn_s: Source function derivative samples with Ns + 1 entries
// d2_fcn_t: Output with Nt + 1 entries
// Nt:       Target grid size parameter
// Ns:       Source grid size parameter
// d:        FH blending parameter
// ng_t=0:   Optionally control skipping of ghost-zone (i.e. control strides)
// ng_s=0:   Optionally control skipping of ghost-zone (i.e. control strides)
//
// Note(s):
// --------
//   - Weights are computed internally
//   - For ng_s!=0 it is assumed that x1_s is passed with first entry at
//     x1_s[ng_s] and similarly for ng_t.
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
  const int ng_t=0,
  const int ng_s=0);

// classes --------------------------------------------------------------------

// Support up to 3d interpolation with weight precomputation
template <typename Tg, typename Tf>
class interp_nd
{
  public:
    // ctor (1d) --------------------------------------------------------------
    interp_nd(Tg* x1_t, Tg* x1_s,
              const int Nt_x1, const int Ns_x1,
              const int d, const int ng_t, const int ng_s);
    interp_nd(Tg* x1_t, Tg* x1_s,
              const int Nt_x1, const int Ns_x1,
              const int d, const int ng);
    interp_nd(Tg* x1_t, Tg* x1_s,
              const int Nt_x1, const int Ns_x1,
              const int d);

    // ctor (2d) --------------------------------------------------------------
    interp_nd(Tg* x1_t, Tg* x2_t,
              Tg* x1_s, Tg* x2_s,
              const int Nt_x1, const int Nt_x2,
              const int Ns_x1, const int Ns_x2,
              const int d, const int ng_t, const int ng_s);
    interp_nd(Tg* x1_t, Tg* x2_t,
              Tg* x1_s, Tg* x2_s,
              const int Nt_x1, const int Nt_x2,
              const int Ns_x1, const int Ns_x2,
              const int d, const int ng);
    interp_nd(Tg* x1_t, Tg* x2_t,
              Tg* x1_s, Tg* x2_s,
              const int Nt_x1, const int Nt_x2,
              const int Ns_x1, const int Ns_x2,
              const int d);

    // ctor (3d) --------------------------------------------------------------
    interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
              Tg* x1_s, Tg* x2_s, Tg* x3_s,
              const int Nt_x1, const int Nt_x2, const int Nt_x3,
              const int Ns_x1, const int Ns_x2, const int Ns_x3,
              const int d, const int ng_t, const int ng_s);
    interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
              Tg* x1_s, Tg* x2_s, Tg* x3_s,
              const int Nt_x1, const int Nt_x2, const int Nt_x3,
              const int Ns_x1, const int Ns_x2, const int Ns_x3,
              const int d, const int ng);
    interp_nd(Tg* x1_t, Tg* x2_t, Tg* x3_t,
              Tg* x1_s, Tg* x2_s, Tg* x3_s,
              const int Nt_x1, const int Nt_x2, const int Nt_x3,
              const int Ns_x1, const int Ns_x2, const int Ns_x3,
              const int d);

    // dtor -------------------------------------------------------------------
    ~interp_nd();

  public:
    // Evaluate interpolant on full (target) grid
    inline void eval(Tf* fcn_t, const Tf* const fcn_s);

    // Evaluate interpolant over (full target grid) indicial range
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    // {jl_t, ju_t} are elements of (0, Nt_x2)
    // {kl_t, ku_t} are elements of (0, Nt_x3)
    inline void eval(Tf* fcn_t, const Tf* const fcn_s,
                     const int il_t, const int iu_t,
                     const int jl_t, const int ju_t,
                     const int kl_t, const int ku_t);

    // Evaluate interpolant over (full target grid) indicial range
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    // {jl_t, ju_t} are elements of (0, Nt_x2)
    inline void eval(Tf* fcn_t, const Tf* const fcn_s,
                     const int il_t, const int iu_t,
                     const int jl_t, const int ju_t);

    // Evaluate interpolant over (full target grid) indicial range.
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    inline void eval(Tf* fcn_t, const Tf* const fcn_s,
                     const int il_t, const int iu_t);

    // Evaluate interpolant on (full target) grid.
    // Utilizes stencils of reduced width.
    inline void eval_nn(Tf* fcn_t, const Tf* const fcn_s);

    // Evaluate interpolant over (target grid) indicial range.
    // Utilizes stencils of reduced width.
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    // {jl_t, ju_t} are elements of (0, Nt_x2)
    // {kl_t, ku_t} are elements of (0, Nt_x3)
    inline void eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                        const int il_t, const int iu_t,
                        const int jl_t, const int ju_t,
                        const int kl_t, const int ku_t);

    // Evaluate interpolant over (target grid) indicial range.
    // Utilizes stencils of reduced width.
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    // {jl_t, ju_t} are elements of (0, Nt_x2)
    inline void eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                        const int il_t, const int iu_t,
                        const int jl_t, const int ju_t);

    // Evaluate interpolant over (target grid) indicial range.
    // Utilizes stencils of reduced width.
    //
    // Note(s):
    // --------
    // {il_t, iu_t} are elements of (0, Nt_x1)
    inline void eval_nn(Tf* fcn_t, const Tf* const fcn_s,
                        const int il_t, const int iu_t);

  private:
    // compute and store weights together with reciprocal factors
    inline void precompute_weights(
      Tg* x_t, Tg* x_s,
      const int Nt, const int Ns,
      Tg* rwei);

    // compute and store weights together with reciprocal factors (nn only)
    inline void precompute_nn_weights(
      Tg* x_t, Tg* x_s,
      const int Nt, const int Ns,
      Tg* rwei_nn, size_t* ix_nn);

  private:
    const int ndim;

    const int Nt_x1, Nt_x2, Nt_x3;
    const int Ns_x1, Ns_x2, Ns_x3;

    const int ne {0};
    const int d;
    const int W;
    const int ng_t, ng_s;

    Tg *rwei_x1, *rwei_x2, *rwei_x3;
    Tg *rwei_nn_x1, *rwei_nn_x2, *rwei_nn_x3;
    size_t *ix_nn_x1, *ix_nn_x2, *ix_nn_x3;
    // ----------------------------------------------------

    // weight & function of common value
    typedef typename std::common_type<Tg, Tf>::type Tr;
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
static inline T weight(const T gr_t, const T * const gr_s, int Ns,
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
// Targets 2d see interp_1d
template<typename Tg, typename Tf>
static Tf interp_2d(Tg x1_t, Tg x2_t,
                    Tg * x1_s, Tg * x2_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int d, double gam, int ng=0);
// Targets 3d see interp_1d
template<typename Tg, typename Tf>
static Tf interp_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                    Tg * x1_s, Tg * x2_s, Tg * x3_s,
                    Tf * fcn_s,
                    int N_x1, int N_x2, int N_x3, int d, double gam,
                    int ng=0);

// Direct interpolation based on one-dimensional generalized Floater-Hormann
// barycentric form. Control on total number of nearest-neighbour points
// involved.
//
// Parameters:
// -----------
// x1_t:  Target base-point to interpolate to
// x1_s:  Source grid with N + 1 entries
// fcn_s: Source function samples with N + 1 entries
// Ns:    Source grid size parameter
// d:     FH blending parameter
// gam:   Rational power for barycentric form (reduced to FH for gam=1)
// W:     Number of points involved in stencil; nearest points to x_t selected
// ng=0:  Optionally control skipping of ghost-zone (i.e. control strides)
// ne=0:  Optionally control skipping of an interior edge zone
//
// Note(s):
// --------
//   - Target node is assumed distinct from fundamental (source) nodes.
//   - For ng!=0 it is assumed that x1_s is passed with first entry at x1_s[ng]
template<typename Tg, typename Tf>
static Tf interp_nn_1d(Tg x1_t, Tg * x1_s,
                       Tf * fcn_s,
                       int Ns,
                       int d, double gam, int W, int ng=0, int ne=0);

// Targets 2d see interp_nn_1d
template<typename Tg, typename Tf>
static Tf interp_nn_2d(Tg x1_t, Tg x2_t,
                       Tg * x1_s, Tg * x2_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2,
                       int d, double gam, int W, int ng=0, int ne=0);

// Targets 3d see interp_nn_1d
template<typename Tg, typename Tf>
static Tf interp_nn_3d(Tg x1_t, Tg x2_t, Tg x3_t,
                       Tg * x1_s, Tg * x2_s, Tg * x3_s,
                       Tf * fcn_s,
                       int Ns_x1, int Ns_x2, int Ns_x3,
                       int d, double gam, int W, int ng=0, int ne=0);

// ============================================================================
}
// ============================================================================

// ============================================================================
namespace shared {
// ============================================================================

// Given (sorted) grid values infer indices (with range W) of closest values to
// a given point.
//
// Parameters:
// -----------
// ix_np: Index corresponding to value in x_s nearest x_t; populated by
//        function.
// ix_il: Lower index corresponding to value in x_s nearest x_t; populated by
//        function.
// ix_iu: Upper index corresponding to value in x_s nearest x_t; populated by
//        function.
// gr_t:  Target point.
// gr_s:  Source grid with N_s + 1 entries
// N:     Grid size parameter
// width: Width of indicial range (iu-il+1) to construct.
// ne:    Number of nodes from edge to pad by.
//
// Note(s):
// --------
//   - x_s is assumed sorted.
template<typename Tg>
inline void idx_range_nearest_point(
  size_t* ix_np,
  size_t* ix_il,
  size_t* ix_iu,
  const Tg x_t,
  const Tg* x_s,
  const int Ns,
  const int width,
  const int ne=0);

// ============================================================================
}
// ============================================================================


// ============================================================================
namespace impl {
// ============================================================================

// Implementation details

// For standard barycentric representations.
template<typename Tg, typename Tf>
static void _D1(
  const Tf * const fcn_s,
  const Tg * const r1d_x1_wei,
  Tf * d1_fcn_t,
  const int Ns,
  const int ng=0);

template<typename Tg, typename Tf>
static inline void _D1(
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  const Tg * const r2d_x1_wei,
  Tf * d1_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int ng_t=0,
  const int ng_s=0);

template<typename Tg, typename Tf>
static void _D2(
  const Tf * const fcn_s,
  const Tf * const d1_fcn_s,
  const Tg * const r1d_x1,
  const Tg * const r2d_x1_wei,
  Tf * d1_fcn_t,
  const int Ns,
  const int ng=0);

template<typename Tg, typename Tf>
static void _D2(
  const Tf * const fcn_t,
  const Tf * const fcn_s,
  const Tf * const d1_fcn_t,
  const Tg * const r1d_x1,
  const Tg * const r2d_x1_wei,
  Tf * d2_fcn_t,
  const int Nt_x1,
  const int Ns_x1,
  const int ng_t=0,
  const int ng_s=0);


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