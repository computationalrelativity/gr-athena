#ifndef M1_CALC_CLOSURE_HPP
#define M1_CALC_CLOSURE_HPP

// c++
#include <functional>

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"


// ============================================================================
namespace M1::Closures {
// ============================================================================

struct Bracket {
  Real a;
  Real b;

  inline bool sign_change()
  {
    return a * b < 0;
  }

  inline bool bounds(const Real x)
  {
    return (a <= x) && (x <= b);
  }

  inline bool bounds_strict(const Real x)
  {
    return (a < x) && (x < b);
  }

  inline Real midpoint()
  {
    return a + 0.5 * (b-a);
  }

  inline Real squeeze(const Real x)
  {
    return std::min(std::max(x, a), b);
  }
};

enum class status_closure {
  success,
  fail_tolerance_not_met,
  fail_fallback,
  fail_unknown
};

struct ClosureMetaVector {
  M1 & pm1;
  const int ix_g;
  const int ix_s;

  // geometric quantities
  AT_N_sym & sp_g_dd;
  AT_N_sym & sp_g_uu;

  // fiducial quantities
  AT_N_vec & sp_v_d;
  AT_N_vec & sp_v_u;
  AT_C_sca & sc_W;

  // state-vector dependent
  AT_C_sca & sc_E;
  AT_N_vec & sp_F_d;

  // group-dependent, but common
  AT_N_sym & sp_P_dd;
  AT_C_sca & sc_chi;
  AT_C_sca & sc_xi;

  // Lagrangian frame
  AT_C_sca & sc_J;
  AT_C_sca & sc_H_t;
  AT_N_vec & sp_H_d;

  // scratch
  AT_N_vec & sp_dH_d_;
  AT_N_sym & sp_P_tn_dd_;  // ClosureThin with populate_scratch
  AT_N_sym & sp_P_tk_dd_;  // ClosureThick
  AT_N_sym & sp_dP_dd_;

  // Store state prior to iteration (for fallback)
  std::array<Real, 1> U_0_xi;

  void FallbackStore(const int k, const int j, const int i)
  {
    U_0_xi[0] = sc_xi(k,j,i);
  }

  void Fallback(const int k, const int j, const int i)
  {
    sc_xi(k,j,i) = U_0_xi[0];
  }

  // For methods requiring a bracket
  Bracket br_xi;
  Bracket br_Z;

  // Closure functions
  typedef std::function<void(const int k, const int j, const int i)> fcn_kji;

  fcn_kji Closure;      // selected
  fcn_kji ClosureThin;  // fallback
  fcn_kji ClosureThick;

};

ClosureMetaVector ConstructClosureMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s);

void ClosureThin(M1 & pm1,
                 ClosureMetaVector & C,
                 const int k, const int j, const int i,
                 const bool populate_scratch);

void ClosureThick(M1 & pm1,
                  ClosureMetaVector & C,
                  const int k, const int j, const int i,
                  const bool populate_scratch);

// ============================================================================
namespace Minerbo {
// ============================================================================

// Take min(max(xi, xi_min), xi_max).
//
// If enforced recompute P_dd and return true, otherwise false
bool EnforceClosureLimits(
  M1 & pm1, ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd);

// Function to find root of.
// Warning: P_dd, xi, chi, etc are modified in par struct
Real Z_xi(
  const Real xi,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd);

Real dZ_xi(
  const Real xi,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i,
  const bool compute_limiting_P_dd);

// field definitions ----------------------------------------------------------

// Eddington factor
inline Real chi(const Real xi)
{
  const Real xi2 = SQR(xi);
  return ONE_3RD + xi2 / 15.0 * (6.0 - 2.0 * xi + 6 * xi2);
}

inline void sp_P_dd_(
  AT_N_sym & sp_tar_dd,
  const AT_C_sca & sc_chi,
  const AT_N_sym & sp_P_tn_dd_,
  const AT_N_sym & sp_P_tk_dd_,
  const int k, const int j,
  const int il, const int iu)
{

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  #pragma omp simd
  for (int i=il; i<=iu; ++i)
  {
    sp_tar_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * sc_chi(k,j,i) - 1.0) * sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - sc_chi(k,j,i)) * sp_P_tk_dd_(a,b,i)
    );
  }
}

inline void sp_P_dd__(
  AT_N_sym & sp_tar_dd,
  const AT_C_sca & sc_chi,
  const AT_N_sym & sp_P_tn_dd_,
  const AT_N_sym & sp_P_tk_dd_,
  const int k, const int j, const int i)
{

  for (int a=0; a<N; ++a)
  for (int b=a; b<N; ++b)
  {
   sp_tar_dd(a,b,k,j,i) = 0.5 * (
      (3.0 * sc_chi(k,j,i) - 1.0) * sp_P_tn_dd_(a,b,i) +
      3.0 * (1.0 - sc_chi(k,j,i)) * sp_P_tk_dd_(a,b,i)
    );
  }
}

void ClosureMinerboPicard(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i);

void ClosureMinerboBisection(M1 & pm1,
                             ClosureMetaVector & C,
                             const int k, const int j, const int i);

void ClosureMinerboNewton(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i);

// ============================================================================
namespace gsl {
// ============================================================================

void ClosureMinerboBrent(M1 & pm1,
                         ClosureMetaVector & C,
                         const int k, const int j, const int i);

void ClosureMinerboNewton(M1 & pm1,
                          ClosureMetaVector & C,
                          const int k, const int j, const int i);

// ============================================================================
} // namespace M1::Closures::Minerbo::gsl
// ============================================================================

// ============================================================================
} // namespace M1::Closures::Minerbo
// ============================================================================

// ============================================================================
} // namespace M1::Closures
// ============================================================================

#endif // M1_CALC_CLOSURE_HPP

