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
  AT_C_sca & sc_chi;
  AT_C_sca & sc_xi;

  // Lagrangian frame
  AT_C_sca & sc_J;
  AT_D_vec & st_H_u;

  // computer xi, chi based on input selection of opt_closure_variety
  void Closure(const int k, const int j, const int i);
};

ClosureMetaVector ConstructClosureMetaVector(
  M1 & pm1, M1::vars_Lab & vlab,
  const int ix_g, const int ix_s);

// ============================================================================
namespace utils {
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

// ============================================================================
} // namespace M1::Closures::utils
// ============================================================================

// ============================================================================
namespace solvers {
// ============================================================================

enum class status
{
  success,
  fail_tolerance_not_met,
  fail_bracket,
  fail_value,
  fail_unknown
};

// bump xi to live in [XI_MIN, XI_MAX]; return if bumped
bool Enforce_Xi_Limits(Real & xi);

// fix xi / chi at a limit, if required
void Fallback_Xi_Chi_Limits(M1 & pm1, ClosureMetaVector & C,
                            const int k, const int j, const int i);

// Function to find root of.
Real Z_xi(
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i);

Real dZ_xi(
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i);

void ZdZ_xi(
  Real & Z__,
  Real & dZ__,
  const Real xi__,
  M1 & pm1,
  ClosureMetaVector & C,
  const int k, const int j, const int i);

status gsl_Brent(M1 & pm1,
                 ClosureMetaVector & C,
                 const int k, const int j, const int i);

status gsl_Newton(M1 & pm1,
                  ClosureMetaVector & C,
                  const int k, const int j, const int i);

status custom_NB(M1 & pm1,
                 ClosureMetaVector & C,
                 const int k, const int j, const int i);

status custom_NAB(M1 & pm1,
                  ClosureMetaVector & C,
                  const int k, const int j, const int i);

status custom_ONAB(M1 & pm1,
                   ClosureMetaVector & C,
                   const int k, const int j, const int i);
// ============================================================================
} // namespace M1::Closures::solvers
// ============================================================================

// ============================================================================
namespace EddingtonFactors {
// ============================================================================

void ThinLimit(Real & xi, Real & chi);
void ThickLimit(Real & xi, Real & chi);
void Minerbo(Real & xi, Real & chi);
void Kershaw(Real & xi, Real & chi);

void Compute(M1 & pm1, Real & xi, Real & chi);

// ============================================================================
namespace D1 {
// ============================================================================

// First derivatives (wrt. xi);
// chi=chi(xi) -> chi'(xi)

void ThinLimit(Real & xi, Real & chi);
void ThickLimit(Real & xi, Real & chi);
void Minerbo(Real & xi, Real & chi);
void Kershaw(Real & xi, Real & chi);

void Compute(M1 & pm1, Real & xi, Real & dchi_dxi);
// ============================================================================
} // namespace M1::Closures::EddingtonFactors::D1
// ============================================================================

// ============================================================================
} // namespace M1::Closures::EddingtonFactors
// ============================================================================

// ============================================================================
} // namespace M1::Closures
// ============================================================================

#endif // M1_CALC_CLOSURE_HPP

