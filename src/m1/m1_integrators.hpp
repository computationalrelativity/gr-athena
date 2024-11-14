#ifndef M1_INTEGRATORS_HPP
#define M1_INTEGRATORS_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_update.hpp"
#include "m1_calc_closure.hpp"
#include "m1_sources.hpp"

// External libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

// ============================================================================
namespace M1::Integrators {
// ============================================================================

// Dispatch integrator method (sources are _not_ limited)
void DispatchIntegrationMethod(
  M1 & pm1,
  const Real dt,
  M1::vars_Lab & U_C,        // current (target) step
  const M1::vars_Lab & U_P,  // previous step data
  const M1::vars_Lab & U_I,  // inhomogeneity
  M1::vars_Source & U_S      // for construction of matter source contribution
);

// ============================================================================
namespace Explicit {
// ============================================================================

// Evolution of U ~ (E, F_d):
// U* <- U + dt * [ -div[F_A[U]] + G_A[U] + S_A[U] ]
//
// Note:
// - Internally U* is enforced to be non-zero & physical (causal)
// - Closures are _not_ internally computed
void StepExplicit_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  const Update::SourceMetaVector & S,  // carries matter source contribution
  const int k, const int j, const int i);

// Evolution of U ~ (nG, ):
// U* <- U + dt * [ -div[F_A[U]] + G_A[U] + S_A[U] ]
//
// Note:
// - Closures are _not_ internally computed
void StepExplicit_nG(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,         // current (target) step
  const Update::StateMetaVector & P,   // previous step data
  const Update::StateMetaVector & I,   // inhomogeneity
  const Update::SourceMetaVector & S,  // carries matter source contribution
  const int k, const int j, const int i);


// Given (an explicitly evolved) state U* ~ (E*, F_d*) prepare O(v)
// approximation of solution to the implicit system.
//
// Note:
// - Propagate with StepExplicit_E_F_d, then apply this
// - Internally U* is enforced to be non-zero & physical (causal)
// - Closures are _not_ internally computed
void PrepareApproximateFirstOrder_E_F_d(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & V, // state to utilize
  const int k, const int j, const int i);

// ============================================================================
} // namespace M1::Integrators::Explicit
// ============================================================================

// ============================================================================
namespace Implicit {
// ============================================================================

// Evolution of U ~ (nG, ) via explicit solution of:
// U* <- U + dt * [ -div[F_A[U]] + G_A[U] + S_A[U*] ]
//
// Note:
// - (sc_E*, sp_E*) needs to be available (in C)
// - (sc_n*, ) is internally computed
// - Sources are internally updated
void SolveImplicitNeutrinoCurrent(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  const int k, const int j, const int i);

// Prepare initial guess (sc_E, sp_F_d) -> (sc_E^, sp_F_d^);
// O(v) approx for implicit solver.
//
// See \S3.2.4 of [1]
void StepImplicitPrepareInitialGuess(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i);

// ============================================================================
namespace gsl {
// ============================================================================

// Evolution of U ~ (E, F_d) utilizing FD approximant to Jacobian via gsl:
// U* <- U + dt * [ -div[F_A[U]] + G_A[U] + S_A[U*] ]
//
// Note:
// - Internally U* is enforced via `EnforcePhysical_E_F_d`
// - Closures are internally updated
// - Sources are internally updated
void StepImplicitHybrids(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i);

// Evolution of U ~ (E, F_d) utilizing Jacobian via gsl:
// U* <- U + dt * [ -div[F_A[U]] + G_A[U] + S_A[U*] ]
//
// Note:
// - Internally U* is enforced via `EnforcePhysical_E_F_d`
// - Closures are internally updated
// - Sources are internally updated
void StepImplicitHybridsJ(
  M1 & pm1,
  const Real dt,
  Update::StateMetaVector & C,        // current step
  const Update::StateMetaVector & P,  // previous step data
  const Update::StateMetaVector & I,  // inhomogeneity
  Update::SourceMetaVector & S,       // carry source contribution
  Closures::ClosureMetaVector & CL_C,
  const int k, const int j, const int i);

// ============================================================================
} // namespace M1::Integrators::Implicit::gsl
// ============================================================================

// ============================================================================
} // namespace M1::Integrators::Implicit
// ============================================================================

// ============================================================================
} // namespace M1::Integrators
// ============================================================================


#endif // M1_INTEGRATORS_HPP

