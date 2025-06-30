#ifndef M1_CALC_EQUILIBRIUM_HPP
#define M1_CALC_EQUILIBRIUM_HPP

// c++
// ...

// Athena++ classes headers
#include "m1.hpp"
#include "m1_containers.hpp"
#include "m1_macro.hpp"
#include "m1_utils.hpp"
#include "m1_calc_closure.hpp"
#include "m1_calc_update.hpp"

// ============================================================================
namespace M1::Equilibrium {
// ============================================================================

void SetEquilibrium(
  M1 & pm1,
  M1::vars_Lab & U_C,
  M1::vars_Source & U_S,
  const int k,
  const int j,
  const int i);

// Enforce equilibrium only on species using energy averages.
//
// - Takes internal sc_avg_nrg (should be populated during opacity calc.)
// - Takes current J; Optionally fiducial frame is reconstructed at the point
//   based on current (E, F_d)
// - Expression: sc_avg_nrg = J / n
void SetEquilibrium_n_nG(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::SourceMetaVector & S,
  const int k,
  const int j,
  const int i,
  const bool construct_fiducial,
  const bool construct_src_nG,
  const bool construct_src_E_F_d,
  const bool use_diff_src
);

// Set also (E, F_d)
void SetEquilibrium_E_F_d_n_nG(
  M1 & pm1,
  Update::StateMetaVector & C,
  Update::StateMetaVector & P,
  Update::SourceMetaVector & S,
  const int k,
  const int j,
  const int i,
  const bool construct_src_nG,
  const bool construct_src_E_F_d,
  const bool use_diff_src
);


// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


#endif // M1_CALC_EQUILIBRIUM_HPP

