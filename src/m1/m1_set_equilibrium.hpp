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

// ============================================================================
} // namespace M1::Equilibrium
// ============================================================================


#endif // M1_CALC_EQUILIBRIUM_HPP

