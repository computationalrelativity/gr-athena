// c++
#include <iostream>

// Athena++ headers
#include "m1_calc_fluxes.hpp"
#include "../reconstruct/reconstruction.hpp"

// ============================================================================
namespace M1::Fluxes {
// ============================================================================

// N.B.
// Eigenvalues and closure are assembled at interfaces through nn average.

void RiemannHLLEmod(
  M1 * pm1,
  AA & u,
  const bool use_lo
)
{
  // DEBUG --------------------------------------------------------------------
  assert(false); // not implemented, probably never will be
}

// ============================================================================
} // namespace M1::Fluxes
// ============================================================================

//
// :D
//