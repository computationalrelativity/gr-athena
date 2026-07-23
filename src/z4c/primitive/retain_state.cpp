/// @brief Implements retained-state terminal C2P failure recovery.

#include "retain_state.hpp"

namespace Primitive
{

RetainState::RetainState() = default;

bool RetainState::RetainedFailureResponse(Real prim[NPRIM],
                                          Real n,
                                          Real T,
                                          const Real* Y,
                                          int n_species)
{
  prim[IDN] = n;
  prim[IVX] = 0.0;
  prim[IVY] = 0.0;
  prim[IVZ] = 0.0;
  prim[ITM] = T;
  for (int s = 0; s < n_species; ++s)
  {
    prim[IYF + s] = Y[s];
  }
  return true;
}

}  // namespace Primitive

//
// :D
//
