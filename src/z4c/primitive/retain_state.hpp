#ifndef RETAIN_STATE_HPP
#define RETAIN_STATE_HPP

/// @brief Defines retained-state terminal C2P failure recovery.

#include "reset_floor.hpp"

namespace Primitive
{

class RetainState : public ResetFloor
{
  protected:
  static constexpr RetainedFailureMode retained_failure_mode =
    RetainedFailureMode::state;

  RetainState();

  bool RetainedFailureResponse(Real prim[NPRIM],
                               Real n,
                               Real T,
                               const Real* Y,
                               int n_species);
};

}  // namespace Primitive

#endif

//
// :D
//
