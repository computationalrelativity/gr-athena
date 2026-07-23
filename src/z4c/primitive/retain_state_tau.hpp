#ifndef RETAIN_STATE_TAU_HPP
#define RETAIN_STATE_TAU_HPP

/// @brief Defines energy-retaining terminal C2P failure recovery.

#include "retain_state.hpp"

namespace Primitive
{

class RetainStateTau : public RetainState
{
  protected:
  static constexpr RetainedFailureMode retained_failure_mode =
    RetainedFailureMode::state_tau;

  RetainStateTau();
};

}  // namespace Primitive

#endif

//
// :D
//
