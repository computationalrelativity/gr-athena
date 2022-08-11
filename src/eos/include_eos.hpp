#ifndef INCLUDE_EOS_HPP
#define INCLUDE_EOS_HPP

#include "../athena.hpp"
// TODO: This horrible construction needs to go the way of the dinosaur. It's
// not user-friendly and it's an eyesore, but it's the only way I could think
// to make this work. - JF

#if EOS_POLICY_CODE == 0
  #define USE_IDEAL_GAS
  //#pragma message("EOS_POLICY is IdealGas")
  #include "../z4c/primitive/idealgas.hpp"
#elif EOS_POLICY_CODE == 1
  #define USE_PIECEWISE_POLY
  //#pragma message("EOS_POLICY is PiecewisePolytrope")
  #include "../z4c/primitive/piecewise_polytrope.hpp"
#elif EOS_POLICY_CODE == 2
  #ifndef HDF5OUTPUT
    #error "HDF5 must be enabled to use EOSCompOSE."
  #endif
  #define USE_COMPOSE_EOS
  #include "../z4c/primitive/eos_compose.hpp"
#else
  #error EOS_POLICY_CODE not recognized.
#endif

#if ERROR_POLICY_CODE == 0
  //#pragma message("ERROR_POLICY is DoNothing")
  #include "../z4c/primitive/do_nothing.hpp"
#elif ERROR_POLICY_CODE == 1
  //#pragma message("ERROR_POLICY is ResetFloor")
  #include "../z4c/primitive/reset_floor.hpp"
#else
  #error ERROR_POLICY_CODE not recognized.
#endif

#endif
