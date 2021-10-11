#ifndef INCLUDE_EOS_HPP
#define INCLUDE_EOS_HPP

#include "../athena.hpp"

#if EOS_POLICY == IdealGas
#include "../z4c/primitive/idealgas.hpp"
#elif EOS_POLICY == PiecewisePolytrope
#include "../z4c/primitive/piecewise_polytrope.hpp"
#endif

#if ERROR_POLICY == DoNothing
#include "../z4c/primitive/do_nothing.hpp"
#elif ERROR_POLICY == ResetFloor
#include "../z4c/primitive/reset_floor.hpp"
#endif

#endif
