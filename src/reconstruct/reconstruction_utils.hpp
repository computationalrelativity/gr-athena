#ifndef RECONSTRUCTION_UTILS_
#define RECONSTRUCTION_UTILS_

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"

namespace reconstruction::utils {

inline Real sign(const Real x)
{
  return (x >= 0) ? 1. : -1.;
}

inline Real sign(const Real x, const Real y)
{
  return ((y) >= 0.0 ? std::abs(x) : -std::abs(x));
}

inline Real MC2(const Real x, const Real y)
{
  const Real s1  = sign(1, x);
  const Real s2  = sign(1, y);
  const Real min = 2 * std::min(std::abs(x), std::abs(y));

  return( 0.5*(s1+s2) * std::min( min, (0.5*std::abs(x+y)) ) );
}

}  // namespace reconstruction::utils

#endif  // RECONSTRUCTION_UTILS_
