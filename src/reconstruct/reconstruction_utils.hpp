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

inline Real min(const Real a, const Real b, const Real c)
{
  return std::min(std::min(a, b), c);
}

inline Real max(const Real a, const Real b, const Real c)
{
  return std::max(std::max(a, b), c);
}

inline Real min_abs(const Real a, const Real b)
{
  return std::min(std::abs(a), std::abs(b));
}

inline Real min_abs(const Real a, const Real b,
                    const Real c, const Real d)
{
  return std::min(
    std::min(std::abs(a), std::abs(b)),
    std::min(std::abs(c), std::abs(d))
  );
}

inline Real min_abs(const Real a, const Real b,
                    const Real c, const Real d,
                    const Real e, const Real f)
{
  const Real min_4 = std::min(
    std::min(std::abs(a), std::abs(b)),
    std::min(std::abs(c), std::abs(d))
  );
  const Real min_2 = std::min(std::abs(e), std::abs(f));
  return std::min(min_2, min_4);
}

inline Real minmod(const Real x, const Real y)
{
  const Real oo2 = 0.5;
  return oo2 * (sign(x) + sign(y)) * min_abs(x, y);
}

inline Real minmod(const Real w, const Real x, const Real y, const Real z)
{
  const Real oo8 = 1.0 / 8.0;
  return oo8 * (sign(w) + sign(x)) * std::abs(
    (sign(w) + sign(y)) * (sign(w) + sign(z))
  ) * min_abs(w, x, y, z);
}

inline Real minmod(const Real z1, const Real z2,
                   const Real z3, const Real z4,
                   const Real z5, const Real z6)
{
  const Real oo32 = 1.0 / 32.0;

  return oo32 * (sign(z1) + sign(z2)) * std::abs(
    (sign(z1) + sign(z3)) *
    (sign(z1) + sign(z4)) *
    (sign(z1) + sign(z5)) *
    (sign(z1) + sign(z6))
  ) * min_abs(z1, z2, z3, z4, z5, z6);
}

}  // namespace reconstruction::utils

#endif  // RECONSTRUCTION_UTILS_
