#ifndef TENSOR_SYMMETRY_HPP_
#define TENSOR_SYMMETRY_HPP_

//========================================================================================
// GR-Athena++
//========================================================================================
//! \file tensor_symmetry.hpp
//  \brief Utility helpers for tensor index symmetry operations (bitant, etc.)

// Bitant symmetry sign factor: (-1)^(count of z-indices).
// Returns 1 if bitant_sym is false.
// Usage: BitantSign(bitant_sym, a, b) for rank-2, (a, b, c) for rank-3, etc.
template <typename... Ints>
inline int BitantSign(bool bitant_sym, Ints... indices)
{
  if (!bitant_sym)
    return 1;
  int sign = 1;
  ((sign *= (indices == 2) ? -1 : 1), ...);
  return sign;
}

#endif  // TENSOR_SYMMETRY_HPP_
