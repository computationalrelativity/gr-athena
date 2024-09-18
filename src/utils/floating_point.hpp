#ifndef FLOATING_POINT_HPP_
#define FLOATING_POINT_HPP_
// TODO:
// Compensated summation:
//  - Kahan-Babushka-Neumaier compensated summation
//  - std::min({s1,s2,s3}) + std::max({s1,s2,s3})

// C headers

// C++ headers

// Athena++ headers
#include "../defs.hpp"
#include "../athena_aliases.hpp"

namespace FloatingPoint {

  // Average of summation orders to get back associativity for floats
  template<typename T>
  static inline T sum_associative(T v11, T v12, T v21, T v22)
  {
    const T ca = ((v11 + v12) + (v21 + v22));
    const T cb = ((v11 + v21) + (v12 + v22));
    const T cc = ((v11 + v22) + (v12 + v21));

    return ONE_6TH * (((ca + cb) + cc) + ((ca + cc) + cb));
  }

  template<typename T>
  inline T sum_associative(
    T v111, T v112, T v121, T v211, T v122, T v221, T v212, T v222
  )
  {
    return (
      sum_associative(v111, v112, v121, v211) +
      sum_associative(v122, v221, v212, v222)
    );
  }

  template<typename T>
  inline T KB_compensated(
    AthenaArray<T> & array,
    const int nl, const int nu,
    const int kl, const int ku,
    const int jl, const int ju,
    const int il, const int iu
  )
  {
    T sum = 0.0;  // accumulator
    T c = 0.0;    // lost low-order bit compensation

    for (int n=nl; n<=nu; ++n)
    for (int k=kl; k<=ku; ++k)
    for (int j=jl; j<=ju; ++j)
    for (int i=il; i<=iu; ++i)
    {
      const T y = array(n,k,j,i) - c;
      const T t = sum + y;

      // t-sum cancels high-order part of y
      // subtracting y recovers negative (low part of y)
      c = (t - sum) - y;
      sum = t;
    }

    return sum;
  }

  template<typename T>
  inline T KB_compensated(
    AthenaArray<T> & array,
    const int il, const int iu
  )
  {
    T sum = 0.0;  // accumulator
    T c = 0.0;    // lost low-order bit compensation

    for (int i=il; i<=iu; ++i)
    {
      const T y = array(i) - c;
      const T t = sum + y;

      // t-sum cancels high-order part of y
      // subtracting y recovers negative (low part of y)
      c = (t - sum) - y;
      sum = t;
    }

    return sum;
  }

} // namespace FloatingPoint
#endif // FLOATING_POINT_HPP_
