#ifndef UTILS_SPHERICAL_HARMONICS_HPP_
#define UTILS_SPHERICAL_HARMONICS_HPP_

#include <array>
#include <cmath>
#include <sstream>

#include "../athena.hpp"

namespace gra
{
namespace sph_harm
{

// n! is exactly representable in double for n <= 22 (since 22! < 2^53).
// For 23 <= n <= 35 the double representation is the nearest floating-point
// value, which is the same as what the original hand-written table contained.
constexpr int kMaxFactorialTable = 35;

namespace detail
{
constexpr std::array<Real, kMaxFactorialTable + 1> make_factorial_table()
{
  std::array<Real, kMaxFactorialTable + 1> t{};
  t[0] = 1.0;
  for (int i = 1; i <= kMaxFactorialTable; ++i)
    t[i] = t[i - 1] * static_cast<Real>(i);
  return t;
}
}  // namespace detail

constexpr std::array<Real, kMaxFactorialTable + 1> factorial_table_ =
  detail::make_factorial_table();

// Factorial n! for integer n >= 0.
// Uses a lookup table for n <= 35, recursion for n > 35.
inline Real Factorial(const int n)
{
  if (n < 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in gra::sph_harm::Factorial" << std::endl
        << "factorial requires integer nonnegative argument " << n
        << std::endl;
    ATHENA_ERROR(msg);
  }
  else if (n <= kMaxFactorialTable)
  {
    return factorial_table_[n];
  }
  else
  {
    return ((Real)n) * Factorial(n - 1);
  }
}

// Unnormalized associated Legendre polynomial P_l^m(x) with Condon-Shortley
// phase (-1)^m included. Requires 0 <= m <= l and -1 <= x <= 1.
// Computed via the standard three-term recurrence.
inline Real Plm(const int l, const int m, const Real x)
{
  Real pmm = 1.0;

  if (m >= 0)
  {
    Real somx2 = std::sqrt((1.0 - x) * (1.0 + x));
    Real fact  = 1.0;
    for (int i = 1; i <= m; ++i)
    {
      pmm = -pmm * fact * somx2;
      fact += 2.0;
    }
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in gra::sph_harm::Plm" << std::endl
        << "Plm requires nonnegative m argument " << m << std::endl;
    ATHENA_ERROR(msg);
  }

  if (l == m)
  {
    return pmm;
  }

  Real pmmp1 = x * (2.0 * m + 1.0) * pmm;

  if (l == (m + 1))
  {
    return pmmp1;
  }

  for (int i = m + 2; i <= l; ++i)
  {
    Real pll = (x * ((Real)(2 * i - 1)) * pmmp1 - ((Real)(i + m - 1)) * pmm) /
               ((Real)(i - m));
    pmm   = pmmp1;
    pmmp1 = pll;
  }
  return pmmp1;
}

// 4pi-normalized associated Legendre function:
//
//   NPlm(l, m, x) = sqrt( (2l+1)/(4pi) * (l-|m|)!/(l+|m|)! ) * P_l^|m|(x)
//
// Accepts negative m (uses |m| internally). This is the normalization used
// by AHF for real spherical harmonics.
inline Real NPlm(const int l, const int m, const Real x)
{
  const int abs_m = std::abs(m);
  const Real norm = std::sqrt((Real)(2 * l + 1) / (4.0 * PI) *
                              Factorial(l - abs_m) / Factorial(l + abs_m));
  return norm * Plm(l, abs_m, x);
}

// Complex scalar spherical harmonic Y_l^m(theta, phi).
//
//                    ( 2l+1 (l-|m|)! )^{1/2}                     i m phi
//   Y_l^m = (-1)^a  ( ---- -------- )        P_l^|m|(cos theta) e
//                    ( 4 pi (l+|m|)! )
//
// where a = m/2*(sign(m)+1), i.e. the Condon-Shortley phase applies
// only for m < 0.
//
// Returns the real and imaginary parts via YlmR and YlmI.
inline void Ylm(const int l,
                const int m,
                const Real theta,
                const Real phi,
                Real* YlmR,
                Real* YlmI)
{
  const int abs_m      = std::abs(m);
  const Real fact_norm = Factorial(l + abs_m) / Factorial(l - abs_m);

  const Real a   = std::sqrt((Real)(2 * l + 1) / (4.0 * PI * fact_norm));
  const int mfac = (m < 0) ? std::pow(-1.0, m) : 1.0;
  const Real P   = mfac * a * Plm(l, abs_m, std::cos(theta));

  *YlmR = P * std::cos((Real)(m)*phi);
  *YlmI = P * std::sin((Real)(m)*phi);
}

// Angular derivatives of Y_l^m and the vector/tensor spherical harmonics
// X_lm and W_lm.
//
// Outputs:
//   (YthR, YthI)  =  dY_lm / dtheta
//   (YphR, YphI)  =  dY_lm / dphi
//   (XR, XI)      =  X_lm  (magnetic-type vector harmonic)
//   (WR, WI)      =  W_lm  (electric-type tensor harmonic)
//
// Computed from the identity relating dY_lm/dtheta to Y_lm and Y_{l+1,m}
// via the recurrence on l.
inline void D_Ylm(const int l_,
                  const int m_,
                  const Real theta,
                  const Real phi,
                  Real* YthR,
                  Real* YthI,
                  Real* YphR,
                  Real* YphI,
                  Real* XR,
                  Real* XI,
                  Real* WR,
                  Real* WI)
{
  const Real l = (Real)l_;
  const Real m = (Real)m_;

  const Real div_sin_theta = 1.0 / (std::sin(theta));
  const Real cot_theta     = std::cos(theta) * div_sin_theta;

  const Real a = -(l + 1.0) * cot_theta;
  const Real b =
    std::sqrt((SQR(l + 1.0) - SQR(m)) * (l + 0.5) / (l + 1.5)) * div_sin_theta;

  Real YR, YI;
  Ylm(l, m, theta, phi, &YR, &YI);

  Real YplusR, YplusI;
  Ylm(l + 1, m, theta, phi, &YplusR, &YplusI);

  const Real _YthR = a * YR + b * YplusR;
  const Real _YthI = a * YI + b * YplusI;

  const Real c = -2.0 * cot_theta;
  const Real d = (2.0 * SQR(m * div_sin_theta) - l * (l + 1.0));

  *YthR = _YthR;
  *YthI = _YthI;

  *YphR = -m * YI;
  *YphI = m * YR;

  *WR = c * (*YthR) + d * YR;
  *WI = c * (*YthI) + d * YI;

  *XR = 2.0 * m * (cot_theta * YI - _YthI);
  *XI = 2.0 * m * (_YthR - cot_theta * YR);
}

// Spin-weighted spherical harmonic _sY_l^m(theta, phi) for arbitrary
// integer spin weight s, computed via the Wigner-d matrix formulation
// (see Eq. II.7, II.8 in arXiv:0709.0093).
//
//                   (2l+1)^{1/2}
//   _sY_l^m(th,ph) = (---------)  d^l_{m,-s}(th) e^{i m ph}
//                   (  4 pi   )
//
// where the Wigner (small) d-matrix element is:
//
//   d^l_{m,-s}(th) = sum_k (-1)^k * [C(l,m,s,k)] *
//                    cos(th/2)^{2l+m+s-2k} * sin(th/2)^{2k-m-s}
//
// with combinatorial prefactor C involving factorials.
//
// Symmetry (bitant reflection):
//   _sY_l^m(pi - theta, phi) = (-1)^{l+s} _sY_l^{-m}(theta, phi)
//
// This identity is used for bitant-symmetric grids: the harmonics in
// the southern hemisphere (theta > pi/2) are related to those in the
// northern hemisphere with m -> -m and a sign factor (-1)^{l+s}.
//
// Returns the real and imaginary parts via YR and YI.
inline void sYlm(const int s,
                 const int l,
                 const int m,
                 const Real theta,
                 const Real phi,
                 Real* YR,
                 Real* YI)
{
  Real wignerd = 0;
  const int k1 = std::max(0, m + s);
  const int k2 = std::min(l + m, l + s);
  for (int k = k1; k <= k2; ++k)
  {
    wignerd += std::pow(-1.0, k) *
               std::sqrt(Factorial(l + m) * Factorial(l - m) *
                         Factorial(l - s) * Factorial(l + s)) *
               std::pow(std::cos(theta / 2.0), 2 * l + m + s - 2 * k) *
               std::pow(std::sin(theta / 2.0), 2 * k - m - s) /
               (Factorial(l + m - k) * Factorial(l + s - k) * Factorial(k) *
                Factorial(k - m - s));
  }
  const Real norm = std::sqrt((2 * l + 1) / (4.0 * PI));
  *YR             = norm * wignerd * std::cos(m * phi);
  *YI             = norm * wignerd * std::sin(m * phi);
}

// Batch computation of 4pi-normalized associated Legendre functions and their
// first and second theta-derivatives for all 0 <= m <= l <= lmax.
//
// The 4pi-normalized Legendre function is:
//
//   P(l, m) = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! ) * (-1)^m * P_l^m(cos th)
//
// where P_l^m is the standard (unnormalized) associated Legendre function with
// Condon-Shortley phase. The diagonal seeds P(l,l) are computed directly from
// the closed-form expression, and the off-diagonal entries use the three-term
// recurrence on l. Derivatives dPdth and dPdth2 are obtained by
// differentiating the recurrence (product rule), yielding a coupled recurrence
// that depends on P and dPdth values from prior rows.
//
// The caller must pre-allocate P, dPdth, dPdth2 as (lmax+1) * (lmax+1)
// AthenaArrays before calling.
inline void NPlm(const Real theta,
                 const int lmax,
                 AthenaArray<Real>& P,
                 AthenaArray<Real>& dPdth,
                 AthenaArray<Real>& dPdth2)
{
  const int lmax1  = lmax + 1;
  const Real costh = std::cos(theta);
  const Real sinth = std::sin(theta);

  P.ZeroClear();
  dPdth.ZeroClear();
  dPdth2.ZeroClear();

  Real fac[2 * lmax1 + 1];
  fac[0] = 1.0;
  for (int i = 1; i <= 2 * lmax; ++i)
    fac[i] = fac[i - 1] * i;

  // Diagonal seeds: P(l,l)
  for (int l = 0; l <= lmax; ++l)
  {
    P(l, l) = std::sqrt((2 * l + 1) * fac[2 * l] / (4.0 * PI)) /
              (std::pow(2, l) * fac[l]) * std::pow((-sinth), l);
  }

  // Off-diagonal recurrence: P(l,m) for m < l
  P(1, 0) = SQRT3 * costh * P(0, 0);
  for (int l = 2; l <= lmax; l++)
  {
    int m;
    for (m = 0; m < l - 1; m++)
    {
      P(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
      P(l, m) *= (std::sqrt((Real)2 * l - 1) * costh * P(l - 1, m) -
                  std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
                    P(l - 2, m));
    }
    P(l, l - 1) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m)) *
                  (std::sqrt((Real)2 * l - 1) * costh * P(l - 1, m));
  }

  // First theta-derivatives: diagonal seeds
  for (int l = 0; l <= lmax; l++)
  {
    dPdth(l, l) = std::sqrt((2 * l + 1) * fac[2 * l] / (4.0 * PI)) /
                  (std::pow(2, l) * fac[l]) * l * std::pow((-sinth), l - 1) *
                  (-costh);
  }

  // First theta-derivatives: off-diagonal recurrence
  dPdth(1, 0) = SQRT3 * (-sinth * P(0, 0) + costh * dPdth(0, 0));

  for (int l = 2; l <= lmax; l++)
  {
    int m;
    for (m = 0; m < l - 1; m++)
    {
      dPdth(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
      dPdth(l, m) *=
        (std::sqrt((Real)2 * l - 1) *
           (-sinth * P(l - 1, m) + costh * dPdth(l - 1, m)) -
         std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
           dPdth(l - 2, m));
    }
    dPdth(l, l - 1) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
    dPdth(l, l - 1) *= (std::sqrt((Real)2 * l - 1) *
                        (-sinth * P(l - 1, m) + costh * dPdth(l - 1, m)));
  }

  // Second theta-derivatives: diagonal seeds
  for (int l = 0; l <= lmax; l++)
  {
    dPdth2(l, l) = std::sqrt((Real)(2 * l + 1) * fac[2 * l] / (4.0 * PI)) /
                   (std::pow(2, l) * fac[l]) * l *
                   ((l - 1) * std::pow(-sinth, l - 2) * costh * costh +
                    std::pow(-sinth, l - 1) * sinth);
  }

  // Second theta-derivatives: off-diagonal recurrence
  dPdth2(1, 0) = SQRT3 * (-costh * P(0, 0) - 2.0 * sinth * dPdth(0, 0) +
                          costh * dPdth2(0, 0));

  for (int l = 2; l <= lmax; l++)
  {
    int m;
    for (m = 0; m < l - 1; m++)
    {
      dPdth2(l, m) =
        std::sqrt((Real)(2 * l + 1) / (l * l - m * m)) *
        (std::sqrt((Real)2 * l - 1) *
           (-costh * P(l - 1, m) - 2.0 * sinth * dPdth(l - 1, m) +
            costh * dPdth2(l - 1, m)) -
         std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
           dPdth2(l - 2, m));
    }
    dPdth2(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
    dPdth2(l, m) *= (std::sqrt((Real)2 * l - 1) *
                     (-costh * P(l - 1, m) - 2.0 * sinth * dPdth(l - 1, m) +
                      costh * dPdth2(l - 1, m)));
  }
}

}  // namespace sph_harm
}  // namespace gra

#endif  // UTILS_SPHERICAL_HARMONICS_HPP_
