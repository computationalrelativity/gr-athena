#ifndef UTILS_SPHERICAL_HARMONICS_HPP_
#define UTILS_SPHERICAL_HARMONICS_HPP_

#include <array>
#include <cmath>
#include <sstream>
#include <vector>

#include "../athena.hpp"
#include "grid_theta_phi_fields.hpp"

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
// Uses a lookup table for n <= kMaxFactorialTable, iterative computation
// otherwise.
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
    Real result = factorial_table_[kMaxFactorialTable];
    for (int i = kMaxFactorialTable + 1; i <= n; ++i)
      result *= static_cast<Real>(i);
    return result;
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
  const int abs_m = std::abs(m);
  const int sign  = (m < 0 && (abs_m % 2 != 0)) ? -1 : 1;
  const Real P    = sign * NPlm(l, abs_m, std::cos(theta));

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
    wignerd += ((k % 2 == 0) ? 1.0 : -1.0) *
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

  std::vector<Real> fac(2 * lmax1 + 1);
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
  if (lmax >= 1)
    P(1, 0) = SQRT3 * costh * P(0, 0);
  for (int l = 2; l <= lmax; l++)
  {
    for (int m = 0; m < l - 1; m++)
    {
      P(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
      P(l, m) *= (std::sqrt((Real)2 * l - 1) * costh * P(l - 1, m) -
                  std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
                    P(l - 2, m));
    }
    const int m = l - 1;
    P(l, l - 1) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m)) *
                  (std::sqrt((Real)2 * l - 1) * costh * P(l - 1, m));
  }

  // First theta-derivatives: diagonal seeds
  // l=0: d/dtheta of constant is identically zero (avoids 0 * pow(0,-1) = NaN)
  dPdth(0, 0) = 0.0;
  // l=1: d/dtheta[norm * (-sin theta)] = norm * (-cos theta)
  //       (avoids pow(0, 0) reliance at exact poles)
  if (lmax >= 1)
  {
    dPdth(1, 1) =
      std::sqrt(3.0 * fac[2] / (4.0 * PI)) / (2.0 * fac[1]) * (-costh);
  }
  for (int l = 2; l <= lmax; l++)
  {
    dPdth(l, l) = std::sqrt((2 * l + 1) * fac[2 * l] / (4.0 * PI)) /
                  (std::pow(2, l) * fac[l]) * l * std::pow((-sinth), l - 1) *
                  (-costh);
  }

  // First theta-derivatives: off-diagonal recurrence
  if (lmax >= 1)
    dPdth(1, 0) = SQRT3 * (-sinth * P(0, 0) + costh * dPdth(0, 0));

  for (int l = 2; l <= lmax; l++)
  {
    for (int m = 0; m < l - 1; m++)
    {
      dPdth(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
      dPdth(l, m) *=
        (std::sqrt((Real)2 * l - 1) *
           (-sinth * P(l - 1, m) + costh * dPdth(l - 1, m)) -
         std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
           dPdth(l - 2, m));
    }
    const int m     = l - 1;
    dPdth(l, l - 1) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
    dPdth(l, l - 1) *= (std::sqrt((Real)2 * l - 1) *
                        (-sinth * P(l - 1, m) + costh * dPdth(l - 1, m)));
  }

  // Second theta-derivatives: diagonal seeds
  // l=0: d^2/dtheta^2 of constant is identically zero
  dPdth2(0, 0) = 0.0;
  // l=1: general formula has 0 * pow(0,-1) at poles; evaluate analytically:
  //   d^2/dtheta^2 [(-sin theta)^1] = d/dtheta[-cos theta] = sin theta
  if (lmax >= 1)
  {
    dPdth2(1, 1) =
      std::sqrt(3.0 * fac[2] / (4.0 * PI)) / (2.0 * fac[1]) * sinth;
  }
  for (int l = 2; l <= lmax; l++)
  {
    dPdth2(l, l) = std::sqrt((Real)(2 * l + 1) * fac[2 * l] / (4.0 * PI)) /
                   (std::pow(2, l) * fac[l]) * l *
                   ((l - 1) * std::pow(-sinth, l - 2) * costh * costh +
                    std::pow(-sinth, l - 1) * sinth);
  }

  // Second theta-derivatives: off-diagonal recurrence
  if (lmax >= 1)
    dPdth2(1, 0) = SQRT3 * (-costh * P(0, 0) - 2.0 * sinth * dPdth(0, 0) +
                            costh * dPdth2(0, 0));

  for (int l = 2; l <= lmax; l++)
  {
    for (int m = 0; m < l - 1; m++)
    {
      dPdth2(l, m) =
        std::sqrt((Real)(2 * l + 1) / (l * l - m * m)) *
        (std::sqrt((Real)2 * l - 1) *
           (-costh * P(l - 1, m) - 2.0 * sinth * dPdth(l - 1, m) +
            costh * dPdth2(l - 1, m)) -
         std::sqrt((Real)((l - 1) * (l - 1) - m * m) / (2 * l - 3)) *
           dPdth2(l - 2, m));
    }
    const int m  = l - 1;
    dPdth2(l, m) = std::sqrt((Real)(2 * l + 1) / (l * l - m * m));
    dPdth2(l, m) *= (std::sqrt((Real)2 * l - 1) *
                     (-costh * P(l - 1, m) - 2.0 * sinth * dPdth(l - 1, m) +
                      costh * dPdth2(l - 1, m)));
  }
}

// Multipolar single index (l,m) -> linear index for real spherical harmonics.
// Used by AHF and related codes that expand in a0(l) * Y0 + ac(lm) * Yc +
// as(lm) * Ys.
inline int lmindex_real(int l, int m, int lmax)
{
  return l * (lmax + 1) + m;
}

// Derivative multi-index for the leading dimension of Y0, Yc, Ys.
// Encodes (dth, dph) pairs: D10 = d/dtheta, D01 = d/dphi, etc.
// Theta-only derivatives (D00, D10, D20) are first so that Y0, which
// stores only m=0 harmonics (no phi dependence), can use a leading
// dimension of 3.
namespace ix_D
{
constexpr int D00    = 0;
constexpr int D10    = 1;
constexpr int D20    = 2;
constexpr int D01    = 3;
constexpr int D11    = 4;
constexpr int D02    = 5;
constexpr int NDERIV = 6;
}  // namespace ix_D

// ============================================================================
// Complex-harmonic index helpers  (l = 2 .. lmax, m = -l .. +l)
// ============================================================================

// Real / Imaginary component index for complex harmonics.
namespace ix_C
{
constexpr int Re  = 0;
constexpr int Im  = 1;
constexpr int NRI = 2;
}  // namespace ix_C

// Number of complex (l,m) modes for l = 2 .. lmax.
//   sum_{l=2}^{lmax} (2l+1) = lmax*(lmax+2) - 3
inline int lmpoints_complex(int lmax)
{
  return lmax * (lmax + 2) - 3;
}

// Single linear index for a complex (l,m) mode, l >= 2, -l <= m <= l.
//   Offset of block l is (l-1)*(l+1) - 3, then m+l within the block.
inline int lmindex_complex(int l, int m)
{
  return (l - 1) * ((l - 1) + 2) - 3 + (m + l);
}

// ============================================================================
// ComplexHarmonicTable
// ============================================================================
//! \brief Pre-tabulated complex spherical harmonics, angular derivatives, and
//!        tensor harmonics (W, X) on a (theta, phi) grid.
//!
//! Stores Y, dY/dtheta, dY/dphi, W_lm, X_lm for l = 2 .. lmax, m = -l .. +l,
//! with Re/Im in the trailing dimension (index 0 = Re, 1 = Im).
//!
//! Access pattern:
//!   Y(i, j, lm, c)    -- value of Y_l^m at grid point (i,j), component c
//!   Yth(i, j, lm, c)  -- dY/dtheta
//!   Yph(i, j, lm, c)  -- dY/dphi = i m Y
//!   W(i, j, lm, c)    -- electric tensor harmonic
//!   X(i, j, lm, c)    -- magnetic tensor harmonic
//!
//! Projection methods perform full-grid integrals and accumulate into a
//! caller-supplied buffer (typically a slice of integrals_multipoles).
struct ComplexHarmonicTable
{
  int lmin     = 2;
  int lmax     = 0;
  int lmpoints = 0;
  int ntheta   = 0;
  int nphi     = 0;

  AthenaArray<Real> Y, Yth, Yph, W, X;  // (ntheta, nphi, lmpoints, 2)

  int lmindex(int l, int m) const
  {
    return lmindex_complex(l, m);
  }

  // --------------------------------------------------------------------------
  //! Allocate arrays and batch-compute all harmonics + derivatives.
  //!
  //! Uses the NPlm recurrence to compute P(l,m) and dP(l,m)/dth for all
  //! (l,m) at each th, then builds Y, Yth, Yph, W, X from closed-form
  //! formulas involving P, dP, cos(mph), sin(mph), and trig of th.
  // --------------------------------------------------------------------------
  template <typename GridType>
  void Initialize(const GridType& grid, int lmax_in)
  {
    lmax     = lmax_in;
    lmpoints = lmpoints_complex(lmax);
    ntheta   = grid.ntheta;
    nphi     = grid.nphi;

    Y.NewAthenaArray(ntheta, nphi, lmpoints, ix_C::NRI);
    Yth.NewAthenaArray(ntheta, nphi, lmpoints, ix_C::NRI);
    Yph.NewAthenaArray(ntheta, nphi, lmpoints, ix_C::NRI);
    W.NewAthenaArray(ntheta, nphi, lmpoints, ix_C::NRI);
    X.NewAthenaArray(ntheta, nphi, lmpoints, ix_C::NRI);

    Y.ZeroClear();
    Yth.ZeroClear();
    Yph.ZeroClear();
    W.ZeroClear();
    X.ZeroClear();

    // Legendre scratch - P(l,m) and dP(l,m)/dth for 0 <= m <= l <= lmax.
    // We only need first derivatives for the complex harmonics (no d2P).
    AthenaArray<Real> P, dP, d2P_unused;
    P.NewAthenaArray(lmax + 1, lmax + 1);
    dP.NewAthenaArray(lmax + 1, lmax + 1);
    d2P_unused.NewAthenaArray(lmax + 1, lmax + 1);

    for (int i = 0; i < ntheta; ++i)
    {
      const Real theta = grid.th_grid(i);
      NPlm(theta, lmax, P, dP, d2P_unused);

      const Real sinth     = grid.sin_theta(i);
      const Real costh     = grid.cos_theta(i);
      const Real div_sinth = 1.0 / sinth;
      const Real cot_theta = costh * div_sinth;

      for (int j = 0; j < nphi; ++j)
      {
        const Real phi = grid.ph_grid(j);

        for (int l = lmin; l <= lmax; ++l)
        {
          const Real ll = static_cast<Real>(l);
          for (int m = -l; m <= l; ++m)
          {
            const int abs_m = std::abs(m);
            const Real mm   = static_cast<Real>(m);
            const int lm    = lmindex(l, m);

            // Sign for negative m: (-1)^|m| when m < 0
            const int sign = (m < 0 && (abs_m % 2 != 0)) ? -1 : 1;

            // NPlm stores P(l, |m|) with 4pi normalisation and CS phase
            const Real Plm  = sign * P(l, abs_m);
            const Real dPlm = sign * dP(l, abs_m);

            const Real cosmph = std::cos(mm * phi);
            const Real sinmph = std::sin(mm * phi);

            // Y_l^m = P * e^{imph}
            Y(i, j, lm, ix_C::Re) = Plm * cosmph;
            Y(i, j, lm, ix_C::Im) = Plm * sinmph;

            // dY/dth = dP/dth * e^{imph}
            Yth(i, j, lm, ix_C::Re) = dPlm * cosmph;
            Yth(i, j, lm, ix_C::Im) = dPlm * sinmph;

            // dY/dph = im * Y
            Yph(i, j, lm, ix_C::Re) = -mm * Plm * sinmph;
            Yph(i, j, lm, ix_C::Im) = mm * Plm * cosmph;

            // W_lm = -2 cot(th) dY/dth + (2m^2/sin^2th - l(l+1)) Y
            const Real Wcoeff =
              2.0 * mm * mm * div_sinth * div_sinth - ll * (ll + 1.0);
            W(i, j, lm, ix_C::Re) =
              -2.0 * cot_theta * Yth(i, j, lm, ix_C::Re) +
              Wcoeff * Y(i, j, lm, ix_C::Re);
            W(i, j, lm, ix_C::Im) =
              -2.0 * cot_theta * Yth(i, j, lm, ix_C::Im) +
              Wcoeff * Y(i, j, lm, ix_C::Im);

            // X_lm = 2m (cot(th) Y_Im - dY_Im/dth, dY_Re/dth - cot(th) Y_Re)
            // That is: X_Re = 2m (cot(th) Y_Im - Yth_Im)
            //          X_Im = 2m (Yth_Re - cot(th) Y_Re)
            X(i, j, lm, ix_C::Re) =
              2.0 * mm *
              (cot_theta * Y(i, j, lm, ix_C::Im) - Yth(i, j, lm, ix_C::Im));
            X(i, j, lm, ix_C::Im) =
              2.0 * mm *
              (Yth(i, j, lm, ix_C::Re) - cot_theta * Y(i, j, lm, ix_C::Re));
          }  // m
        }  // l
      }  // j (phi)
    }  // i (theta)
  }

  // --------------------------------------------------------------------------
  //! Scalar projection: accumulate Int f(i,j) Y*_lm(i,j) dOmega into buf.
  //!
  //! buf must point to a block of size 2*lmpoints (zeroed by the caller).
  //! Result is stored as buf[2*lm + c] for c in {Re, Im}.
  //! The conjugate Y* flips the sign of the Im part.
  // --------------------------------------------------------------------------
  template <typename GridType, typename ScalarField>
  void ProjectScalar(Real* buf, const GridType& grid, ScalarField&& f) const
  {
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          if (!grid.IsOwned(i, j))
            continue;

          const Real vol  = grid.weights(i, j);
          const Real fval = f(i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              // Y* = (Y_Re, -Y_Im)
              t_buf[2 * lm + ix_C::Re] += vol * fval * Y(i, j, lm, ix_C::Re);
              t_buf[2 * lm + ix_C::Im] -= vol * fval * Y(i, j, lm, ix_C::Im);
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf[k] += t_buf[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Scalar projection from a rank-1 DTensorField component.
  //! Projects v(deriv, a, i, j) against Y*.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectScalar(
    Real* buf,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 1>& v,
    int deriv,
    int a) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;
    ProjectScalar(buf,
                  grid,
                  [&](int i, int j) -> Real
                  { return (deriv == D00) ? v(a, i, j) : v(deriv, a, i, j); });
  }

  // --------------------------------------------------------------------------
  //! Scalar projection from a rank-2 DTensorField component.
  //! Projects t(deriv, a, b, i, j) against Y*.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectScalar(
    Real* buf,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 2>& t,
    int deriv,
    int a,
    int b) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;
    ProjectScalar(
      buf,
      grid,
      [&](int i, int j) -> Real
      { return (deriv == D00) ? t(a, b, i, j) : t(deriv, a, b, i, j); });
  }

  // --------------------------------------------------------------------------
  //! Even vector projection:
  //!   Int (f_th d_thY* + f_ph (1/sin^2th) d_phY*) dOmega
  //!
  //! The 1/sin^2th factor is intrinsic to the angular metric g^{AB} and is
  //! applied internally using precomputed grid trig values.
  // --------------------------------------------------------------------------
  template <typename GridType, typename ThetaField, typename PhiField>
  void ProjectEvenVector(Real* buf,
                         const GridType& grid,
                         ThetaField&& f_th,
                         PhiField&& f_ph) const
  {
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real div_sinth2 =
            1.0 / (grid.sin_theta(i) * grid.sin_theta(i));

          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fth = f_th(i, j);
          const Real fph = f_ph(i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              // conj: Yth* = (Yth_Re, -Yth_Im), Yph* = (Yph_Re, -Yph_Im)
              const Real contrib_Re =
                fth * Yth(i, j, lm, ix_C::Re) +
                fph * div_sinth2 * Yph(i, j, lm, ix_C::Re);
              const Real contrib_Im =
                fth * Yth(i, j, lm, ix_C::Im) +
                fph * div_sinth2 * Yph(i, j, lm, ix_C::Im);

              t_buf[2 * lm + ix_C::Re] += vol * contrib_Re;
              t_buf[2 * lm + ix_C::Im] -= vol * contrib_Im;
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf[k] += t_buf[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Odd vector projection:
  //!   Int (1/sinth) (-f_th d_phY* + f_ph d_thY*) dOmega
  //!
  //! The 1/sinth factor comes from the Levi-Civita tensor on the sphere.
  // --------------------------------------------------------------------------
  template <typename GridType, typename ThetaField, typename PhiField>
  void ProjectOddVector(Real* buf,
                        const GridType& grid,
                        ThetaField&& f_th,
                        PhiField&& f_ph) const
  {
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real div_sinth = 1.0 / grid.sin_theta(i);

          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fth = f_th(i, j);
          const Real fph = f_ph(i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              // conj flips Im sign
              const Real contrib_Re =
                div_sinth * (-fth * Yph(i, j, lm, ix_C::Re) +
                             fph * Yth(i, j, lm, ix_C::Re));
              const Real contrib_Im =
                div_sinth * (-fth * Yph(i, j, lm, ix_C::Im) +
                             fph * Yth(i, j, lm, ix_C::Im));

              t_buf[2 * lm + ix_C::Re] += vol * contrib_Re;
              t_buf[2 * lm + ix_C::Im] -= vol * contrib_Im;
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf[k] += t_buf[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Even tensor projection:
  //!   Int (f_W W* + f_X X*) dOmega
  //!
  //! W and X already contain the angular-metric factors.
  // --------------------------------------------------------------------------
  template <typename GridType, typename WField, typename XField>
  void ProjectEvenTensor(Real* buf,
                         const GridType& grid,
                         WField&& f_W,
                         XField&& f_X) const
  {
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fw  = f_W(i, j);
          const Real fx  = f_X(i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              const Real contrib_Re =
                fw * W(i, j, lm, ix_C::Re) + fx * X(i, j, lm, ix_C::Re);
              const Real contrib_Im =
                fw * W(i, j, lm, ix_C::Im) + fx * X(i, j, lm, ix_C::Im);

              t_buf[2 * lm + ix_C::Re] += vol * contrib_Re;
              t_buf[2 * lm + ix_C::Im] -= vol * contrib_Im;
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf[k] += t_buf[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Odd tensor projection:
  //!   Int (-f_W X* + f_X W*) dOmega
  //!
  //! The sign/swap pattern comes from the dual (Levi-Civita) coupling.
  // --------------------------------------------------------------------------
  template <typename GridType, typename WField, typename XField>
  void ProjectOddTensor(Real* buf,
                        const GridType& grid,
                        WField&& f_W,
                        XField&& f_X) const
  {
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fw  = f_W(i, j);
          const Real fx  = f_X(i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              const Real contrib_Re =
                -fw * X(i, j, lm, ix_C::Re) + fx * W(i, j, lm, ix_C::Re);
              const Real contrib_Im =
                -fw * X(i, j, lm, ix_C::Im) + fx * W(i, j, lm, ix_C::Im);

              t_buf[2 * lm + ix_C::Re] += vol * contrib_Re;
              t_buf[2 * lm + ix_C::Im] -= vol * contrib_Im;
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf[k] += t_buf[k];
      }
    }  // omp parallel
  }

  // ==========================================================================
  // Domain-specific projection methods
  // ==========================================================================
  //
  // These methods take DTensorField references directly and bake in the
  // l-dependent prefactors (1/lam for vectors, 1/(lam(lam-2)) for tensors) so
  // that buffer values come out correct per-mode.  This eliminates post-MPI
  // post-multiply steps in the caller.
  //
  // "Pair" methods project both even and odd parity in a single grid pass.
  // Pass nullptr for buf_even or buf_odd to skip a parity.
  // ==========================================================================

  // --------------------------------------------------------------------------
  //! Paired even+odd vector projection from a rank-1 DTensorField.
  //!
  //! Extracts the angular components v(deriv, 1, i, j) and v(deriv, 2, i, j)
  //! as th and ph components, then projects:
  //!   Even: Int (v_th d_thY* + v_ph (1/sin^2th) d_phY*) dOmega / lam
  //!   Odd:  Int (1/sinth)(-v_th d_phY* + v_ph d_thY*) dOmega / lam
  //!
  //! Pass nullptr for buf_even or buf_odd to skip that parity.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectVectorPair(
    Real* buf_even,
    Real* buf_odd,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 1>& v,
    int deriv) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_even(buf_even ? bufsize : 0, 0.0);
      std::vector<Real> t_odd(buf_odd ? bufsize : 0, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real sinth      = grid.sin_theta(i);
          const Real div_sinth  = 1.0 / sinth;
          const Real div_sinth2 = div_sinth * div_sinth;

          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fth = (deriv == D00) ? v(1, i, j) : v(deriv, 1, i, j);
          const Real fph = (deriv == D00) ? v(2, i, j) : v(deriv, 2, i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            const Real div_lambda = 1.0 / static_cast<Real>(l * (l + 1));
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);

              if (buf_even)
              {
                const Real eRe = fth * Yth(i, j, lm, ix_C::Re) +
                                 fph * div_sinth2 * Yph(i, j, lm, ix_C::Re);
                const Real eIm = fth * Yth(i, j, lm, ix_C::Im) +
                                 fph * div_sinth2 * Yph(i, j, lm, ix_C::Im);
                t_even[2 * lm + ix_C::Re] += vol * div_lambda * eRe;
                t_even[2 * lm + ix_C::Im] -= vol * div_lambda * eIm;
              }

              if (buf_odd)
              {
                const Real oRe = div_sinth * (-fth * Yph(i, j, lm, ix_C::Re) +
                                              fph * Yth(i, j, lm, ix_C::Re));
                const Real oIm = div_sinth * (-fth * Yph(i, j, lm, ix_C::Im) +
                                              fph * Yth(i, j, lm, ix_C::Im));
                t_odd[2 * lm + ix_C::Re] += vol * div_lambda * oRe;
                t_odd[2 * lm + ix_C::Im] -= vol * div_lambda * oIm;
              }
            }
          }
        }
      }

#pragma omp critical
      {
        if (buf_even)
          for (int k = 0; k < bufsize; ++k)
            buf_even[k] += t_even[k];
        if (buf_odd)
          for (int k = 0; k < bufsize; ++k)
            buf_odd[k] += t_odd[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Paired even+odd vector projection from a rank-2 DTensorField with a
  //! fixed first index.
  //!
  //! Extracts t(deriv, a, 1, i, j) and t(deriv, a, 2, i, j) as th and ph
  //! components, then projects the same even/odd formulas as above with 1/lam.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectVectorPair(
    Real* buf_even,
    Real* buf_odd,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 2>& t,
    int deriv,
    int a) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_even(buf_even ? bufsize : 0, 0.0);
      std::vector<Real> t_odd(buf_odd ? bufsize : 0, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real sinth      = grid.sin_theta(i);
          const Real div_sinth  = 1.0 / sinth;
          const Real div_sinth2 = div_sinth * div_sinth;

          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);
          const Real fth =
            (deriv == D00) ? t(a, 1, i, j) : t(deriv, a, 1, i, j);
          const Real fph =
            (deriv == D00) ? t(a, 2, i, j) : t(deriv, a, 2, i, j);

          for (int l = lmin; l <= lmax; ++l)
          {
            const Real div_lambda = 1.0 / static_cast<Real>(l * (l + 1));
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);

              if (buf_even)
              {
                const Real eRe = fth * Yth(i, j, lm, ix_C::Re) +
                                 fph * div_sinth2 * Yph(i, j, lm, ix_C::Re);
                const Real eIm = fth * Yth(i, j, lm, ix_C::Im) +
                                 fph * div_sinth2 * Yph(i, j, lm, ix_C::Im);
                t_even[2 * lm + ix_C::Re] += vol * div_lambda * eRe;
                t_even[2 * lm + ix_C::Im] -= vol * div_lambda * eIm;
              }

              if (buf_odd)
              {
                const Real oRe = div_sinth * (-fth * Yph(i, j, lm, ix_C::Re) +
                                              fph * Yth(i, j, lm, ix_C::Re));
                const Real oIm = div_sinth * (-fth * Yph(i, j, lm, ix_C::Im) +
                                              fph * Yth(i, j, lm, ix_C::Im));
                t_odd[2 * lm + ix_C::Re] += vol * div_lambda * oRe;
                t_odd[2 * lm + ix_C::Im] -= vol * div_lambda * oIm;
              }
            }
          }
        }
      }

#pragma omp critical
      {
        if (buf_even)
          for (int k = 0; k < bufsize; ++k)
            buf_even[k] += t_even[k];
        if (buf_odd)
          for (int k = 0; k < bufsize; ++k)
            buf_odd[k] += t_odd[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Paired even+odd tensor projection from a rank-2 DTensorField.
  //!
  //! Projects the angular-block (A,B) components of gamma into even (G) and
  //! odd (H) multipole slots with 1/(lam(lam-2)) prefactor baked in per-mode.
  //!
  //! Even side: computes the r-power-corrected (product rule of 1/r^2)
  //!   f_W = (gamma_thth - gamma_phph/sin^2th)/r^2  and  f_X = 2gamma_thph/(r^2 sin^2th)
  //!   then projects Int (f_W W* + f_X X*) dOmega / (lam(lam-2)).
  //!
  //! Odd side: no r-power correction,
  //!   f_W = (gamma_thth - gamma_phph/sin^2th)/sinth  and  f_X = 2gamma_thph/sinth
  //!   then projects Int (-f_W X* + f_X W*) dOmega / (lam(lam-2)).
  //!
  //! The deriv parameter selects which output derivative to compute.
  //! For the even side, the product rule of d^n_r (1/r^2 f) mixes lower
  //! derivative slots of gamma; the method reads them automatically.
  //! For the odd side, only the deriv slot is used (no r-power correction).
  //!
  //! Pass nullptr for buf_even or buf_odd to skip that parity.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectTensorPair(
    Real* buf_even,
    Real* buf_odd,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 2>& gamma,
    int deriv,
    Real div_r) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;

    const Real div_r2 = div_r * div_r;
    const Real div_r3 = div_r2 * div_r;
    const Real div_r4 = div_r3 * div_r;

    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_even(buf_even ? bufsize : 0, 0.0);
      std::vector<Real> t_odd(buf_odd ? bufsize : 0, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          const Real sinth      = grid.sin_theta(i);
          const Real div_sinth  = 1.0 / sinth;
          const Real div_sinth2 = div_sinth * div_sinth;

          if (!grid.IsOwned(i, j))
            continue;

          const Real vol = grid.weights(i, j);

          // ----- Even side: r-power-corrected f_W and f_X
          // ---------------------
          Real even_fW = 0.0;
          Real even_fX = 0.0;

          if (buf_even)
          {
            auto trace_minus = [&](int d) -> Real
            {
              return (d == D00)
                     ? (gamma(1, 1, i, j) - gamma(2, 2, i, j) * div_sinth2)
                     : (gamma(d, 1, 1, i, j) -
                        gamma(d, 2, 2, i, j) * div_sinth2);
            };
            auto cross = [&](int d) -> Real
            {
              return (d == D00) ? (gamma(1, 2, i, j) * div_sinth2)
                                : (gamma(d, 1, 2, i, j) * div_sinth2);
            };

            switch (deriv)
            {
              case D00:
                even_fW = div_r2 * trace_minus(D00);
                even_fX = 2.0 * div_r2 * cross(D00);
                break;
              case D10:
                even_fW =
                  div_r2 * trace_minus(D10) - 2.0 * div_r3 * trace_minus(D00);
                even_fX =
                  2.0 * (div_r2 * cross(D10) - 2.0 * div_r3 * cross(D00));
                break;
              case D01:
                even_fW = div_r2 * trace_minus(D01);
                even_fX = 2.0 * div_r2 * cross(D01);
                break;
              case D20:
                even_fW = div_r2 * trace_minus(D20) -
                          4.0 * div_r3 * trace_minus(D10) +
                          6.0 * div_r4 * trace_minus(D00);
                even_fX =
                  2.0 * (div_r2 * cross(D20) - 4.0 * div_r3 * cross(D10) +
                         6.0 * div_r4 * cross(D00));
                break;
              case D11:
                even_fW =
                  div_r2 * trace_minus(D11) - 2.0 * div_r3 * trace_minus(D01);
                even_fX =
                  2.0 * (div_r2 * cross(D11) - 2.0 * div_r3 * cross(D01));
                break;
            }
          }

          // ----- Odd side: 1/sinth weighting, no r-power
          // -----------------------
          Real odd_fW = 0.0;
          Real odd_fX = 0.0;

          if (buf_odd)
          {
            const Real gtt =
              (deriv == D00) ? gamma(1, 1, i, j) : gamma(deriv, 1, 1, i, j);
            const Real gtp =
              (deriv == D00) ? gamma(1, 2, i, j) : gamma(deriv, 1, 2, i, j);
            const Real gpp =
              (deriv == D00) ? gamma(2, 2, i, j) : gamma(deriv, 2, 2, i, j);

            odd_fW = div_sinth * (gtt - gpp * div_sinth2);
            odd_fX = 2.0 * gtp * div_sinth;
          }

          // ----- Accumulate into buffers per (l,m) with prefactor
          // -------------
          for (int l = lmin; l <= lmax; ++l)
          {
            const Real lambda  = static_cast<Real>(l * (l + 1));
            const Real div_ll2 = 1.0 / (lambda * (lambda - 2.0));

            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);

              if (buf_even)
              {
                const Real cRe = even_fW * W(i, j, lm, ix_C::Re) +
                                 even_fX * X(i, j, lm, ix_C::Re);
                const Real cIm = even_fW * W(i, j, lm, ix_C::Im) +
                                 even_fX * X(i, j, lm, ix_C::Im);
                t_even[2 * lm + ix_C::Re] += vol * div_ll2 * cRe;
                t_even[2 * lm + ix_C::Im] -= vol * div_ll2 * cIm;
              }

              if (buf_odd)
              {
                const Real cRe = -odd_fW * X(i, j, lm, ix_C::Re) +
                                 odd_fX * W(i, j, lm, ix_C::Re);
                const Real cIm = -odd_fW * X(i, j, lm, ix_C::Im) +
                                 odd_fX * W(i, j, lm, ix_C::Im);
                t_odd[2 * lm + ix_C::Re] += vol * div_ll2 * cRe;
                t_odd[2 * lm + ix_C::Im] -= vol * div_ll2 * cIm;
              }
            }
          }
        }
      }

#pragma omp critical
      {
        if (buf_even)
          for (int k = 0; k < bufsize; ++k)
            buf_even[k] += t_even[k];
        if (buf_odd)
          for (int k = 0; k < bufsize; ++k)
            buf_odd[k] += t_odd[k];
      }
    }  // omp parallel
  }

  // --------------------------------------------------------------------------
  //! Trace scalar projection from a rank-2 DTensorField, with K correction.
  //!
  //! Projects 0.5 * [r-power-corrected] (gamma_thth + gamma_phph/sin^2th) against Y*,
  //! then adds the K correction: 0.5 * lam * buf_G[lm] per-mode.
  //!
  //! buf_G must already be filled by ProjectTensorPair for the same deriv.
  //! The K correction is valid before MPI reduce because the operation is
  //! linear (commutes with summation).
  //!
  //! The r-power product rule for d^n_r (1/r^2 f) is applied based on deriv.
  // --------------------------------------------------------------------------
  template <typename GridType, typename T, TensorSymm sym, int ndim>
  void ProjectTrace(
    Real* buf_K,
    const Real* buf_G,
    const GridType& grid,
    const gra::grids::theta_phi::DTensorField<T, sym, ndim, 2>& gamma,
    int deriv,
    Real div_r) const
  {
    using namespace gra::grids::theta_phi::ix_DRT;

    const Real div_r2 = div_r * div_r;
    const Real div_r3 = div_r2 * div_r;
    const Real div_r4 = div_r3 * div_r;

    // ----- Grid integral: 0.5 * [r-power-corrected] trace against Y* -------
    const int bufsize = 2 * lmpoints;

#pragma omp parallel
    {
      std::vector<Real> t_buf(bufsize, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; ++i)
      {
        for (int j = 0; j < nphi; ++j)
        {
          if (!grid.IsOwned(i, j))
            continue;

          const Real div_sinth2 =
            1.0 / (grid.sin_theta(i) * grid.sin_theta(i));
          const Real vol = grid.weights(i, j);

          // Helper: angular trace at a given derivative slot
          auto trace = [&](int d) -> Real
          {
            return (d == D00)
                   ? (gamma(1, 1, i, j) + gamma(2, 2, i, j) * div_sinth2)
                   : (gamma(d, 1, 1, i, j) +
                      gamma(d, 2, 2, i, j) * div_sinth2);
          };

          // Product rule of d^n_r (1/r^2 \cdot f), with overall factor 0.5
          Real fval = 0.0;
          switch (deriv)
          {
            case D00:
              fval = 0.5 * div_r2 * trace(D00);
              break;
            case D10:
              fval = 0.5 * (div_r2 * trace(D10) - 2.0 * div_r3 * trace(D00));
              break;
            case D01:
              fval = 0.5 * div_r2 * trace(D01);
              break;
            case D20:
              fval = 0.5 * (div_r2 * trace(D20) - 4.0 * div_r3 * trace(D10) +
                            6.0 * div_r4 * trace(D00));
              break;
            case D11:
              fval = 0.5 * (div_r2 * trace(D11) - 2.0 * div_r3 * trace(D01));
              break;
          }

          // Project against Y*
          for (int l = lmin; l <= lmax; ++l)
          {
            for (int m = -l; m <= l; ++m)
            {
              const int lm = lmindex(l, m);
              t_buf[2 * lm + ix_C::Re] += vol * fval * Y(i, j, lm, ix_C::Re);
              t_buf[2 * lm + ix_C::Im] -= vol * fval * Y(i, j, lm, ix_C::Im);
            }
          }
        }
      }

#pragma omp critical
      {
        for (int k = 0; k < bufsize; ++k)
          buf_K[k] += t_buf[k];
      }
    }  // omp parallel

    // ----- K correction: K += 0.5 * lam * G per-mode -------------------------
    // Executed once, after all thread-private buffers have been merged above.
    for (int l = lmin; l <= lmax; ++l)
    {
      const Real half_lambda = 0.5 * static_cast<Real>(l * (l + 1));
      for (int m = -l; m <= l; ++m)
      {
        const int lm = lmindex(l, m);
        buf_K[2 * lm + ix_C::Re] += half_lambda * buf_G[2 * lm + ix_C::Re];
        buf_K[2 * lm + ix_C::Im] += half_lambda * buf_G[2 * lm + ix_C::Im];
      }
    }
  }
};

// ============================================================================
// RealHarmonicTable
// ============================================================================
//! \brief Pre-tabulated real spherical harmonics and angular derivatives on
//!        a (theta, phi) grid.
//!
//! The real-harmonic convention splits each (l, m>0) mode into cosine (Yc)
//! and sine (Ys) parts, while m=0 modes live in Y0.  All derivatives up to
//! second order in (theta, phi) are stored.
//!
//! Access pattern (derivative enum as leading index):
//!   Y0(D00, i, j, l)    - value of m=0 harmonic
//!   Yc(D10, i, j, lm)   - d/dtheta of cosine harmonic
//!   Ys(D01, i, j, lm)   - d/dphi of sine harmonic
//!   etc.
struct RealHarmonicTable
{
  int lmax     = 0;
  int lmpoints = 0;

  //! Backing arrays - accessed via derivative-degree leading index.
  //!   Y0 : (3, ntheta, nphi, lmax+1)        - m=0 (D00, D10, D20 only)
  //!   Yc : (NDERIV, ntheta, nphi, lmpoints)  - m>0 cosine
  //!   Ys : (NDERIV, ntheta, nphi, lmpoints)  - m>0 sine
  AthenaArray<Real> Y0;
  AthenaArray<Real> Yc;
  AthenaArray<Real> Ys;

  //! (l,m) -> linear index
  int lmindex(int l, int m) const
  {
    return lmindex_real(l, m, lmax);
  }

  // --------------------------------------------------------------------------
  //! Allocate arrays and compute all harmonics + derivatives on the grid.
  //! th_grid(ntheta) and ph_grid(nphi) must already be filled.
  // --------------------------------------------------------------------------
  void Initialize(int lmax_in,
                  int ntheta,
                  int nphi,
                  const AthenaArray<Real>& th_grid,
                  const AthenaArray<Real>& ph_grid)
  {
    using namespace ix_D;
    lmax     = lmax_in;
    lmpoints = (lmax + 1) * (lmax + 1);

    // Legendre scratch (only needed during this call)
    AthenaArray<Real> P_all, P, dP, d2P;
    P_all.NewAthenaArray(3, lmax + 1, lmax + 1);
    P.InitWithShallowSlice(P_all, 3, 0, 1);
    dP.InitWithShallowSlice(P_all, 3, 1, 1);
    d2P.InitWithShallowSlice(P_all, 3, 2, 1);

    // Allocate harmonic tables
    Y0.NewAthenaArray(3, ntheta, nphi, lmax + 1);
    Yc.NewAthenaArray(NDERIV, ntheta, nphi, lmpoints);
    Ys.NewAthenaArray(NDERIV, ntheta, nphi, lmpoints);
    Y0.ZeroClear();
    Yc.ZeroClear();
    Ys.ZeroClear();

    // Fill tables
    for (int i = 0; i < ntheta; ++i)
    {
      const Real theta = th_grid(i);
      NPlm(theta, lmax, P, dP, d2P);

      for (int j = 0; j < nphi; ++j)
      {
        const Real phi = ph_grid(j);

        // m=0 harmonics
        for (int l = 0; l <= lmax; l++)
        {
          Y0(D00, i, j, l) = P(l, 0);
          Y0(D10, i, j, l) = dP(l, 0);
          Y0(D20, i, j, l) = d2P(l, 0);
        }

        // m>0 harmonics
        for (int l = 1; l <= lmax; l++)
        {
          for (int m = 1; m <= l; m++)
          {
            const int l1 = lmindex(l, m);

            const Real cosmph = std::cos(m * phi);
            const Real sinmph = std::sin(m * phi);

            // Values
            Yc(D00, i, j, l1) = SQRT2 * P(l, m) * cosmph;
            Ys(D00, i, j, l1) = SQRT2 * P(l, m) * sinmph;

            // First derivatives
            Yc(D10, i, j, l1) = SQRT2 * dP(l, m) * cosmph;
            Ys(D10, i, j, l1) = SQRT2 * dP(l, m) * sinmph;
            Yc(D01, i, j, l1) = -SQRT2 * P(l, m) * m * sinmph;
            Ys(D01, i, j, l1) = SQRT2 * P(l, m) * m * cosmph;

            // Second derivatives
            Yc(D20, i, j, l1) = SQRT2 * d2P(l, m) * cosmph;
            Ys(D20, i, j, l1) = SQRT2 * d2P(l, m) * sinmph;
            Yc(D11, i, j, l1) = -SQRT2 * dP(l, m) * m * sinmph;
            Ys(D11, i, j, l1) = SQRT2 * dP(l, m) * m * cosmph;
            Yc(D02, i, j, l1) = -SQRT2 * P(l, m) * m * m * cosmph;
            Ys(D02, i, j, l1) = -SQRT2 * P(l, m) * m * m * sinmph;
          }
        }
      }  // phi loop
    }  // theta loop
  }

  // --------------------------------------------------------------------------
  //! Evaluate r(theta,phi) = sum_lm a_lm Y_lm and angular derivatives.
  //! Also computes rr_min = min over all grid points of rr(i,j).
  // --------------------------------------------------------------------------
  void Synthesize(const AthenaArray<Real>& a0,
                  const AthenaArray<Real>& ac,
                  const AthenaArray<Real>& as,
                  int ntheta,
                  int nphi,
                  AthenaArray<Real>& rr,
                  AthenaArray<Real>& rr_dth,
                  AthenaArray<Real>& rr_dph,
                  Real& rr_min) const
  {
    using namespace ix_D;
    rr.ZeroClear();
    rr_dth.ZeroClear();
    rr_dph.ZeroClear();

    rr_min = std::numeric_limits<Real>::infinity();
    for (int i = 0; i < ntheta; i++)
    {
      for (int j = 0; j < nphi; j++)
      {
        for (int l = 0; l <= lmax; l++)
        {
          rr(i, j) += a0(l) * Y0(D00, i, j, l);
          rr_dth(i, j) += a0(l) * Y0(D10, i, j, l);
        }

        for (int l = 1; l <= lmax; l++)
        {
          for (int m = 1; m <= l; m++)
          {
            const int l1 = lmindex(l, m);
            rr(i, j) +=
              ac(l1) * Yc(D00, i, j, l1) + as(l1) * Ys(D00, i, j, l1);
            rr_dth(i, j) +=
              ac(l1) * Yc(D10, i, j, l1) + as(l1) * Ys(D10, i, j, l1);
            rr_dph(i, j) +=
              ac(l1) * Yc(D01, i, j, l1) + as(l1) * Ys(D01, i, j, l1);
          }
        }
        rr_min = std::min(rr_min, rr(i, j));
      }  // phi loop
    }  // theta loop
  }

  // --------------------------------------------------------------------------
  //! Compute weighted inner products <field, Y_lm> (local sums only).
  //! Caller is responsible for MPI reduction of spec0, specc, specs.
  //! Arrays spec0[lmax+1], specc[lmpoints], specs[lmpoints] must be
  //! zero-initialized by the caller.
  //!
  //! The is_owned functor should return true for points owned by this rank.
  // --------------------------------------------------------------------------
  template <typename IsOwnedFunc>
  void Project(const AthenaArray<Real>& weights,
               const AthenaArray<Real>& field,
               int ntheta,
               int nphi,
               Real* spec0,
               Real* specc,
               Real* specs,
               IsOwnedFunc is_owned) const
  {
    using namespace ix_D;
    const int nlm0 = lmax + 1;
    const int nlm  = lmpoints;

#pragma omp parallel
    {
      // Thread-private accumulators
      std::vector<Real> t_spec0(nlm0, 0.0);
      std::vector<Real> t_specc(nlm, 0.0);
      std::vector<Real> t_specs(nlm, 0.0);

#pragma omp for collapse(2) schedule(dynamic)
      for (int i = 0; i < ntheta; i++)
      {
        for (int j = 0; j < nphi; j++)
        {
          if (!is_owned(i, j))
            continue;
          const Real drho = weights(i, j) * field(i, j);

          for (int l = 0; l <= lmax; l++)
            t_spec0[l] += drho * Y0(D00, i, j, l);

          for (int l = 1; l <= lmax; l++)
          {
            for (int m = 1; m <= l; m++)
            {
              const int l1 = lmindex(l, m);
              t_specc[l1] += drho * Yc(D00, i, j, l1);
              t_specs[l1] += drho * Ys(D00, i, j, l1);
            }
          }
        }  // phi loop
      }  // theta loop

      // Merge thread-private buffers into caller's arrays
#pragma omp critical
      {
        for (int l = 0; l < nlm0; l++)
          spec0[l] += t_spec0[l];
        for (int l = 0; l < nlm; l++)
        {
          specc[l] += t_specc[l];
          specs[l] += t_specs[l];
        }
      }
    }  // omp parallel
  }
};

}  // namespace sph_harm
}  // namespace gra

#endif  // UTILS_SPHERICAL_HARMONICS_HPP_
