//! \file cold_piecewise_polytrope.cpp
//  \brief Implementation of ColdPiecewisePolytrope

#include "cold_piecewise_polytrope.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "unit_system.hpp"

using namespace Primitive;
using namespace std;

ColdPiecewisePolytrope::ColdPiecewisePolytrope() : PiecewisePolytrope()
{
  eos_units = &Nuclear;
  T         = 0.0;
  mb        = 1.0;
}

ColdPiecewisePolytrope::~ColdPiecewisePolytrope()
{
}

Real ColdPiecewisePolytrope::Pressure(Real n)
{
  assert(IsInitialized());
  int p = FindPiece(n);
  return GetColdPressure(n, p) + n * T;
}

Real ColdPiecewisePolytrope::Energy(Real n)
{
  assert(IsInitialized());
  int p = FindPiece(n);
  return GetColdEnergy(n, p) + n * T / (gamma_thermal - 1.0);
}

Real ColdPiecewisePolytrope::dPdn(Real n)
{
  assert(IsInitialized());
  int p = FindPiece(n);
  return pressure_pieces[p] * gamma_pieces[p] *
           std::pow(n, gamma_pieces[p] - 1) /
           std::pow(density_pieces[p], gamma_pieces[p]) +
         T;
}

Real ColdPiecewisePolytrope::SpecificInternalEnergy(Real n)
{
  assert(IsInitialized());
  int p         = FindPiece(n);
  Real eps_cold = GetColdEnergy(n, p) / (n * mb) - 1.0;
  return eps_cold + T / (mb * (gamma_thermal - 1.0));
}

Real ColdPiecewisePolytrope::Y(Real n, int iy)
{
  throw std::logic_error("ColdPiecewisePolytrope::Y not implemented");
}

Real ColdPiecewisePolytrope::Enthalpy(Real n)
{
  assert(IsInitialized());
  int p = FindPiece(n);
  return (GetColdEnergy(n, p) + GetColdPressure(n, p)) / n +
         gamma_thermal / (gamma_thermal - 1.0) * T;
}

int ColdPiecewisePolytrope::FindPieceFromP(Real P) const
{
  // Throw an error if the polytrope hasn't been initialized yet.
  if (!initialized)
  {
    throw std::runtime_error(
      "ColdPiecewisePolytrope::FindPieceFromP - EOS not initialized.");
  }

  for (int i = 0; i < n_pieces - 1; ++i)
  {
    if (P < pressure_pieces[i + 1])
    {
      return i;
    }
  }

  return n_pieces - 1;
}

int ColdPiecewisePolytrope::FindPieceFromE(Real e) const
{
  // Throw an error if the polytrope hasn't been initialized yet.
  if (!initialized)
  {
    throw std::runtime_error(
      "ColdPiecewisePolytrope::FindPieceFromE - EOS not initialized.");
  }

  for (int i = 0; i < n_pieces - 1; ++i)
  {
    if (e < mb * density_pieces[i + 1] * (1.0 + eps_pieces[i + 1]))
    {
      return i;
    }
  }

  return n_pieces - 1;
}

Real ColdPiecewisePolytrope::DensityFromPressure(Real P)
{
  assert(IsInitialized());
  if (T > 1e-10)
  {
    throw std::runtime_error(
      "ColdPiecewisePolytrope::DensityFromPressure only implemented for T=0");
  }

  int p = FindPieceFromP(P);
  return density_pieces[p] *
         std::pow(P / pressure_pieces[p], 1.0 / gamma_pieces[p]);
}

Real ColdPiecewisePolytrope::DensityFromEnergy(Real e)
{
  assert(IsInitialized());
  if (T > 1e-10)
  {
    throw std::runtime_error(
      "ColdPiecewisePolytrope::DensityFromEnergy only implemented for T=0");
  }
  // Unfortunately, e(rho) cannot be inverted simply. Therefore, we use a
  // Newton-Raphson solver to get this instead.
  int p   = FindPieceFromE(e);
  Real lb = density_pieces[p];
  Real ub = density_pieces[p + 1];
  auto f  = [&](Real n) -> Real { return GetColdEnergy(n, p) - e; };
  auto df = [&](Real n) -> Real
  {
    return mb * (1.0 + eps_pieces[p]) + gamma_pieces[p] *
                                          GetColdPressure(n, p) /
                                          (n * (gamma_pieces[p] - 1.0));
  };
  Real flb       = f(lb);
  Real fub       = f(ub);
  Real x         = (fub * lb - flb * ub) / (fub - flb);
  Real fx        = f(x);
  const Real tol = 1e-15;
  while (abs(fx) > e * tol)
  {
    Real xnew = x - fx / df(x);
    // If the guess is no good, throw it away and use bisection instead.
    if (xnew > ub || xnew < lb)
    {
      xnew = 0.5 * (ub + lb);
    }
    fx = f(xnew);
    if (fx > 0)
    {
      ub = xnew;
    }
    else
    {
      lb = xnew;
    }
    x = xnew;
  }

  return x;
}
