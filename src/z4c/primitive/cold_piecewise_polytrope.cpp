//! \file cold_piecewise_polytrope.cpp
//  \brief Implementation of ColdPiecewisePolytrope

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "unit_system.hpp"
#include "cold_piecewise_polytrope.hpp"

using namespace Primitive;
using namespace std;

ColdPiecewisePolytrope::ColdPiecewisePolytrope():
  PiecewisePolytrope() {
  eos_units = &Nuclear;
  T = 0.0;
  mb = 1.0;
}

ColdPiecewisePolytrope::~ColdPiecewisePolytrope() {}


Real ColdPiecewisePolytrope::Pressure(Real n) {
  assert (IsInitialized());
  int p = FindPiece(n);
  return GetColdPressure(n, p) + n*T;
}

Real ColdPiecewisePolytrope::Energy(Real n) {
  assert (IsInitialized());
  int p = FindPiece(n);
  return GetColdEnergy(n, p) + n*T/(gamma_thermal - 1.0);
}

Real ColdPiecewisePolytrope::dPdn(Real n) {
  assert (IsInitialized());
  int p = FindPiece(n);
  return pressure_pieces[p] * gamma_pieces[p]
    * std::pow(n, gamma_pieces[p]-1)
    / std::pow(density_pieces[p], gamma_pieces[p])
    + T;
}

Real ColdPiecewisePolytrope::SpecificInternalEnergy(Real n) {
  assert (IsInitialized());
  int p = FindPiece(n);
  Real eps_cold = GetColdEnergy(n, p)/(n*mb) - 1.0;
  return eps_cold + T/(mb*(gamma_thermal - 1.0));
}

Real ColdPiecewisePolytrope::Y(Real n, int iy) {
  throw std::logic_error("ColdPiecewisePolytrope::Y not implemented");
}

Real ColdPiecewisePolytrope::Enthalpy(Real n) {
  assert (IsInitialized());
  int p = FindPiece(n);
  return (GetColdEnergy(n, p) + GetColdPressure(n, p))/n + gamma_thermal/(gamma_thermal - 1.0)*T;
}
