//! \file cold_idealgas.cpp
//  \brief Implementation of ColdPolytrope

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <limits>

#include "unit_system.hpp"
#include "cold_idealgas.hpp"

using namespace Primitive;
using namespace std;

Polytrope::Polytrope() {
  gamma = numeric_limits<Real>::quiet_NaN();
  gammam1 = numeric_limits<Real>::quiet_NaN();
  K = numeric_limits<Real>::quiet_NaN();
  mb = 1.0;
  min_n = 0.0;
  max_n = std::numeric_limits<Real>::max();
  T = 0.0;
  n_species = 0;
  eos_units = &Nuclear;
}

Polytrope::~Polytrope() {}

Real Polytrope::Pressure(Real n) {
  assert (IsInitialized());
  return K * pow(n, gamma);
}

Real Polytrope::Energy(Real n) {
  assert (IsInitialized());
  return n*mb + K*pow(n, gamma)/gammam1;
}

Real Polytrope::dPdn(Real n) {
  assert (IsInitialized());
  return K * gamma * pow(n, gammam1);
}

Real Polytrope::SpecificInternalEnergy(Real n) {
  return K*pow(n, gammam1)/gammam1/mb;
}

Real Polytrope::Y(Real n, int iy) {
  throw std::logic_error("Polytrope::Y not implemented");
}

Real Polytrope::Enthalpy(Real n) {
  return (Energy(n) + Pressure(n))/n/mb;
}

Real Polytrope::DensityFromPressure(Real P) {
  return pow(P/K, 1.0/gamma);
}

void Polytrope::SetNSpecies(int n) {
  if (n > MAX_SPECIES || n < 0) {
    throw std::out_of_range("IdealGas::SetNSpecies - n cannot exceed MAX_SPECIES.");
  }
  n_species = n;
}
