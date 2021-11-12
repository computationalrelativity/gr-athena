#include <cmath>
#include <stdexcept>
#include <limits>
#include "idealgas.hpp"

//! \file idealgas.cpp
//  \brief Implementation of IdealGas.

using namespace Primitive;

/// Constructor
IdealGas::IdealGas() {
  gamma = 5.0/3.0;
  gammam1 = gamma - 1.0;
  mb = 1.0;

  min_n = 0.0;
  max_n = std::numeric_limits<Real>::max();
  min_T = 0.0;
  max_T = std::numeric_limits<Real>::max();
  n_species = 0;
}

Real IdealGas::TemperatureFromE(Real n, Real e, Real *Y) {
  return gammam1*(e - mb*n)/n;
}

Real IdealGas::TemperatureFromP(Real n, Real p, Real *Y) {
  return p/n;
}

Real IdealGas::Energy(Real n, Real T, Real *Y) {
  return n*(mb + T/gammam1);
}

Real IdealGas::Pressure(Real n, Real T, Real *Y) {
  return n*T;
}

Real IdealGas::Entropy(Real n, Real T, Real *Y) {
  throw std::logic_error("IdealGas::Entropy not currently implemented.");
}

Real IdealGas::Enthalpy(Real n, Real T, Real *Y) {
  return mb + gamma/gammam1*T;
}

Real IdealGas::MinimumEnthalpy() {
  return mb;
}

Real IdealGas::SoundSpeed(Real n, Real T, Real *Y) {
  return std::sqrt(gamma*gammam1*T/(gammam1*mb + gamma*T));
}

Real IdealGas::SpecificEnergy(Real n, Real T, Real *Y) {
  return T/(mb*gammam1);
}

void IdealGas::SetNSpecies(int n) {
  if (n > MAX_SPECIES || n < 0) {
    throw std::out_of_range("IdealGas::SetNSpecies - n cannot exceed MAX_SPECIES.");
  }
  n_species = n;
}
