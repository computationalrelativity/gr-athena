//! \file piecewise_polytrope.cpp
//  \brief Implementation of PiecewisePolytrope EOSPolicy

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>

#include "piecewise_polytrope.hpp"
#include "unit_system.hpp"

using namespace Primitive;

/// Constructor
PiecewisePolytrope::PiecewisePolytrope() {
  n_pieces = 0;
  initialized = false;
  n_species = 0;
  gamma_thermal = 5.0/3.0;
  for (int i = 0; i < MAX_SPECIES; i++) {
    min_Y[i] = 0.0;
    max_Y[i] = 1.0;
  }
  eos_units = &Nuclear;
}

/// Destructor
PiecewisePolytrope::~PiecewisePolytrope() {
  if (initialized) {
    delete[] density_pieces;
    delete[] gamma_pieces;
    delete[] pressure_pieces;
    delete[] eps_pieces;
  }
}

void PiecewisePolytrope::AllocateMemory() {
  if (initialized) {
    delete[] density_pieces;
    delete[] gamma_pieces;
    delete[] pressure_pieces;
    delete[] eps_pieces;
  }
  density_pieces = new Real[n_pieces];
  gamma_pieces = new Real[n_pieces];
  pressure_pieces = new Real[n_pieces];
  eps_pieces = new Real[n_pieces];
}

int PiecewisePolytrope::FindPiece(Real n) const {
  // Throw an error if the polytrope hasn't been initialized yet.
  if (!initialized) {
    throw std::runtime_error("PiecewisePolytrope::FindPiece - EOS not initialized.");
  }

  for (int i = 0; i < n_pieces-1; ++i) {
    if (n < density_pieces[i+1]) {
      return i;
    }
  }

  return n_pieces - 1;
}

Real PiecewisePolytrope::GetColdEnergy(Real n, int p) {
  return mb*n*(1.0 + eps_pieces[p]) + GetColdPressure(n, p)/(gamma_pieces[p] - 1.0);
}

Real PiecewisePolytrope::GetColdPressure(Real n, int p) {
  return pressure_pieces[p]*std::pow((n/density_pieces[p]),gamma_pieces[p]);
}

Real PiecewisePolytrope::TemperatureFromE(Real n, Real e, Real *Y) {
  int p = FindPiece(n);
  Real e_cold = GetColdEnergy(n, p);
  return (e - e_cold)*(gamma_thermal - 1.0)/n;
}

Real PiecewisePolytrope::TemperatureFromP(Real n, Real p, Real *Y) {
  int i = FindPiece(n);
  Real p_cold = GetColdPressure(n, i);
  return (p - p_cold)/n;
}

Real PiecewisePolytrope::TemperatureFromEntropy(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::TemperatureFromEntropy not currently implemented.");
}

Real PiecewisePolytrope::Energy(Real n, Real T, Real *Y) {
  int p = FindPiece(n);

  return GetColdEnergy(n, p) + n*T/(gamma_thermal - 1.0);
}

Real PiecewisePolytrope::Pressure(Real n, Real T, Real *Y) {
  int p = FindPiece(n);

  return GetColdPressure(n, p) + n*T;
}

Real PiecewisePolytrope::Enthalpy(Real n, Real T, Real *Y) {
  int p = FindPiece(n);
  return (GetColdEnergy(n, p) + GetColdPressure(n, p))/n + gamma_thermal/(gamma_thermal - 1.0)*T;
}

Real PiecewisePolytrope::MinimumEnthalpy() {
  return mb;
}

Real PiecewisePolytrope::SoundSpeed(Real n, Real T, Real *Y) {
  int p = FindPiece(n);

  Real rho = n*mb;

  Real h_cold = (GetColdEnergy(n, p) + GetColdPressure(n, p))/rho;
  Real h_th   = gamma_thermal/(gamma_thermal - 1.0)*T/mb;

  Real P_cold = GetColdPressure(n, p);

  Real csq_cold_w = gamma_pieces[p]*P_cold/rho;
  Real csq_th_w = (gamma_thermal - 1.0)*h_th;

  return std::sqrt((csq_cold_w + csq_th_w)/(h_th + h_cold));
}

Real PiecewisePolytrope::FrYn(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::FrYn not currently implemented.");
}

Real PiecewisePolytrope::FrYp(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::FrYp not currently implemented.");
}

Real PiecewisePolytrope::FrYh(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::FrYh not currently implemented.");
}

Real PiecewisePolytrope::AN(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::AN not currently implemented.");
}

Real PiecewisePolytrope::ZN(Real n, Real s, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::ZN not currently implemented.");
}

Real PiecewisePolytrope::SpecificInternalEnergy(Real n, Real T, Real *Y) {
  int p = FindPiece(n);
  Real eps_cold = GetColdEnergy(n, p)/(n*mb) - 1.0;
  return eps_cold + T/(mb*(gamma_thermal - 1.0));
}

Real PiecewisePolytrope::BaryonChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::BaryonChemicalPotential not currently implemented.");
}

Real PiecewisePolytrope::ChargeChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::ChargeChemicalPotential not currently implemented.");
}

Real PiecewisePolytrope::ElectronLeptonChemicalPotential(Real n, Real T, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::ElectronLeptonChemicalPotential not currently implemented.");
}

Real PiecewisePolytrope::MinimumPressure(Real n, Real *Y) {
  int p = FindPiece(n);
  return GetColdPressure(n, p);
}

Real PiecewisePolytrope::MaximumPressure(Real n, Real *Y) {
  return std::numeric_limits<Real>::max();
}

Real PiecewisePolytrope::MinimumEnergy(Real n, Real *Y) {
  int p = FindPiece(n);

  return GetColdEnergy(n, p);
}

Real PiecewisePolytrope::MaximumEnergy(Real n, Real *Y) {
  return std::numeric_limits<Real>::max();
}

bool PiecewisePolytrope::ReadParametersFromFile(std::string fname) {
  return false;
}

bool PiecewisePolytrope::InitializeFromData(Real *densities,
      Real *gammas, Real P0, Real m, int n) {
  // Make sure that we actually *have* polytropes
  if (n <= 1) {
    printf("PiecewisePolytrope: Invalid number of polytropes requested."); // NOLINT
    return false;
  }
  // Before we even try to construct anything, we need to make sure that
  // the densities are ordered properly.
  for (int i = 1; i < n; i++) {
    if(densities[i] <= densities[i-1]) {
      // The densities must be ordered from smallest to largest and strictly
      // increasing.
      printf("PiecewisePolytrope: Densities must be strictly increasing."); // NOLINT
      return false;
    }
  }

  // Initialize (most of) the member variables.
  n_pieces = n;
  mb = m;
  min_n = 0.0;
  max_n = std::numeric_limits<Real>::max();
  AllocateMemory();

  // Now we can construct the different pieces.
  //
  // Note that we store densities 1 twice, because on the first segment we need to
  // write the pressure in terms of rho1 and not rho0 (which would give a
  // division by zero)
  density_pieces[0] = densities[1]/mb;
  gamma_pieces[0] = gammas[0];
  pressure_pieces[0] = P0;

  for (int i = 1; i < n; i++) {
    density_pieces[i] = densities[i]/mb;
    gamma_pieces[i] = gammas[i];
    pressure_pieces[i] = pressure_pieces[i-1] *
        pow(density_pieces[i]/density_pieces[i-1], gamma_pieces[i-1]);
    // Because we've rewritten the EOS in terms of temperature, we don't need
    // kappa in its current form. However, we can use it to define the a
    // constants that show up in our equations.
    eps_pieces[i] = eps_pieces[i-1] + pressure_pieces[i-1] /
                  (density_pieces[i-1] * mb) * (1.0/(gammas[i-1] - 1.0) - 1.0/(gammas[i] - 1.0));
  }

  // Because we're adding in a finite-temperature component via the ideal gas,
  // the only restriction on the temperature is that it needs to be nonnegative.
  min_T = 0.0;
  max_T = std::numeric_limits<Real>::max();

  // DEBUG ONLY:
  // for (int i = 0; i < n; i++) {
  //   std::cout << "Polytrope: i = " << i << "\n";
  //   std::cout << "  n = " << density_pieces[i] << "\n";
  //   std::cout << "  gamma = " << gamma_pieces[i] << "\n";
  //   std::cout << "  eps = " << eps_pieces[i] << "\n";
  //   std::cout << "  pressure = " << pressure_pieces[i] << "\n";
  // }
  // std::cout << "  P0 = " << P0 << "\n";
  // std::cout << "  mb = " << mb << "\n";
  // std::cout << "  gamma_thermal = " << gamma_thermal << "\n";

  initialized = true;
  return true;
}

void PiecewisePolytrope::SetNSpecies(int n) {
  if (n > MAX_SPECIES || n < 0) {
    throw std::out_of_range("PiecewisePolytrope::SetNSpecies - n cannot exceed MAX_SPECIES.");
  }
  n_species = n;
}
