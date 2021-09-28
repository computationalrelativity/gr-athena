//! \file piecewise_polytrope.cpp
//  \brief Implementation of PiecewisePolytrope EOSPolicy

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>

#include "piecewise_polytrope.hpp"

using namespace Primitive;

/// Constructor
PiecewisePolytrope::PiecewisePolytrope() {
  n_pieces = 0;
  initialized = false;
  n_species = 0;
  gamma_thermal = 5.0/3.0;
}

/// Destructor
PiecewisePolytrope::~PiecewisePolytrope() {
  if (initialized) {
    delete[] density_pieces;
    delete[] a_pieces;
    delete[] gamma_pieces;
    delete[] pressure_pieces;
  }
}

void PiecewisePolytrope::AllocateMemory() {
  if (initialized) {
    delete[] density_pieces;
    delete[] a_pieces;
    delete[] gamma_pieces;
    delete[] pressure_pieces;
  }
  // We store one extra polytrope just in case a
  // minimum isn't specified.
  density_pieces = new Real[n_pieces+1];
  a_pieces = new Real[n_pieces+1];
  gamma_pieces = new Real[n_pieces+1];
  pressure_pieces = new Real[n_pieces+1];
}

int PiecewisePolytrope::FindPiece(Real n) const {
  // Throw an error if the polytrope hasn't been initialized yet.
  if (!initialized) {
    throw std::runtime_error("PiecewisePolytrope::FindPiece - EOS not initialized.");
  }
  // In case the density is below the minimum, we
  // implement a default case that is stored just
  // past the current polytrope.
  int polytrope = n_pieces-1;
  if (n < density_pieces[n_pieces]) {
    return n_pieces;
  }
  for (int i = 0; i < n_pieces; i++) {
    if (n <= density_pieces[i]) {
      polytrope = i;
      break;
    }
  }

  return polytrope;
}

Real PiecewisePolytrope::GetColdEnergy(Real n, int p) {
  return mb*n*(1.0 + a_pieces[p]) + GetColdPressure(n, p)/(gamma_pieces[p] - 1.0);
}

Real PiecewisePolytrope::GetColdPressure(Real n, int p) {
  return pressure_pieces[p]*std::pow((n/density_pieces[p]),gamma_pieces[p]);
}

Real PiecewisePolytrope::TemperatureFromE(Real n, Real e, Real *Y) {
  int p = FindPiece(n);
  Real e_cold = GetColdEnergy(n, p);
  return std::fmax((e - e_cold)*(gamma_thermal - 1.0)/n, 0.0);
}

Real PiecewisePolytrope::TemperatureFromP(Real n, Real p, Real *Y) {
  int i = FindPiece(n);
  Real p_cold = GetColdPressure(n, i);
  return std::fmax((p - p_cold)/n, 0.0);
}


Real PiecewisePolytrope::Energy(Real n, Real T, Real *Y) {
  int p = FindPiece(n);

  return GetColdEnergy(n, p) + n*T/(gamma_thermal - 1.0);
}

Real PiecewisePolytrope::Pressure(Real n, Real T, Real *Y) {
  int p = FindPiece(n);

  return GetColdPressure(n, p) + n*T;
}

Real PiecewisePolytrope::Entropy(Real n, Real T, Real *Y) {
  throw std::logic_error("PiecewisePolytrope::Entropy not currently implemented.");
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

Real PiecewisePolytrope::SpecificEnergy(Real n, Real T, Real *Y) {
  int p = FindPiece(n);
  Real eps_cold = GetColdEnergy(n, p)/(n*mb) - 1.0;
  return eps_cold + T/(mb*(gamma_thermal - 1.0));
}

bool PiecewisePolytrope::ReadParametersFromFile(std::string fname) {
  return false;
}

bool PiecewisePolytrope::InitializeFromData(Real *densities, 
      Real *gammas, Real rho_min, Real P0, Real m, int n) {
  // Make sure that we actually *have* polytropes.
  if (n <= 0) {
    return false;
  }
  // Before we even try to construct anything, we need to make sure that
  // the densities are ordered properly.
  for (int i = 1; i < n; i++) {
    if(densities[i] <= densities[i-1]) {
      // The densities must be ordered from smallest to largest and strictly
      // increasing.
      return false;
    }
  }
  if (rho_min >= densities[0]) {
    return false;
  }

  // Initialize (most of) the member variables.
  n_pieces = n;
  mb = m;
  min_n = 0.0;
  max_n = densities[n-1]/mb;
  AllocateMemory();

  // Now we can construct the different pieces.
  density_pieces[0] = densities[0]/m;
  gamma_pieces[0] = gammas[0];
  pressure_pieces[0] = P0;
  if (n > 1){
    a_pieces[0] = P0/densities[0]*(1.0/(gammas[0] - 1.0) - 1.0/(gammas[1] - 1.0));
  }
  else {
    a_pieces[0] = 0.0;
  }
  for (int i = 1; i < n; i++) {
    density_pieces[i] = densities[i]/m;
    gamma_pieces[i] = gammas[i];
    pressure_pieces[i] = pressure_pieces[i-1]*std::pow(densities[i]/densities[i-1],gammas[i]);
    // Because we've rewritten the EOS in terms of temperature, we don't need
    // kappa in its current form. However, we can use it to define the a
    // constants that show up in our equations.
    a_pieces[i] = a_pieces[i-1] + pressure_pieces[i-1]/
                    densities[i-1]*(1.0/(gammas[i-1] - 1.0) - 1.0/(gammas[i] - 1.0));
    // Let's double-check that the density is physical.
    if (gamma_pieces[i] > 2.0) {
      Real rho_max = std::pow((gammas[i] - 1.0)*(1.0 + a_pieces[i])/(gammas[i]*(gammas[i]-2.0)*
                        densities[i]/pressure_pieces[i]),1.0/(gammas[i]-1.0))*densities[i];
      if (densities[i] > rho_max) {
        std::cout << "The i = " << i 
                  << " piece of the polytrope permits superluminal densities: \n";
        std::cout << "  rho[i]     = " << densities[i] << "\n";
        std::cout << "  rho_max[i] = " << rho_max << "\n";
        return false;
      }
    }
  }

  // Set up the default case. Because of thermodynamic constraints,
  // we must force a to be zero. Gamma is fixed by continuity.
  a_pieces[n] = 0.0;
  Real factor = (gammas[0] - 1.0)*rho_min*a_pieces[0];
  Real P_min = P0*std::pow(rho_min/densities[0],gammas[0]);
  gamma_pieces[n] = (gammas[0]*P_min + factor)/(P_min + factor);
  density_pieces[n] = rho_min/mb;
  pressure_pieces[n] = P_min;

  // Because of the finite-temperature component, the energy density basically
  // just has to be positive.
  min_e = 0.0;
  max_e = std::numeric_limits<Real>::max();

  // DEBUG ONLY:
  /*for (int i = 0; i <= n; i++) {
    std::cout << "Polytrope: i = " << i << "\n";
    std::cout << "  n = " << density_pieces[i] << "\n";
    std::cout << "  gamma = " << gamma_pieces[i] << "\n";
    std::cout << "  a = " << a_pieces[i] << "\n";
    std::cout << "  pressure = " << pressure_pieces[i] << "\n";
  }
  std::cout << "  P0 = " << P0 << "\n";*/

  initialized = true;
  return true;
}

void PiecewisePolytrope::SetNSpecies(int n) {
  if (n > MAX_SPECIES || n < 0) {
    throw std::out_of_range("PiecewisePolytrope::SetNSpecies - n cannot exceed MAX_SPECIES.");
  }
  n_species = n;
}
