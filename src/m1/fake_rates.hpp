//#ifndef FAKE_RATES_HPP
//#define FAKE_RATES_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fake_rates.hpp
//  \brief definitions for the FakeRates class

// C++ standard headers
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>

// Athena++ headers
#include "../athena.hpp"
//#include "../athena_arrays.hpp"

// Forward declaration
class ParameterInput;

//! \class FakeRates
//! \brief Computes fake neutrino rates for testing
class FakeRates {
public:
  FakeRates(ParameterInput * pin, int _nspecies, int _ngroups);
  ~FakeRates();
public:
  int Emission(Real const rho, Real const temp, Real const Y_e,
	       Real *_eta);
  int Absorption_abs(Real const rho, Real const temp, Real const Y_e,
		     Real * _kappa_abs);
  int Absorption_sca(Real const rho, Real const temp, Real const Y_e,
		     Real * _kappa_sca);
  int AverageAtomicMass(Real const rho, Real const temp, Real const Y_e,
			Real * Abar);
  Real AverageBaryonMass();
private:
  int nspecies, ngroups;
  Real * eta; // Specific emissivity coefficient
  Real * kappa_sca; // Specific opacity 
  Real * kappa_abs; // Specific absorption opacity
  Real avg_atomic_mass;  
  Real avg_baryon_mass; 
};
