//#ifndef FAKE_OPACITIES_HPP
//#define FAKE_OPACITIES_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fake_opacities.hpp
//  \brief definitions for the FakeOpacities class

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

//! \class FakeOpacities
//! \brief Computes fake neutrino opacities for testing
class FakeOpacities {
public:
  FakeOpacities(ParameterInput * pin, int _nspecies, int _ngroups);
  ~FakeOpacities();
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
//#endif
