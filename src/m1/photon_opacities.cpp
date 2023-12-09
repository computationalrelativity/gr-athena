//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_opacities.cpp
//  \brief this class implements photon opacities 

// C++ standard headers
#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>

// Athena++ headers
#include "../athena.hpp"
//#include "../athena_arrays.hpp"
#include "photon_opacities.hpp"
#include "../parameter_input.hpp"

class ParameterInput;

PhotonOpacities::PhotonOpacities(ParameterInput *pin) {
  rad_const = pin->GetOrAddReal("PhotonOpacities","Radiation constant Msun^-2/MeV^4",
				2.471313401078565e-13);
  kappa_abs = pin->GetOrAddReal("PhotonOpacities","Absorption opacity",1.0);
  kappa_sca = pin->GetOrAddReal("PhotonOpacities","Scattering opacity",1.0);
}

PhotonOpacities::~PhotonOpacities() { }

Real PhotonOpacities::PhotonBlackBody(Real const temperature) {
  return rad_const * POW4(temperature);
}

int PhotonOpacities::PhotonOpacity(Real const rho, Real const temperature, Real const Y_e,
				   Real * abs_1, Real * sca_1) {
  *abs_1 = kappa_abs * rho;
  *sca_1 = kappa_sca * rho;
  return 0;
}
