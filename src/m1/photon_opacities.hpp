//#ifndef PHOTON_OPACITIES_HPP
//#define PHOTON_OPACITIES_HPP
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_opacities.hpp
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

//! \class PhotonOpacities
//! \brief Computes photon opacities
class PhotonOpacities {
public:
  PhotonOpacities(ParameterInput * pin);
  ~PhotonOpacities();
public:
  Real PhotonBlackBody(Real const temperature);
  int PhotonOpacity(Real const rho, Real const temperature, Real const Y_e,
		    Real * abs_1, Real * sca_1);
private:
  Real kappa_sca; 
  Real kappa_abs; 
  Real rad_const;
};
//#endif
