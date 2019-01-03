#ifndef FINITE_DIFFERENCING_HPP_
#define FINITE_DIFFERENCING_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.hpp
//  \brief High-performance finite-differencing kernel

#include "athena.hpp"
#include "athena_arrays.hpp"

// Centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDCenteredStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = 2*nghost_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Left-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDLeftBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = 2*nghost_};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Right-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDRightBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_ - 1};
  // Width of the stencil
  enum {width = 2*nghost_};
  // Finite differencing coefficients
  static Real const coeff[width];
};

#endif
