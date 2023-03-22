#ifndef FINITE_DIFFERENCING_HPP_
#define FINITE_DIFFERENCING_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.hpp
//  \brief High-performance finite-differencing kernel

#include "../athena.hpp"

// Centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDCenteredStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = 2*nghost_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Choose the right order for the dissipation operators
template<int nghost_>
class FDDissChoice {
public:
  enum {degree = 2*(nghost_ + 1)};
  enum {nghost = nghost_ + 1};
};

// Choose the left or right biased stencil
template<int degree_, int nghost_>
class FDBiasedChoice {
public:
  enum {degree = degree_};
  enum {nghost = nghost_+1};
  enum {lopsize = nghost_};
};
template<>
class FDBiasedChoice<1, 1> {
public:
  enum {degree = 1};
  enum {nghost = 2};
  enum {lopsize = 0};
};
template<>
class FDBiasedChoice<1, 2> {
public:
  enum {degree = 1};
  enum {nghost = 3};
  enum {lopsize = 1};
};
template<>
class FDBiasedChoice<1, 3> {
public:
  enum {degree = 1};
  enum {nghost = 4};
  enum {lopsize = 2};
};
template<>
class FDBiasedChoice<1, 4> {
public:
  enum {degree = 1};
  enum {nghost = 5};
  enum {lopsize = 3};
};

// Left-biased finite differencing stencils
// * degree  : Degree of the derivative, eg, 1 for 1st derivative
// * nghost  : Number of ghost points used for the derivative
// * lopsize : Number of points to the right of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDLeftBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

// Right-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
// * lopsize : Number of points to the left of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDRightBiasedStencil {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = lopsize_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
};

#ifdef DBG_SYMMETRIZE_FD
// Rewrite with denominator LCM factoring to mitigate some error ==============

// Odd degree centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1, 3, ...
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDStencilCenteredDegreeOdd {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the
  // stencil
  enum {offset = nghost_};
  // width of stencil
  enum {width = 2 * nghost_ + 1};
  // coefficient data width
  enum {coeff_width = nghost_};
  // Finite differencing coefficients (these need to be divided by coeff_lcm)
  static Real const coeff[coeff_width];
  static Real const coeff_lcm;
};

// Even degree centered finite differencing stencils
// * degree : Degree of the derivative, eg, 2, 4, ...
// * nghost : Number of ghost points used for the derivative
template<int degree_, int nghost_>
class FDStencilCenteredDegreeEven {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the
  // stencil
  enum {offset = nghost_};
  // width of stencil
  enum {width = 2 * nghost_ + 1};
  // coefficient data width
  enum {coeff_width = nghost_ + 1};
  // Finite differencing coefficients (these need to be divided by coeff_lcm)
  static Real const coeff[coeff_width];
  static Real const coeff_lcm;
};

// Lop-sided finite differencing stencils
// Left:  [o o o x o o o] -> [o o o o x o o - -]
// Right: [o o o x o o o] -> [- - o o x o o o o]


// Left-biased finite differencing stencils
// * degree  : Degree of the derivative, eg, 1 for 1st derivative
// * nghost  : Number of ghost points used for the derivative
// * lopsize : Number of points to the right of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDStencilBiasedLeft {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = nghost_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
  static Real const coeff_lcm;
};

// Right-biased finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * nghost : Number of ghost points used for the derivative
// * lopsize : Number of points to the left of the derivative point
template<int degree_, int nghost_, int lopsize_>
class FDStencilBiasedRight {
public:
  // Degree of the derivative to be approximated
  enum {degree = degree_};
  // Number of ghost points required for the differencing
  enum {nghost = nghost_};
  // Position at which the derivative is computed wrt the beginning of the stencil
  enum {offset = lopsize_};
  // Width of the stencil
  enum {width = nghost_ + lopsize_ + 1};
  // Finite differencing coefficients
  static Real const coeff[width];
  static Real const coeff_lcm;
};

#endif // DBG_SYMMETRIZE_FD

#endif
