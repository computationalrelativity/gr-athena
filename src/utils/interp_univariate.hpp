#ifndef INTERP_UNIVARIATE_HPP_
#define INTERP_UNIVARIATE_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file interp_univariate.hpp
//  \brief Collection of univariate interpolators

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"

// uniform grid; stencil fully biased towards right (target left)
// use for extrapolation
template<int stencil_size_>
class InterpolateLagrangeUniformBiasR {
  public:
    enum {interpolation_order = stencil_size_ - 1};
    enum {npoints = stencil_size_};
    static Real const coeff[npoints];
};
// fully bias left (target right)
template<int stencil_size_>
class InterpolateLagrangeUniformBiasL {
  public:
    enum {interpolation_order = stencil_size_ - 1};
    enum {npoints = stencil_size_};
    static Real const coeff[npoints];
};

// uniform grid assumed
template<int half_stencil_size_>
class InterpolateLagrangeUniform {
  public:
    enum {interpolation_order = 2 * half_stencil_size_ - 1};
    enum {npoints = half_stencil_size_};
    static Real const coeff[npoints];
};

// target prolongation of a cell to two children;
// leftward child coefficients stored
// rightward is the same but with indices reversed
template<int half_stencil_size_>
class InterpolateLagrangeUniformChildren {
  public:
    enum {interpolation_order = 2 * half_stencil_size_};
    enum {npoints = 2 * half_stencil_size_ + 1};
    static Real const coeff[npoints];
};

#endif