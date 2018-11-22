//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.cpp
//  \brief High-performance finite differencing kernel

#include "finite_differencing.hpp"

// TODO: generic FDStencil's

template<>
Real const FDCenteredStencil<1, 1>::coeff[] = {
  1./2., 0., -1./2.,
};

template<>
Real const FDCenteredStencil<2, 1>::coeff[] = {
  1., -2., 1.,
};

template<>
Real const FDCenteredStencil<1, 2>::coeff[] = {
  1./12., -2./3., 0., 2./3., -1./12.,
};

template<>
Real const FDCenteredStencil<2, 2>::coeff[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.
};

template<>
Real const FDCenteredStencil<1, 3>::coeff[] = {
  -1./60., 3./20., -3./4., 0., 3./4., -3./20., 1/60
};

template<>
Real const FDCenteredStencil<2, 3>::coeff[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.
};

template<>
Real const FDCenteredStencil<1, 4>::coeff[] = {
  1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.
};

template<>
Real const FDCenteredStencil<2, 4>::coeff[] = {
  -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.
};
