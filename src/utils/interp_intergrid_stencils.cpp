//! \file interp_intergrid_stencils.cpp
//  \brief provide derivative stencils for intergrid mapping

// C headers

// C++ headers

// Athena++ headers
#include "interp_intergrid.hpp"

// ----------------------------------------------------------------------------
// New impl. here

namespace InterpIntergrid {

// smaller values first, may improve accuracy
template<>
Real const InterpolateVC2DerCC<1, 1>::coeff[1] = {
  1.
};

template<>
Real const InterpolateVC2DerCC<1, 2>::coeff[2] = {
  -1./24., 9./8.
};

template<>
Real const InterpolateVC2DerCC<1, 3>::coeff[3] = {
  3./640., -25./384., 75./64.
};

template<>
Real const InterpolateVC2DerCC<1, 4>::coeff[4] = {
  -(5./7168.),
  49./5120.,
  -(245./3072.),
  1225./1024.
};

template<>
Real const InterpolateVC2DerCC<1, 5>::coeff[5] = {
  35./294912.,
  -(405./229376.),
  567./40960.,
  -(735./8192.),
  19845./16384.
};

template<>
Real const InterpolateVC2DerCC<1, 6>::coeff[6] = {
  -(63./2883584.),
  847./2359296.,
  -(5445./1835008.),
  22869./1310720.,
  -(12705./131072.),
  160083./131072.
};

template<>
Real const InterpolateCC2DerCC<1, 1>::coeff[2] = {
  1./2., 0.
};

template<>
Real const InterpolateCC2DerCC<1, 2>::coeff[3] = {
  -1./12., 2./3., 0.
};

template<>
Real const InterpolateCC2DerCC<1, 3>::coeff[4] = {
  1./60., -(3./20.), 3./4., 0.
};

template<>
Real const InterpolateCC2DerCC<1, 4>::coeff[5] = {
  -(1./280.),
  4./105.,
  -(1./5.),
  4./5.,
  0.
};

template<>
Real const InterpolateCC2DerCC<1, 5>::coeff[6] = {
  1./1260.,
  -(5./504.),
  5./84.,
  -(5./21.),
  5./6.,
  0.
};

template<>
Real const InterpolateCC2DerCC<1, 6>::coeff[7] = {
  -(1./5544.),
  1./385.,
  -(1./56.),
  5./63.,
  -(15./56.),
  6./7.,
  0.
};

}  // namespace InterpIntergrid

//
// :D
//