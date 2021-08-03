//! \file interp_intergrid_stencils.cpp
//  \brief provide derivative stencils for intergrid mapping

// C headers

// C++ headers

// Athena++ headers
#include "interp_intergrid.hpp"

template<>
Real const InterpolateVC2DerCC<1, 1>::coeff[1] = {
  1.
};

template<>
Real const InterpolateVC2DerCC<1, 2>::coeff[2] = {
  9./8., -1./24.
};

template<>
Real const InterpolateVC2DerCC<1, 3>::coeff[3] = {
  75./64., -25./384., 3./640.
};
