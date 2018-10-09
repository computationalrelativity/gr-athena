//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave.cpp
//  \brief Initial conditions for the wave equation

#include <cmath> // sin

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../wave/wave.hpp"

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j)
  for(int i = is; i <= ie; ++i) {
    // Dummy hydrodynamics state
    phydro->u(IDN,k,j,i) = 1.0;
    phydro->u(IM1,k,j,i) = 0.0;
    phydro->u(IM2,k,j,i) = 0.0;
    phydro->u(IM3,k,j,i) = 0.0;
    if (NON_BAROTROPIC_EOS) {
      phydro->u(IEN,k,j,i) = 1.0;
    }

    Real x = pcoord->x1v(i);
    Real y = pcoord->x2v(j);
    Real z = pcoord->x3v(k);

    pwave->u(0,k,j,i) = exp(-x*x/0.05);
    //pwave->u(0,k,j,i) = sin(2*M_PI*x)*cos(2*M_PI*y)*cos(2*M_PI*z);
    pwave->u(1,k,j,i) = 0.0;
    pwave->exact(0,k,j,i) = 1.0; //exp(-x*x/0.05);
  }
  return;
}
