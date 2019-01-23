//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_pulse.cpp
//  \brief Expanding pulse in 3D 

#include <cassert> // assert
#include <cmath> // exp, sqrt
#include <iostream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../wave/wave.hpp"

using namespace std;

namespace {

template<typename T>
Real sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

Real bump(Real r) {
  if(abs(r) < 1.) {
    return exp(-1./(1. - SQR(r)));
  }
  else {
    return 0.;
  }
}

Real bump_diff(Real r) {
  if(abs(r) < 1.) {
    return -2.*r*bump(r)/SQR(-1. + SQR(r));
  }
  else {
    return 0.;
  }
}

typedef Real (*unary_function)(Real);

unary_function prof = NULL;
unary_function prof_diff = NULL;

Real tp0, xp0, yp0, zp0;

} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  prof = bump;
  prof_diff = bump_diff;

  tp0 = pin->GetOrAddReal("problem", "t0", 2.0);
  xp0 = pin->GetOrAddReal("problem", "x0", 0.0);
  yp0 = pin->GetOrAddReal("problem", "y0", 0.0);
  zp0 = pin->GetOrAddReal("problem", "z0", 0.0);

  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j)
  for(int i = is; i <= ie; ++i) {
    Real x = pcoord->x1v(i);
    Real y = pcoord->x2v(j);
    Real z = pcoord->x3v(k);
    Real r = sqrt(SQR(x-xp0) + SQR(y-yp0) + SQR(z-zp0));
    Real c = pwave->c;

    pwave->u(0,k,j,i) = prof(r - c*tp0)/r;
    pwave->u(1,k,j,i) = -c*prof_diff(r - c*tp0)/r;

    pwave->exact(0,k,j,i) = pwave->u(0,k,j,i);
    pwave->error(0,k,j,i) = 0.0;
  }
  return;
}

void MeshBlock::UserWorkInLoop()
{
  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j)
  for(int i = is; i <= ie; ++i) {
    Real x = pcoord->x1v(i);
    Real y = pcoord->x2v(j);
    Real z = pcoord->x3v(k);
    Real r = sqrt(SQR(x-xp0) + SQR(y-yp0) + SQR(z-zp0));
    Real t = pmy_mesh->time + pmy_mesh->dt;
    Real c = pwave->c;

    pwave->exact(0,k,j,i) = prof(r - c*(t + tp0))/r;
    pwave->error(0,k,j,i) = pwave->u(0,k,j,i) - pwave->exact(0,k,j,i);
  }
  return;
}
