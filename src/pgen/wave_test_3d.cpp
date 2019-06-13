//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_test.cpp
//  \brief Initial conditions for the wave equation

#include <cassert> // assert
#include <cmath> // abs, exp, sin, fmod
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

Real linear(Real x) {
  return x;
}

Real linear_diff(Real x) {
  return 1;
}

Real bump(Real x) {
  if(abs(x) < 1.) {
    return exp(-1./(1. - SQR(x)));
  }
  else {
    return 0.;
  }
}

Real bump_diff(Real x) {
  if(abs(x) < 1.) {
    return -2.*x*bump(x)/SQR(-1. + SQR(x));
  }
  else {
    return 0.;
  }
}

typedef Real (*unary_function)(Real);

unary_function prof = NULL;
unary_function prof_diff = NULL;

int direction = 0;

} // namespace

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Sets the initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  direction = pin->GetOrAddInteger("problem", "direction", 1);
  if(abs(direction) > 1) {
    cerr << "Invalid direction: " << direction << endl;
    cerr << "Valid values are: -1, 0, and 1" << endl;
    cerr << flush;
    abort();
  }

  string profile = pin->GetOrAddString("problem", "profile", "linear");
  if(profile == "bump") {
    prof = bump;
    prof_diff = bump_diff;
  }
  else {
    prof = linear;
    prof_diff = linear_diff;
  }
  cout<<'\n'<<direction;
  for(int k = ks; k <= ke; ++k)
  for(int j = js; j <= je; ++j)
  for(int i = is; i <= ie; ++i) {
    Real x = pcoord->x1v(i);
    Real y = pcoord->x2v(j);
    Real z = pcoord->x3v(k);
    Real c = pwave->c;

    //3D
    // T = 2/3
    //pwave->u(0,k,j,i) = prof(cos_x)*prof(cos_y)*prof(cos_z);
    //pwave->u(1,k,j,i) = 0.;
    //pwave->u(0,k,j,i) = prof(sin_x)*prof(sin_y)*prof(sin_z);
    //pwave->u(1,k,j,i) = 3.*M_PI*prof(sin_x)*prof(sin_y)*prof(sin_z);    
    pwave->u(0,k,j,i) = sin(M_PI*(1.*x+2.*y+2.*z));
    pwave->u(1,k,j,i) = 3.*M_PI*cos(M_PI*(1.*x+2.*y+2.*z));    

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
    Real t = pmy_mesh->time + pmy_mesh->dt;
    Real c = pwave->c;
    //pwave->exact(0,k,j,i) = sin(M_PI*t*3.)*sin(M_PI*x*1.)*sin(M_PI*y*2.)*sin(M_PI*z*2.);
    pwave->exact(0,k,j,i) = sin(M_PI*(1.*x+2.*y+2.*z+3.*t));
    pwave->error(0,k,j,i) = pwave->u(0,k,j,i) - pwave->exact(0,k,j,i);
  }
  return;
}
