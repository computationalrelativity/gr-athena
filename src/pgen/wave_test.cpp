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
    
    
    //1D
    //Sinusoidal initial profile
    //Real sin_x = sin(M_PI*x);
    //Real cos_x = cos(M_PI*x);
    
    //pwave->u(0,k,j,i) = prof(sin_x);
    //pwave->u(1,k,j,i) = direction*M_PI*c*cos_x*prof_diff(sin_x);
    
    
    //Gaussian initial profile
    //Real sigma_2 = 0.1*0.1;
    //Real gauss = exp(-x*x/sigma_2);
    
    //pwave->u(0,k,j,i) = prof(gauss);
    //pwave->u(1,k,j,i) = 2.*direction/sigma_2*c*x*prof(gauss);
    
    
    //pwave->exact(0,k,j,i) = pwave->u(0,k,j,i);
    //pwave->error(0,k,j,i) = 0.0;
    
    
    //2D
    Real cos_x = cos(3.*M_PI*x);
    Real cos_y = cos(4.*M_PI*y);
    // T = 0.4
    
    //3D
    Real cos_x = cos(1.*M_PI*x);
    Real cos_y = cos(2.*M_PI*y);
    Real cos_z = cos(2.*M_PI*z);
    // T = 2/3
    
    pwave->u(0,k,j,i) = prof(cos_x)*prof(cos_y);
    pwave->u(1,k,j,i) = 0.;//M_PI*c*prof(cos_x)*prof(cos_y);
  }
  return;
}

//2D
/*
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
    cout<<SQRT2<<'\t';
    cout<<x/SQRT2<<'\t';
    cout<<cos(M_PI*x/SQRT2)<<'\t';
    cout<<pwave->u(0,k,j,i)<<'\t';
    cout<<(cos(M_PI*t*c) + sin(M_PI*t*c))*cos(M_PI*x/SQRT2)*cos(M_PI*y/SQRT2)<<'\t';
    
    pwave->exact(0,k,j,i) = (cos(M_PI*t*c) + sin(M_PI*t*c))*cos(M_PI*x/SQRT2)*cos(M_PI*y/SQRT2);
    cout<<pwave->exact(0,k,j,i)<<'\n';
    pwave->error(0,k,j,i) = pwave->u(0,k,j,i) - pwave->exact(0,k,j,i);
  }
  return;
}
*/
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
    //cout<<SQRT2<<'\t';
    //cout<<x/SQRT2<<'\t';
    //cout<<cos(M_PI*x/SQRT2)<<'\t';
    //cout<<pwave->u(0,k,j,i)<<'\t';
    //cout<<(cos(M_PI*t*c) + sin(M_PI*t*c))*cos(M_PI*x/SQRT2)*cos(M_PI*y/SQRT2)<<'\t';
    
    pwave->exact(0,k,j,i) = (cos(M_PI*t*c) + sin(M_PI*t*c))*cos(M_PI*x/SQRT2)*cos(M_PI*y/SQRT2);
    //cout<<pwave->exact(0,k,j,i)<<'\n';
    pwave->error(0,k,j,i) = pwave->u(0,k,j,i) - pwave->exact(0,k,j,i);
  }
  return;
}
/* 
//1D
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
    Real sigma_2 = 0.1*0.1;
    Real xp, xm;
    switch(direction) {
      case -1:
        //Sine
        xp = sin(M_PI*(x + c*t));
        
        //Gaussian
        //xp = exp(-SQR(x + c*t)/sigma_2) + exp(-SQR(fmod(x + c*t -1.0, 2.0) -1.0)/sigma_2);
        
        pwave->exact(0,k,j,i) = prof(xp);
        break;
      case 0:
        //Sine
        xp = sin(M_PI*(x + c*t));
        xm = sin(M_PI*(x - c*t));
        
        //Gaussian  
        //xp = exp(-(x + c*t)*(x + c*t)/sigma_2)+ exp(-SQR(fmod(x + c*t -1.0, 2.0) -1.0)/sigma_2);
        //xm = exp(-(x - c*t)*(x - c*t)/sigma_2)+ exp(-SQR(fmod(x - c*t + 1.0, 2.0) + 1.0)/sigma_2);        
        
        pwave->exact(0,k,j,i) = 0.5*(prof(xm) + prof(xp));
        break;
      case 1:
        //Sine
        xm = sin(M_PI*(x - c*t));
        
        //Gaussian
        //xm = exp(-SQR(x - c*t)/sigma_2) + exp(-SQR(fmod(x - c*t + 1.0, 2.0) + 1.0)/sigma_2);
        
        pwave->exact(0,k,j,i) = prof(xm);
        break;
      default:
        assert(false); // you shouldn't be here
        abort();
    }
    pwave->error(0,k,j,i) = pwave->u(0,k,j,i) - pwave->exact(0,k,j,i);
  }
  return;
}
*/
