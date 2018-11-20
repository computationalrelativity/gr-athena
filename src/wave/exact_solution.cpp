//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file exact_solution.cpp
//  \brief Computes the exact solution of the scalar wave equation

// C/C++ headers
#include <iostream>
#include <sstream>
#include <cmath>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "wave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//! \fn void Wave::ComputeExactSol
//  \brief Computes the exact solution of the scalar wave equation

void Wave::ComputeExactSol()
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  Real t = (pmb->pmy_mesh->time) + (pmb->pmy_mesh->dt);

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();
#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif

   for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
#pragma omp simd
        for(int i = is; i <= ie; ++i) {

            Real x = pco->x1v(i);
            Real y = pco->x2v(j);
            Real z = pco->x3v(k);

            //Gaussian ----------------------------------------------
//            Real plus   = WaveProfile(x+t);
//            Real minus  = WaveProfile(x-t);

//            Real plusp  = WaveProfile(fmod(x+t-c,2.0)-c);
//            Real minusp = WaveProfile(fmod(x-t+c,2.0)+c);

//            exact(0,k,j,i) = (plus+minus)/2.0 + (plusp+minusp)/2.0;
            //-------------------------------------------------------

            //Sinusoids----------------------------------------------
            exact(0,k,j,i) = cos(2.*M_PI*t)*WaveProfile(x,y,z);

            error(0,k,j,i) = abs( exact(0,k,j,i) - u(0,k,j,i) );
        }
      }
    }
  } // end of parallel region

  return;
}

Real Wave::WaveProfile(Real x, Real y, Real z)
{
    Real const a = 1.0;
    Real const sigma = 0.2;

    //Real profile = a*exp(-SQR(x)/SQR(sigma));
    //Real profile = a*sin(2.*M_PI*x);
    Real profile = a*sin(2.*M_PI*x)*cos(2.*M_PI*y); //*cos(2.*M_PI*z);
    //Real profile = pow(cos(0.5*M_PI*x), 6);

    return profile;
}
