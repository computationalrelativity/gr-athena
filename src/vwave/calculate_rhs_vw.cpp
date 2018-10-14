//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_rhs.cpp
//  \brief Calculate vectorial wave equation RHS

// C/C++ headers
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "vwave.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../athena_tensor.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// FD stencils
static Real const __diff_1_stencil_2[] = {
    1./12., -2./3., 0., 2./3., -1./12.,
};
static Real const __diff_2_stencil_2[] = {
    -1./12., 4./3., -5./2., 4./3., -1./12.
};

static Real const __diff_1_stencil_4[] = {
    -1./60., 3./20., -3./4., 0., 3./4., -3./20., 1/60
};
static Real const __diff_2_stencil_4[] = {
    1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.
};

static Real const __diff_1_stencil_8[] = {
    1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.
};
static Real const __diff_2_stencil_8[] = {
    -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.
};

//! \fn void Vwave::CalculateRHS
//  \brief Calculate RHS for the vectorial wave equation using finite-differencing

void Vwave::VwaveRHS(AthenaArray<Real> & u, int order)
{
  std::stringstream msg;

  MeshBlock *pmb = pmy_block;
  Coordinates * pco = pmb->pcoord;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  g.array().InitWithShallowSlice(u, 4, gab_IDX, 6);
  K.array().InitWithShallowSlice(u, 4, Kab_IDX, 6);

  rhs_g.array().InitWithShallowSlice(rhs, 4, gab_IDX, 6);
  rhs_K.array().InitWithShallowSlice(rhs, 4, Kab_IDX, 6);

  //TODO: define routines for drvts
  Real const * _diff_1_stencil;
  Real const * _diff_2_stencil;
  switch(order) {
    case 2:
      _diff_2_stencil = &__diff_2_stencil_2[0];
      _diff_1_stencil = &__diff_1_stencil_2[0];
      break;
    case 4:
      _diff_2_stencil = &__diff_2_stencil_4[0];
      _diff_1_stencil = &__diff_1_stencil_4[0];
      break;
    //case 6:
    //  _diff_2_stencil = &__diff_2_stencil_6[0];
    //  _diff_1_stencil = &__diff_1_stencil_6[0];
    //  break;
    case 8:
      _diff_2_stencil = &__diff_2_stencil_8[0];
      _diff_1_stencil = &__diff_1_stencil_8[0];
      break;
    default:
      msg << "FD order not supported: " << order << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  int const stencil_offset = order/2;
  int const stencil_size = order + 1;

  int tid = 0;
  int nthreads = pmb->pmy_mesh->GetNumMeshThreads();

#pragma omp parallel default(shared) private(tid) num_threads(nthreads)
  {
#ifdef OPENMP_PARALLEL
    tid = omp_get_thread_num();
#endif
    
    //----------------------------------------------------------------------------------------
    // g rhs 
    
    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      rhs_g(a,b,k,j,i) = -2*K(a,b,k,j,i);
	    }
	  }
	}
      }
    }
    
    //----------------------------------------------------------------------------------------
    // K rhs 
    
    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
	
	//----------------------------------------------------------------------------------------
	// Flat metric and its Inverse 

#pragma omp simd
        for(int i = is; i <= ie; ++i) {
          for(int a = 0; a < 3; ++a)
          for(int b = a; b < 3; ++b) {
            eta(a, b, i) = (a == b) ? (1.0) : (0.0);
          }
        }

#pragma omp simd
	for(int i = is; i <= ie; ++i) {
	  Real det = SpatialDet( eta(0,0,i), eta(0,1,i), eta(0,2,i), 
				 eta(1,1,i), eta(1,2,i), eta(2,2,i) );
	  SpatialInv( det,
		      eta(0,0,i), eta(0,1,i), eta(0,2,i), 
		      eta(1,1,i), eta(1,2,i), eta(2,2,i), 
		      &ieta(0,0,i), &ieta(0,1,i), &ieta(0,2,i), 
		      &ieta(1,1,i), &ieta(1,2,i), &ieta(2,2,i) );
	  
	}
	
	//----------------------------------------------------------------------------------------
	// Metric drvts 
	
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) { 
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    dd_loc += _diff_2_stencil[n]*g(a,b, k, j, i + n - stencil_offset);
		  }
		  dd_loc /= SQR(pco->dx1v(i));
		  ddg(0,0,a,b,i) = dd_loc;
		}
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    dd_loc += _diff_2_stencil[n]*g(a,b, k, j + n - stencil_offset, i);
		  }
		  dd_loc /= SQR(pco->dx2v(j));
		  ddg(1,1,a,b,i) = dd_loc;
		}
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    dd_loc += _diff_2_stencil[n]*g(a,b, k + n - stencil_offset, j, i);
		  }
		  dd_loc /= SQR(pco->dx3v(k));
		  ddg(2,2,a,b,i) = dd_loc;
		}
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    for(int m = 0; m < stencil_size; ++m) {
		      dd_loc += _diff_1_stencil[n]*_diff_1_stencil[m]*g(a,b, k, j + m - stencil_offset, i + n - stencil_offset);
		    }
		  }
		  dd_loc /= (pco->dx1v(i)*pco->dx2v(j));
		  ddg(0,1,a,b,i) = dd_loc;
		}
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    for(int m = 0; m < stencil_size; ++m) {
		      dd_loc += _diff_1_stencil[n]*_diff_1_stencil[m]*g(a,b, k + m - stencil_offset, j, i + n - stencil_offset);
		    }
		  }
		  dd_loc /= (pco->dx1v(i)*pco->dx3v(k));
		  ddg(0,2,a,b,i) = dd_loc;
		}
#pragma omp simd
		for(int i = is; i <= ie; ++i) {
		  Real dd_loc = 0;
		  for(int n = 0; n < stencil_size; ++n) {
		    for(int m = 0; m < stencil_size; ++m) {
		      dd_loc += _diff_1_stencil[n]*_diff_1_stencil[m]*g(a,b, k + m - stencil_offset, j + n - stencil_offset, i);
		    }
		  }
		  dd_loc /= (pco->dx2v(j)*pco->dx3v(k));
		  ddg(1,2,a,b,i) = dd_loc;
		}
	  }
	}
	
	//----------------------------------------------------------------------------------------
	// Ricci tensor
	
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      R(a,b,i) = 0.;
	      for(int c = 0; c < NDIM; ++c) {
		for(int d = 0; d < NDIM; ++d) {
		  R(a,b,i) -= 0.5*ieta(c,d,i)*ddg(c,d,a,b,i);
		}
	      }
	    }
	  }
	}
	
	//----------------------------------------------------------------------------------------
	// K rhs 
	
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      rhs_K(a,b,k,j,i) = R(a,b,i);
	    }
	  }
	}
	
      } // j - loop
    } // k - loop
    
  } // end of parallel region

  return;
}
