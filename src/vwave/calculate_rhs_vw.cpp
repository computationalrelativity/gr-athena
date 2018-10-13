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

static Real const diff_2_ord_2[] = {
  1., -2., 1.
};
static Real const diff_2_ord_4[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.
};
static Real const diff_2_ord_6[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.
};
static Real const diff_2_ord_8[] = {
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

  AthenaTensor<Real, SYM2> g; // Metric tensor
  AthenaTensor<Real, SYM2> K; // Curvature
  g.array().InitWithShallowSlice(u, 4, gab_IDX, 6);
  K.array().InitWithShallowSlice(u, 4, Kab_IDX, 6);

  AthenaTensor<Real, SYM2> rhs_g; 
  AthenaTensor<Real, SYM2> rhs_K; 
  rhs_g.array().InitWithShallowSlice(rhs, 4, gab_IDX, 6);
  rhs_K.array().InitWithShallowSlice(rhs, 4, Kab_IDX, 6);

  // aux 1d vars
  AthenaTensor<Real, SYM2, 2> ig;  // inverse Metric tensor //TODO: check def
  AthenaTensor<Real, SYM2, 4> ddg; // metric drvts //FIXME: symmetries
  AthenaTensor<Real, SYM2, 2> R;   // Ricci 
  
  ig.NewAthenaTensor(2, ncells1); //TODO:check
  ddg.NewAthenaTensor(4, ncells1);
  R.NewAthenaTensor(2, ncells1);

  //TODO: define routines for drvts
  Real const * stencil;
  switch(order) {
    case 2:
      stencil = &diff_2_ord_2[0];
      break;
    case 4:
      stencil = &diff_2_ord_4[0];
      break;
    case 6:
      stencil = &diff_2_ord_6[0];
      break;
    case 8:
      stencil = &diff_2_ord_8[0];
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
    // K rhs , broken in several pieces along i-direction
    
    for(int k = ks; k <= ke; ++k) {
#pragma omp for schedule(static)
      for(int j = js; j <= je; ++j) {
	
	//----------------------------------------------------------------------------------------
	// Inverse metric 
	
	//TODO: spatial_det() and spatial_inv() are temporary defined in vwave.cpp

#pragma omp simd
	for(int i = is; i <= ie; ++i) {
	  Real det = spatial_det( g(0,0,i,j,k), g(0,1,i,j,k), g(0,2,i,j,k), 
				  g(1,1,i,j,k), g(1,2,i,j,k), g(2,2,i,j,k) );
	  spatial_inv( det,
		       g(0,0,i,j,k), g(0,1,i,j,k), g(0,2,i,j,k), 
		       g(1,1,i,j,k), g(1,2,i,j,k), g(2,2,i,j,k), 
		       &ig(0,0,i), &ig(0,1,i), &ig(0,2,i), 
		       &ig(1,1,i), &ig(1,2,i), &ig(2,2,i) );
	  
	}
	
	//----------------------------------------------------------------------------------------
	// Metric drvts i-direction
	
	for(int a = 0; a < NDIM; ++a) {
	  for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	    for(int i = is; i <= ie; ++i) {
	      // assume Cartesian coordinates!
	      Real dd_loc = 0;
	      for(int n = 0; n < stencil_size; ++n) {
		dd_loc += stencil[n]*u(0, k, j, i + n - stencil_offset);
	      }
	      dd_loc *= SQR(c)/SQR(pco->dx1v(i));
	      ddg(a,b,i) = dd_loc;
	    }
	  }
	}
	
	//----------------------------------------------------------------------------------------
	// Metric drvts j-direction
	
	if(pmb->block_size.nx2 > 1) {
	  for(int a = 0; a < NDIM; ++a) {
	    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	      for(int i = is; i <= ie; ++i) {
		Real dd_loc = 0;
		for(int n = 0; n < stencil_size; ++n) {
		  dd_loc += stencil[n]*u(0, k, j + n - stencil_offset, i);
		}
		dd_loc *= SQR(c)/SQR(pco->dx2v(j));
		ddg(a,b,i) += dd_loc;
	      }
	    }
	  }
	}
	
	//----------------------------------------------------------------------------------------
	// Metric drvts k-direction
	
	if(pmb->block_size.nx3 > 1) {
	  for(int a = 0; a < NDIM; ++a) {
	    for(int b = a; b < NDIM; ++b) {
#pragma omp simd
	      for(int i = is; i <= ie; ++i) {
		Real dd_loc = 0;
		for(int n = 0; n < stencil_size; ++n) {
		  dd_loc += stencil[n]*u(0, k + n - stencil_offset, j, i);
		}
		dd_loc *= SQR(c)/SQR(pco->dx3v(k));
		ddg(a,b,i) += dd_loc;
	      }
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
		  R(a,b,i) -= 0.5*ig(c,d,i)*ddg(c,d,a,b,i);
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
	      rhs_K(a,b,i,j,k) = R(a,b,i);
	    }
	  }
	}
	
      } // j - loop
    } // k - loop
    
  } // end of parallel region


  ig.DeleteAthenaTensor();
  ddg.DeleteAthenaTensor();
  R.DeleteAthenaTensor();
  
  return;
}
