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
  1./2., 0., -1./2.,
};
static Real const __diff_2_stencil_2[] = {
  1., -2., 1.,
};

static Real const __diff_1_stencil_4[] = {
  1./12., -2./3., 0., 2./3., -1./12.,
};
static Real const __diff_2_stencil_4[] = {
  -1./12., 4./3., -5./2., 4./3., -1./12.
};

static Real const __diff_1_stencil_6[] = {
  -1./60., 3./20., -3./4., 0., 3./4., -3./20., 1/60
};
static Real const __diff_2_stencil_6[] = {
  1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.
};

static Real const __diff_1_stencil_8[] = {
  1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.
};
static Real const __diff_2_stencil_8[] = {
  -1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.
};

class FDKernel {
  public:
    FDKernel(int order): 
      st_order_(order),
      st_siz_(order + 1),
      st_offset_(order/2) {
      switch(order) {
        case 2:
          diff_1_st_ = __diff_1_stencil_2;
          diff_2_st_ = __diff_2_stencil_2;
          break;
        case 4:
          diff_1_st_ = __diff_1_stencil_4;
          diff_2_st_ = __diff_2_stencil_4;
          break;
        case 6:
          diff_1_st_ = __diff_1_stencil_6;
          diff_2_st_ = __diff_2_stencil_6;
          break;
        case 8:
          diff_1_st_ = __diff_1_stencil_8;
          diff_2_st_ = __diff_2_stencil_8;
          break;
        default:
          std::stringstream msg;
          msg << "FD order not supported: " << order << std::endl;
          throw std::runtime_error(msg.str().c_str());
      }
    }

    inline Real diff_1(int const da,
        AthenaArray<Real> const & fun, int const i, int const j, int const k) {
      Real out = 0.0;
      for(int n = 0; n < st_siz_; ++n) {
        out += diff_1_st_[n]*fun(
            i + (da == 0)*(n - st_offset_),
            j + (da == 1)*(n - st_offset_),
            k + (da == 2)*(n - st_offset_));
      }
      return out;
    }

    inline Real diff_2(int const da, int const db,
        AthenaArray<Real> const & fun, int const i, int const j, int const k) {
      if(da == db) {
        return diff_aa_(da, fun, i, j, k);
      }
      else {
        return diff_ab_(da, db, fun, i, j, k);
      }
    }
  private:
    int const st_order_;
    int const st_siz_;
    int const st_offset_;
    Real const * diff_1_st_;
    Real const * diff_2_st_;

    inline Real diff_aa_(int const da,
        AthenaArray<Real> const & fun, int const i, int const j, int const k) {
      Real out = 0.0;
      for(int n = 0; n < st_siz_; ++n) {
        out += diff_2_st_[n]*fun(
            i + (da == 0)*(n - st_offset_),
            j + (da == 1)*(n - st_offset_),
            k + (da == 2)*(n - st_offset_));
      }
      return out;
    }
    inline Real diff_ab_(int const da, int const db,
        AthenaArray<Real> const & fun, int const i, int const j, int const k) {
      Real out = 0.0;
      for(int na = 0; na < st_siz_; ++na)
      for(int nb = 0; nb < st_siz_; ++nb) {
        out += diff_1_st_[na]*diff_1_st_[nb]*fun(
          i + (da == 0)*(na - st_offset_) + (db == 0)*(nb - st_offset_),
          j + (da == 1)*(na - st_offset_) + (db == 1)*(nb - st_offset_),
          k + (da == 2)*(na - st_offset_) + (db == 2)*(nb - st_offset_));
      }
      return out;
    }
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

  g.array().InitWithShallowSlice(u, gab_IDX, 6);
  K.array().InitWithShallowSlice(u, Kab_IDX, 6);

  rhs_g.array().InitWithShallowSlice(rhs, gab_IDX, 6);
  rhs_K.array().InitWithShallowSlice(rhs, Kab_IDX, 6);

  FDKernel fd(order);

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
        for(int a = 0; a < NDIM; ++a)
        for(int b = a; b < NDIM; ++b) {
#pragma omp simd
          for(int i = is; i <= ie; ++i) {
            rhs_g(a,b,k,j,i) = -2*K(a,b,k,j,i);
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
          for(int a = 0; a < NDIM; ++a)
          for(int b = a; b < NDIM; ++b) {
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
        for(int c = 0; c < NDIM; ++c)
        for(int d = c; d < NDIM; ++d)
        for(int a = 0; a < NDIM; ++a)
        for(int b = a; b < NDIM; ++b) {
#pragma omp simd
          for(int i = is; i <= ie; ++i) {
            ddg(c,d,a,b,i) = fd.diff_2(c, d, g(a,b), k, j, i);
          }
        }
      
      
        //----------------------------------------------------------------------------------------
        // Ricci tensor
      
        for(int a = 0; a < NDIM; ++a)
        for(int b = a; b < NDIM; ++b) {
#pragma omp simd
          for(int i = is; i <= ie; ++i) {
            R(a,b,i) = 0.;
            for(int c = 0; c < NDIM; ++c)
            for(int d = 0; d < NDIM; ++d) {
              R(a,b,i) -= 0.5*ieta(c,d,i)*ddg(c,d,a,b,i);
            }
          }
        }
      
        //----------------------------------------------------------------------------------------
        // K rhs 
      
        for(int a = 0; a < NDIM; ++a)
        for(int b = a; b < NDIM; ++b) {
#pragma omp simd
          for(int i = is; i <= ie; ++i) {
            rhs_K(a,b,k,j,i) = R(a,b,i);
          }
        }
      } // j - loop
    } // k - loop
    
  } // end of parallel region

  return;
}
