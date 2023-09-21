//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file weighted_ave.cpp
//  \brief

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "mesh.hpp"

namespace {

// Weighted average kernel (used for both CC and VC vars)
void WeightedAve_(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                  AthenaArray<Real> &u_in2, const Real wght[3],
                  int il, int jl, int kl, int iu, int ju, int ku) {
  // assuming all 3x arrays are of the same size (or at least u_out il equal or larger
  // than each input array) in each array dimension, and full range il desired:
  // nx4*(3D real MeshBlock cells)
  const int nu = u_out.GetDim4() - 1;

  // u_in2 may be an unallocated AthenaArray if using a 2S time integrator
  if (wght[0] == 1.0) {
    if (wght[2] != 0.0) {
      for (int n=0; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              u_out(n,k,j,i) += wght[1]*u_in1(n,k,j,i) + wght[2]*u_in2(n,k,j,i);
            }
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        for (int n=0; n<=nu; ++n) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                u_out(n,k,j,i) += wght[1]*u_in1(n,k,j,i);
              }
            }
          }
        }
      }
    }
  } else if (wght[0] == 0.0) {
    if (wght[2] != 0.0) {
      for (int n=0; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              u_out(n,k,j,i) = wght[1]*u_in1(n,k,j,i) + wght[2]*u_in2(n,k,j,i);
            }
          }
        }
      }
    } else if (wght[1] == 1.0) {
      // just deep copy
      for (int n=0; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              u_out(n,k,j,i) = u_in1(n,k,j,i);
            }
          }
        }
      }
    } else {
      for (int n=0; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              u_out(n,k,j,i) = wght[1]*u_in1(n,k,j,i);
            }
          }
        }
      }
    }
  } else {
    if (wght[2] != 0.0) {
      for (int n=0; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              u_out(n,k,j,i) = wght[0]*u_out(n,k,j,i) + wght[1]*u_in1(n,k,j,i)
                               + wght[2]*u_in2(n,k,j,i);
            }
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        for (int n=0; n<=nu; ++n) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                u_out(n,k,j,i) = wght[0]*u_out(n,k,j,i) + wght[1]*u_in1(n,k,j,i);
              }
            }
          }
        }
      } else { // do not dereference u_in1
        for (int n=0; n<=nu; ++n) {
          for (int k=kl; k<=ku; ++k) {
            for (int j=jl; j<=ju; ++j) {
#pragma omp simd
              for (int i=il; i<=iu; ++i) {
                u_out(n,k,j,i) *= wght[0];
              }
            }
          }
        }
      }
    }
  }
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn  void MeshBlock::WeightedAveCC
//  \brief Compute weighted average of cell-centered AthenaArrays

void MeshBlock::WeightedAveCC(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                              AthenaArray<Real> &u_in2, const Real wght[3]) {
  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2

  // This property would be better to derive based on general input type...
  int il = is, jl = js, kl = ks;
  int iu = ie, ju = je, ku = ke;

  WeightedAve_(u_out, u_in1, u_in2, wght, il, jl, kl, iu, ju, ku);
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBlock::WeightedAveVC
//  \brief Compute weighted average of vertex-cented AthenaArrays

void MeshBlock::WeightedAveVC(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                              AthenaArray<Real> &u_in2, const Real wght[3]) {
  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2

  // This property would be better to derive based on general input type...
  int il = is, jl = js, kl = ks;
  int iu = ive, ju = jve, ku = kve;

  WeightedAve_(u_out, u_in1, u_in2, wght, il, jl, kl, iu, ju, ku);
}

void MeshBlock::WeightedAveCX(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                              AthenaArray<Real> &u_in2, const Real wght[3]) {
  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2

  // This property would be better to derive based on general input type...
  int il = cx_is, jl = cx_js, kl = cx_ks;
  int iu = cx_ie, ju = cx_je, ku = cx_ke;

  WeightedAve_(u_out, u_in1, u_in2, wght, il, jl, kl, iu, ju, ku);
}