//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file reflect_vc.cpp
//  \brief implementation of reflecting BCs in each dimension

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "bvals_cx.hpp"
// #include "../../z4c/z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectInnerX1(
//          Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh)
//  \brief REFLECTING boundary conditions, inner x1 boundary

void CellCenteredXBoundaryVariable::ReflectInnerX1(
    Real time, Real dt, int il, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<=nu_; ++n) {
//     if(n == (Z4c::I_Z4c_gxy) || n == (Z4c::I_Z4c_gxz) || n == (Z4c::I_Z4c_Axy) || n == (Z4c::I_Z4c_Axz) || n == (Z4c::I_Z4c_Gamx) || n == (Z4c::I_Z4c_betax)){
//       for (int k=kl; k<=ku; ++k) {
//         for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
//           for (int i=1; i<=ngh; ++i) {
//             (*var_cx)(n,k,j,il-i) = -(*var_cx)(n,k,j,(il+i));
//           }
//         }
//       }
//     }
//     else
    {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            (*var_cx)(n,k,j,il-i) = (*var_cx)(n,k,j,(il+i));
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectOuterX1(
//          Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh)
//  \brief REFLECTING boundary conditions, outer x1 boundary

void CellCenteredXBoundaryVariable::ReflectOuterX1(
    Real time, Real dt, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<=nu_; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=1; i<=ngh; ++i) {
          (*var_cx)(n,k,j,iu+i) = (*var_cx)(n,k,j,(iu-i+1));
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectInnerX2(
//          Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh)
//  \brief REFLECTING boundary conditions, inner x2 boundary

void CellCenteredXBoundaryVariable::ReflectInnerX2(
    Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh) {
  for (int n=0; n<=nu_; ++n) {
//     if(n == (Z4c::I_Z4c_gxy) || n == (Z4c::I_Z4c_gyz) || n == (Z4c::I_Z4c_Axy) || n == (Z4c::I_Z4c_Ayz) || n == (Z4c::I_Z4c_Gamy) || n == (Z4c::I_Z4c_betay)){
//       for (int k=kl; k<=ku; ++k) {
//         for (int j=1; j<=ngh; ++j) {
// #pragma omp simd
//           for (int i=il; i<=iu; ++i) {
// 	    (*var_cx)(n,k,jl-j,i) = -(*var_cx)(n,k,jl+j,i);
//           }
//         }
//       }
//     }
//     else
    {
      for (int k=kl; k<=ku; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            (*var_cx)(n,k,jl-j,i) = (*var_cx)(n,k,jl+j,i);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectOuterX2(
//          Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh)
//  \brief REFLECTING boundary conditions, outer x2 boundary

void CellCenteredXBoundaryVariable::ReflectOuterX2(
    Real time, Real dt, int il, int iu, int ju, int kl, int ku, int ngh) {
  for (int n=0; n<=nu_; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          (*var_cx)(n,k,ju+j,i) = (*var_cx)(n,k,ju-j+1,i);
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectInnerX3(
//          Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh)
//  \brief REFLECTING boundary conditions, inner x3 boundary

void CellCenteredXBoundaryVariable::ReflectInnerX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ngh) {
  for (int n=0; n<=nu_; ++n) {
//    if(n == (Z4c::I_Z4c_gxz) || n == (Z4c::I_Z4c_gyz) || n == (Z4c::I_Z4c_Axz) || n == (Z4c::I_Z4c_Ayz) || n == (Z4c::I_Z4c_Gamz) || n == (Z4c::I_Z4c_betaz)){
//     for (int k=1; k<=ngh; ++k) {
//       for (int j=jl; j<=ju; ++j) {
// #pragma omp simd
//         for (int i=il; i<=iu; ++i) {
//           (*var_cx)(n,kl-k,j,i) = -(*var_cx)(n,kl+k,j,i);
//         }
//       }
//     }
//    }
//    else
   {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          (*var_cx)(n,kl-k,j,i) = (*var_cx)(n,kl+k,j,i);
        }
      }
    }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenteredXBoundaryVariable::ReflectOuterX3(
//          Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh)
//  \brief REFLECTING boundary conditions, outer x3 boundary

void CellCenteredXBoundaryVariable::ReflectOuterX3(
    Real time, Real dt, int il, int iu, int jl, int ju, int ku, int ngh) {
  for (int n=0; n<=nu_; ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          (*var_cx)(n,ku+k,j,i) = (*var_cx)(n,ku-k+1,j,i);
        }
      }
    }
  }
  return;
}
