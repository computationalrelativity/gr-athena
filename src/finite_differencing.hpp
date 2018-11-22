#ifndef FINITE_DIFFERENCING_HPP_
#define FINITE_DIFFERENCING_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file finite_differencing.hpp
//  \brief High-performance finite-differencing kernel

#include "athena.hpp"
#include "athena_arrays.hpp"

// Finite differencing stencil
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * order  : Order of the finite differencing, eg, 2 for 2nd order accurate
// * offset : Position at which the derivative is computed wrt the stencil.
//            Example: offset=order/2 for central finite differencing
template<int degree_, int order_, int offset_>
class FDStencil {
  public:
    // Degree of the derivative to be approximated
    enum {degree = degree_};
    // Order of the finite differencing
    enum {order = order_};
    // Position at which the derivative is computed wrt the beginning of the stencil
    enum {offset = offset_};
    // Width of the stencil
    enum {width = order + 1};
    // Finite differencing coefficients
    static Real const coeff[width];
};

// Centered finite differencing stencils
// * degree : Degree of the derivative, eg, 1 for 1st derivative
// * order  : Order of the finite differencing, eg, 2 for 2nd order accurate
template<int degree_, int nghost_>
class FDCenteredStencil {
  public:
    // Degree of the derivative to be approximated
    enum {degree = degree_};
    // Order of the finite differencing
    enum {order = 2*nghost_};
    // Position at which the derivative is computed wrt the beginning of the stencil
    enum {offset = nghost_};
    // Width of the stencil
    enum {width = order + 1};
    // Finite differencing coefficients
    static Real const coeff[width];
};

// Finite differencing kernel for homogeneous derivatives
template<typename stencil, typename T>
class FDKernelH {
  public:
    // Initialize a finite differencing kernel
    // . da  -> direction in which to take the derivative, between 0 and 4
    // . fun -> function to differentiate
    FDKernelH(int da, AthenaArray<T> & fun);
    // Compute the undivided finite differences at a given location
    T operator()(int const n) const {
      return eval_(fun_.data(n));
    }
    T operator()(int const n, int const i) const {
      return eval_(fun_.data(n,i));
    }
    T operator()(int const n, int const j, int const i) const {
      return eval_(fun_.data(n,j,i));
    }
    T operator()(int const n, int const k, int const j, int const i) const {
      return eval_(fun_.data(n,k,j,i));
    }
    T operator()(int const m, int const n, int const k, int const j, int const i) const {
      return eval_(fun_.data(m,n,k,j,i));
    }
  private:
    inline T eval_(T const * fun) const;
  private:
    int stride_;
    Real weight_;
    AthenaArray<T> fun_;
};

// Finite differencing kernel for mixed derivatives
template<typename stencil, typename T>
class FDKernelM {
  public:
    // Initialize a finite differencing kernel
    // . da  -> direction in which to take the first derivative, between 0 and 4
    // . db  -> direction in which to take the second derivative, between 0 and 4
    // . fun -> function to differentiate
    // Initialize the FD kernel with given strides
    FDKernelM(int da, int db, AthenaArray<T> & fun);
    // Compute the undivided finite differences at a given location
    T operator()(int const n) {
      return eval_(fun_.data(n));
    }
    T operator()(int const n, int const i) const {
      return eval_(fun_.data(n,i));
    }
    T operator()(int const n, int const j, int const i) const {
      return eval_(fun_.data(n,j,i));
    }
    T operator()(int const n, int const k, int const j, int const i) const {
      return eval_(fun_.data(n,k,j,i));
    }
    T operator()(int const m, int const n, int const k, int const j, int const i) const {
      return eval_(fun_.data(m,n,k,j,i));
    }
  private:
    inline T eval_(T const * fun) const;
  private:
    int stride1_;
    int stride2_;
    Real weight_;
    AthenaArray<T> fun_;
};

// ---- definitions -----------------------------------------------------------

template<typename stencil, typename T>
FDKernelH<stencil, T>::FDKernelH(int const da, AthenaArray<T> & fun) {
  fun_.InitWithShallowCopy(fun);
  if(fun_.GetDim(da+1) < stencil::width) {
    stride_ = 0;
    weight_ = 0.0;
  }
  else {
    stride_ = fun_.GetStride(da+1);
    weight_ = 1.0;
  }
}

template<typename stencil, typename T>
T FDKernelH<stencil, T>::eval_(T const * fun) const {
  T out(0);
  for(int n = 0; n < stencil::width; ++n) {
    out += stencil::coeff[n]*fun[(n - stencil::offset)*stride_];
  }
  return out*weight_;
}

template<typename stencil, typename T>
FDKernelM<stencil, T>::FDKernelM(int da, int db, AthenaArray<T> & fun) {
  fun_.InitWithShallowCopy(fun);
  if(fun_.GetDim(da+1) < stencil::width) {
    stride1_ = 0;
    stride2_ = 0;
    weight_ = 0.0;
  }
  else if(fun_.GetDim(db+1) < stencil::width) {
    stride1_ = 0;
    stride2_ = 0;
    weight_ = 0.0;
  }
  else {
    stride1_ = fun_.GetStride(da+1);
    stride2_ = fun_.GetStride(db+1);
    weight_  = 1.0;
  }
}

template<typename stencil, typename T>
T FDKernelM<stencil, T>::eval_(T const * fun) const {
  T out(0);
  for(int n1 = 0; n1 < stencil::width; ++n1)
  for(int n2 = 0; n2 < stencil::width; ++n2) {
    out += stencil::coeff[n1]*stencil::coeff[n2]*fun[
      (n1 - stencil::offset)*stride1_ +
      (n2 - stencil::offset)*stride2_];
  }
  return out*weight_;
}

#endif
