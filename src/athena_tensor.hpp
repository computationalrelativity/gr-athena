#ifndef ATHENA_TENSOR_HPP_
#define ATHENA_TENSOR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_tensor.hpp
//  \brief provides classes for tensor-like fields
//
//  Convention: indices a,b,c,d are tensor indices. Indices n,i,j,k are grid indices.

#include <cassert>
#include <iostream>

#include "athena_arrays.hpp"
#include "athena.hpp"

// tensor symmetries
enum class TensorSymm {
  NONE,     // no symmetries
  SYM2,     // symmetric in the last 2 indices
  ISYM2,    // symmetric in the first 2 indices
  SYM22,    // symmetric in the last 2 pairs of indices
};

//----------------------------------------------------------------------------------------
// Compile-time index map computation helpers
namespace tensor_detail {

// Generic result struct for index maps of arbitrary rank.
// Data is stored in a flat array; accessor helpers convert multi-indices to flat offsets.
template<int N>
struct IdxMapResult {
  int data[N];
  int ndof;
};

// Rank-2 index map computation
template<TensorSymm sym, int ndim>
constexpr IdxMapResult<ndim * ndim> compute_idxmap_rank2() {
  IdxMapResult<ndim * ndim> r{};
  r.ndof = 0;
  if constexpr (sym == TensorSymm::NONE) {
    for (int a = 0; a < ndim; ++a)
    for (int b = 0; b < ndim; ++b) {
      r.data[a * ndim + b] = r.ndof++;
    }
  } else if constexpr (sym == TensorSymm::SYM2 || sym == TensorSymm::ISYM2) {
    for (int a = 0; a < ndim; ++a)
    for (int b = a; b < ndim; ++b) {
      r.data[a * ndim + b] = r.ndof++;
      r.data[b * ndim + a] = r.data[a * ndim + b];
    }
  }
  return r;
}

// Rank-3 index map computation
template<TensorSymm sym, int ndim>
constexpr IdxMapResult<ndim * ndim * ndim> compute_idxmap_rank3() {
  IdxMapResult<ndim * ndim * ndim> r{};
  r.ndof = 0;
  if constexpr (sym == TensorSymm::NONE) {
    for (int a = 0; a < ndim; ++a)
    for (int b = 0; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c) {
      r.data[a * ndim * ndim + b * ndim + c] = r.ndof++;
    }
  } else if constexpr (sym == TensorSymm::SYM2) {
    for (int a = 0; a < ndim; ++a)
    for (int b = 0; b < ndim; ++b)
    for (int c = b; c < ndim; ++c) {
      r.data[a * ndim * ndim + b * ndim + c] = r.ndof++;
      r.data[a * ndim * ndim + c * ndim + b] = r.data[a * ndim * ndim + b * ndim + c];
    }
  } else if constexpr (sym == TensorSymm::ISYM2) {
    for (int a = 0; a < ndim; ++a)
    for (int b = a; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c) {
      r.data[a * ndim * ndim + b * ndim + c] = r.ndof++;
      r.data[b * ndim * ndim + a * ndim + c] = r.data[a * ndim * ndim + b * ndim + c];
    }
  }
  return r;
}

// Rank-4 index map computation
template<TensorSymm sym, int ndim>
constexpr IdxMapResult<ndim * ndim * ndim * ndim> compute_idxmap_rank4() {
  constexpr int N2 = ndim * ndim;
  constexpr int N3 = ndim * ndim * ndim;
  IdxMapResult<ndim * ndim * ndim * ndim> r{};
  r.ndof = 0;
  if constexpr (sym == TensorSymm::NONE) {
    for (int a = 0; a < ndim; ++a)
    for (int b = 0; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c)
    for (int d = 0; d < ndim; ++d) {
      r.data[a*N3 + b*N2 + c*ndim + d] = r.ndof++;
    }
  } else if constexpr (sym == TensorSymm::SYM2) {
    for (int a = 0; a < ndim; ++a)
    for (int b = 0; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c)
    for (int d = c; d < ndim; ++d) {
      r.data[a*N3 + b*N2 + c*ndim + d] = r.ndof++;
      r.data[a*N3 + b*N2 + d*ndim + c] = r.data[a*N3 + b*N2 + c*ndim + d];
    }
  } else if constexpr (sym == TensorSymm::ISYM2) {
    for (int a = 0; a < ndim; ++a)
    for (int b = a; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c)
    for (int d = 0; d < ndim; ++d) {
      r.data[a*N3 + b*N2 + c*ndim + d] = r.ndof++;
      r.data[b*N3 + a*N2 + c*ndim + d] = r.data[a*N3 + b*N2 + c*ndim + d];
    }
  } else if constexpr (sym == TensorSymm::SYM22) {
    for (int a = 0; a < ndim; ++a)
    for (int b = a; b < ndim; ++b)
    for (int c = 0; c < ndim; ++c)
    for (int d = c; d < ndim; ++d) {
      r.data[a*N3 + b*N2 + c*ndim + d] = r.ndof++;
      r.data[b*N3 + a*N2 + c*ndim + d] = r.data[a*N3 + b*N2 + c*ndim + d];
      r.data[a*N3 + b*N2 + d*ndim + c] = r.data[a*N3 + b*N2 + c*ndim + d];
      r.data[b*N3 + a*N2 + d*ndim + c] = r.data[a*N3 + b*N2 + c*ndim + d];
    }
  }
  return r;
}

} // namespace tensor_detail


// this is the abstract base class
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaTensor: scalar fields
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 0>
{
public:
  // xtors: -------------------------------------------------------------------
  AthenaTensor() { }  // NewAthenaTensor needed for alloc.
  // Allocate at point of decl.
  AthenaTensor(int nx1) { NewAthenaTensor(nx1); }
  AthenaTensor(int nx1, int nx2) { NewAthenaTensor(nx1, nx2); }
  AthenaTensor(int nx1, int nx2, int nx3) { NewAthenaTensor(nx1, nx2, nx3); }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4);
  }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4, int nx5)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4, nx5);
  }
  // Shallow slice at point of decl.
  AthenaTensor(AthenaArray<T> &src, const int indx)
  {
    InitWithShallowSlice(src, indx);
  }
  // Cleanup
  ~AthenaTensor() { DeleteAthenaTensor(); }

  AthenaTensor(AthenaTensor<T, sym, ndim, 0> const &) = default;
  AthenaTensor<T, sym, ndim, 0> & operator=(AthenaTensor<T, sym, ndim, 0> const &) = default;
  // --------------------------------------------------------------------------

  // functions to allocate/de-allocate the data
  void NewAthenaTensor(int nx1) {
    data_.NewAthenaArray(nx1);
  }
  void NewAthenaTensor(int nx1, int nx2) {
    data_.NewAthenaArray(nx1, nx2);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3) {
    data_.NewAthenaArray(nx1, nx2, nx3);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4) {
    data_.NewAthenaArray(nx1, nx2, nx3, nx4);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4, int nx5) {
    data_.NewAthenaArray(nx1, nx2, nx3, nx4, nx5);
  }
  void DeleteAthenaTensor() {
    data_.DeleteAthenaArray();
  }

  // USE WITH CAUTION!
  // Operands must both be allocated and have same sizes
  void SwapAthenaTensor(AthenaTensor<T, sym, ndim, 0>& other)
  {
    std::swap(data_, other.data_);
    return;
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  static constexpr int idxmap() {
    return 0;
  }
  static constexpr int ndof() {
    return 1;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }
  // mask by multiplicative constant
  void MulConst(T const val) { data_.MulConst(val); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  AthenaArray<T> const & operator()() {
    return data_;
  }
  T & operator()(int const i) {
    return data_(i);
  }
  T & operator()(int const j, int const i) {
    return data_(j,i);
  }
  T & operator()(int const k, int const j, int const i) {
    return data_(k,j,i);
  }
  T & operator()(int const n, int const k, int const j, int const i) {
    return data_(n,k,j,i);
  }
  T & operator()(int const m, int const n, int const k, int const j, int const i) {
    return data_(m,n,k,j,i);
  }
  T operator()(int const i) const {
    return data_(i);
  }
  T operator()(int const j, int const i) const {
    return data_(j,i);
  }
  T operator()(int const k, int const j, int const i) const {
    return data_(k,j,i);
  }
  T operator()(int const n, int const k, int const j, int const i) const {
    return data_(n,k,j,i);
  }
  T operator()(int const m, int const n, int const k, int const j, int const i) const {
    return data_(m,n,k,j,i);
  }

  // functions that initialize a tensor with shallow slice from an array
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx)
  {
    data_.InitWithShallowSlice(src, indx, ndof());
  }
  void InitWithShallowSlice(AthenaArray<T> &src, const int dim, const int indx)
  {
    data_.InitWithShallowSlice(src, dim, indx, ndof());
  }

  void PrintPoint(const std::string & name,
                  const int i)
  {
    std::cout << name << "(" << i << ") = ";
    std::cout << data_(i) << std::endl;
  }

  void PrintPoint(const std::string & name,
                  const int k, const int j, const int i)
  {
    std::cout << name << "(" << k << "," << j << "," << i << ") = ";
    std::cout << data_(k,j,i) << std::endl;
  }

private:
  AthenaArray<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 1 AthenaTensor: vector and co-vector fields
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 1>
{
public:
  // xtors: -------------------------------------------------------------------
  AthenaTensor() { }  // NewAthenaTensor needed for alloc.
  // Allocate at point of decl.
  AthenaTensor(int nx1) { NewAthenaTensor(nx1); }
  AthenaTensor(int nx1, int nx2) { NewAthenaTensor(nx1, nx2); }
  AthenaTensor(int nx1, int nx2, int nx3) { NewAthenaTensor(nx1, nx2, nx3); }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4);
  }
  // Shallow slice at point of decl.
  AthenaTensor(AthenaArray<T> &src, const int indx)
  {
    InitWithShallowSlice(src, indx);
  }
  // Cleanup
  ~AthenaTensor() { DeleteAthenaTensor(); }

  AthenaTensor(AthenaTensor<T, sym, ndim, 1> const &) = default;
  AthenaTensor<T, sym, ndim, 1> & operator=(AthenaTensor<T, sym, ndim, 1> const &) = default;
  // --------------------------------------------------------------------------

  // functions to allocate/de-allocate the data
  void NewAthenaTensor(int nx1) {
    data_.NewAthenaArray(ndim, nx1);
  }
  void NewAthenaTensor(int nx1, int nx2) {
    data_.NewAthenaArray(ndim, nx1, nx2);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3) {
    data_.NewAthenaArray(ndim, nx1, nx2, nx3);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4) {
    data_.NewAthenaArray(ndim, nx1, nx2, nx3, nx4);
  }
  void DeleteAthenaTensor() {
    data_.DeleteAthenaArray();
  }

  // USE WITH CAUTION!
  // Operands must both be allocated and have same sizes
  void SwapAthenaTensor(AthenaTensor<T, sym, ndim, 1>& other)
  {
    std::swap(data_, other.data_);
    return;
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  static constexpr int idxmap(int const a) {
    return a;
  }
  static constexpr int ndof() {
    return ndim;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // slice vector index:
  // - leaves only grid sampling behind
  // - non-singleton
  void slice(const int a,
             AthenaTensor<T, TensorSymm::NONE, ndim, 0> & tar)
  {
    tar.InitWithShallowSlice(data_, ndim+1, a);
  }

  // operators to access the data
  T & operator()(int const a,
                 int const i) {
    return data_(a,i);
  }
  T & operator()(int const a,
                 int const j, int const i) {
    return data_(a,j,i);
  }
  T & operator()(int const a,
                 int const k, int const j, int const i) {
    return data_(a,k,j,i);
  }
  T & operator()(int const a,
                 int const n, int const k, int const j, int const i) {
    return data_(a,n,k,j,i);
  }
  T operator()(int const a,
               int const i) const {
    return data_(a,i);
  }
  T operator()(int const a,
               int const j, int const i) const {
    return data_(a,j,i);
  }
  T operator()(int const a,
               int const k, int const j, int const i) const {
    return data_(a,k,j,i);
  }
  T operator()(int const a,
               int const n, int const k, int const j, int const i) const {
    return data_(a,n,k,j,i);
  }

  // functions that initialize a tensor with shallow copy or slice from an array
  // functions that initialize a tensor with shallow slice from an array
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx)
  {
    data_.InitWithShallowSlice(src, indx, ndof());
  }

  void PrintPoint(const std::string & name,
                  const int i)
  {
    for (int a=0; a<ndim; ++a)
    {
      std::cout << name << "(" << a << ";";
      std::cout << i << ") = ";
      std::cout << data_(a,i) << std::endl;
    }
  }

  void PrintPoint(const std::string & name,
                  const int k, const int j, const int i)
  {
    for (int a=0; a<ndim; ++a)
    {
      std::cout << name << "(" << a << ";";
      std::cout << k << "," << j << "," << i << ") = ";
      std::cout << data_(a,k,j,i) << std::endl;
    }
  }

private:
  AthenaArray<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 2 AthenaTensor, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 2>
{
  static constexpr auto idxmap_result_ = tensor_detail::compute_idxmap_rank2<sym, ndim>();
public:
  // xtors: -------------------------------------------------------------------
  AthenaTensor() { } // NewAthenaTensor needed for alloc.
  // Allocate at point of decl.
  AthenaTensor(int nx1)
  {
    NewAthenaTensor(nx1);
  }
  AthenaTensor(int nx1, int nx2)
  {
    NewAthenaTensor(nx1, nx2);
  }
  AthenaTensor(int nx1, int nx2, int nx3)
  {
    NewAthenaTensor(nx1, nx2, nx3);
  }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4);
  }
  // Shallow slice at point of decl.
  AthenaTensor(AthenaArray<T> &src, const int indx)
  {
    InitWithShallowSlice(src, indx);
  }
  // Cleanup
  ~AthenaTensor() { DeleteAthenaTensor(); }

  AthenaTensor(AthenaTensor<T, sym, ndim, 2> const &) = default;
  AthenaTensor<T, sym, ndim, 2> & operator=(AthenaTensor<T, sym, ndim, 2> const &) = default;
  // --------------------------------------------------------------------------

  // functions to allocate/de-allocate the data
  void NewAthenaTensor(int nx1) {
    data_.NewAthenaArray(ndof(), nx1);
  }
  void NewAthenaTensor(int nx1, int nx2) {
    data_.NewAthenaArray(ndof(), nx1, nx2);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3, nx4);
  }
  void DeleteAthenaTensor() {
    data_.DeleteAthenaArray();
  }

  // USE WITH CAUTION!
  // Operands must both be allocated and have same sizes
  void SwapAthenaTensor(AthenaTensor<T, sym, ndim, 2>& other)
  {
    std::swap(data_, other.data_);
    return;
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  static constexpr int idxmap(int const a, int const b) {
    return idxmap_result_.data[a * ndim + b];
  }
  static constexpr int ndof() {
    return idxmap_result_.ndof;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b,
                 int const i) {
    return data_(idxmap(a,b),i);
  }
  T & operator()(int const a, int const b,
                 int const j, int const i) {
    return data_(idxmap(a,b),j,i);
  }
  T & operator()(int const a, int const b,
                 int const k, int const j, int const i) {
    return data_(idxmap(a,b),k,j,i);
  }
  T & operator()(int const a, int const b,
                 int const n, int const k, int const j, int const i) {
    return data_(idxmap(a,b),n,k,j,i);
  }
  T operator()(int const a, int const b,
               int const i) const {
    return data_(idxmap(a,b),i);
  }
  T operator()(int const a, int const b,
               int const j, int const i) const {
    return data_(idxmap(a,b),j,i);
  }

  T operator()(int const a, int const b,
               int const k, int const j, int const i) const {
    return data_(idxmap(a,b),k,j,i);
  }

  T operator()(int const a, int const b,
               int const n, int const k, int const j, int const i) const {
    return data_(idxmap(a,b),n,k,j,i);
  }

  // functions that initialize a tensor with shallow slice from an array
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx)
  {
    data_.InitWithShallowSlice(src, indx, ndof());
  }

  void PrintPoint(const std::string & name,
                  const int i)
  {
    switch (sym)
    {
      case TensorSymm::SYM2:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=a; b<ndim; ++b)
        {
          std::cout << name << "(" << a << "," << b << ";";
          std::cout << i << ") = ";
          std::cout << data_(idxmap(a,b),i) << std::endl;
        }
        break;
      }
      case TensorSymm::NONE:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=0; b<ndim; ++b)
        {
          std::cout << name << "(" << a << "," << b << ";";
          std::cout << i << ") = ";
          std::cout << data_(idxmap(a,b),i) << std::endl;
        }
        break;
      }
      default:
      {
        assert(0);
      }
    }
  }

  void PrintPoint(const std::string & name,
                  const int k, const int j, const int i)
  {
    switch (sym)
    {
      case TensorSymm::SYM2:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=a; b<ndim; ++b)
        {
          std::cout << name << "(" << a << "," << b << ";";
          std::cout << k << "," << j << "," << i << ") = ";
          std::cout << data_(idxmap(a,b),k,j,i) << std::endl;
        }
        break;
      }
      case TensorSymm::NONE:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=0; b<ndim; ++b)
        {
          std::cout << name << "(" << a << "," << b << ";";
          std::cout << k << "," << j << "," << i << ") = ";
          std::cout << data_(idxmap(a,b),k,j,i) << std::endl;
        }
        break;
      }
      default:
      {
        assert(0);
      }
    }
  }

private:
  AthenaArray<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 3 AthenaTensor, e.g., Christoffel symbols
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 3>
{
  static constexpr auto idxmap_result_ = tensor_detail::compute_idxmap_rank3<sym, ndim>();
public:
  // xtors: -------------------------------------------------------------------
  AthenaTensor() { } // NewAthenaTensor needed for alloc.
  // Allocate at point of decl.
  AthenaTensor(int nx1)
  {
    NewAthenaTensor(nx1);
  }
  AthenaTensor(int nx1, int nx2)
  {
    NewAthenaTensor(nx1, nx2);
  }
  AthenaTensor(int nx1, int nx2, int nx3)
  {
    NewAthenaTensor(nx1, nx2, nx3);
  }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4);
  }
  // Shallow slice at point of decl.
  AthenaTensor(AthenaArray<T> &src, const int indx)
  {
    InitWithShallowSlice(src, indx);
  }
  // Cleanup
  ~AthenaTensor() { DeleteAthenaTensor(); }

  AthenaTensor(AthenaTensor<T, sym, ndim, 3> const &) = default;
  AthenaTensor<T, sym, ndim, 3> & operator=(AthenaTensor<T, sym, ndim, 3> const &) = default;
  // --------------------------------------------------------------------------

  // functions to allocate/de-allocate the data
  void NewAthenaTensor(int nx1) {
    data_.NewAthenaArray(ndof(), nx1);
  }
  void NewAthenaTensor(int nx1, int nx2) {
    data_.NewAthenaArray(ndof(), nx1, nx2);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3, nx4);
  }
  void DeleteAthenaTensor() {
    data_.DeleteAthenaArray();
  }

  // USE WITH CAUTION!
  // Operands must both be allocated and have same sizes
  void SwapAthenaTensor(AthenaTensor<T, sym, ndim, 3>& other)
  {
    std::swap(data_, other.data_);
    return;
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  static constexpr int idxmap(int const a, int const b, int const c) {
    return idxmap_result_.data[a * ndim * ndim + b * ndim + c];
  }
  static constexpr int ndof() {
    return idxmap_result_.ndof;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b, int const c,
                 int const i) {
    return data_(idxmap(a,b,c),i);
  }
  T & operator()(int const a, int const b, int const c,
                 int const j, int const i) {
    return data_(idxmap(a,b,c),j,i);
  }
  T & operator()(int const a, int const b, int const c,
                 int const k, int const j, int const i) {
    return data_(idxmap(a,b,c),k,j,i);
  }
  T & operator()(int const a, int const b, int const c,
                 int const n, int const k, int const j, int const i) {
    return data_(idxmap(a,b,c),n,k,j,i);
  }
  T operator()(int const a, int const b, int const c,
               int const i) const {
    return data_(idxmap(a,b,c),i);
  }
  T operator()(int const a, int const b, int const c,
               int const j, int const i) const {
    return data_(idxmap(a,b,c),j,i);
  }
  T operator()(int const a, int const b, int const c,
               int const k, int const j, int const i) const {
    return data_(idxmap(a,b,c),k,j,i);
  }
  T operator()(int const a, int const b, int const c,
               int const n, int const k, int const j, int const i) const {
    return data_(idxmap(a,b,c),n,k,j,i);
  }

  // functions that initialize a tensor with shallow slice from an array
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx) {
    data_.InitWithShallowSlice(src, indx, ndof());
  }

  void PrintPoint(const std::string & name,
                  const int k, const int j, const int i)
  {
    switch (sym)
    {
      case TensorSymm::SYM2:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=0; b<ndim; ++b)
        for (int c=b; c<ndim; ++c)
        {
          std::cout << name << "(" << a << "," << b << "," << c << ";";
          std::cout << k << "," << j << "," << i << ") = ";
          std::cout << data_(idxmap(a,b,c),k,j,i) << std::endl;
        }
        break;
      }
      case TensorSymm::NONE:
      {
        for (int a=0; a<ndim; ++a)
        for (int b=0; b<ndim; ++b)
        for (int c=0; c<ndim; ++c)
        {
          std::cout << name << "(" << a << "," << b << "," << c << ";";
          std::cout << k << "," << j << "," << i << ") = ";
          std::cout << data_(idxmap(a,b,c),k,j,i) << std::endl;
        }
        break;
      }
      default:
      {
        assert(0);
      }
    }
  }


private:
  AthenaArray<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 4 AthenaTensor, e.g., mixed derivatives of the metric
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 4>
{
  static constexpr auto idxmap_result_ = tensor_detail::compute_idxmap_rank4<sym, ndim>();
public:
  // xtors: -------------------------------------------------------------------
  AthenaTensor() { } // NewAthenaTensor needed for alloc.
  // Allocate at point of decl.
  AthenaTensor(int nx1)
  {
    NewAthenaTensor(nx1);
  }
  AthenaTensor(int nx1, int nx2)
  {
    NewAthenaTensor(nx1, nx2);
  }
  AthenaTensor(int nx1, int nx2, int nx3)
  {
    NewAthenaTensor(nx1, nx2, nx3);
  }
  AthenaTensor(int nx1, int nx2, int nx3, int nx4)
  {
    NewAthenaTensor(nx1, nx2, nx3, nx4);
  }
  // Shallow slice at point of decl.
  AthenaTensor(AthenaArray<T> &src, const int indx)
  {
    InitWithShallowSlice(src, indx);
  }
  // Cleanup
  ~AthenaTensor() { DeleteAthenaTensor(); }

  AthenaTensor(AthenaTensor<T, sym, ndim, 4> const &) = default;
  AthenaTensor<T, sym, ndim, 4> & operator=(AthenaTensor<T, sym, ndim, 4> const &) = default;
  // --------------------------------------------------------------------------

  // functions to allocate/de-allocate the data
  void NewAthenaTensor(int nx1) {
    data_.NewAthenaArray(ndof(), nx1);
  }
  void NewAthenaTensor(int nx1, int nx2) {
    data_.NewAthenaArray(ndof(), nx1, nx2);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3);
  }
  void NewAthenaTensor(int nx1, int nx2, int nx3, int nx4) {
    data_.NewAthenaArray(ndof(), nx1, nx2, nx3, nx4);
  }
  void DeleteAthenaTensor() {
    data_.DeleteAthenaArray();
  }

  // USE WITH CAUTION!
  // Operands must both be allocated and have same sizes
  void SwapAthenaTensor(AthenaTensor<T, sym, ndim, 4>& other)
  {
    std::swap(data_, other.data_);
    return;
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  static constexpr int idxmap(int const a, int const b, int const c, int const d) {
    return idxmap_result_.data[a * ndim * ndim * ndim + b * ndim * ndim + c * ndim + d];
  }
  static constexpr int ndof() {
    return idxmap_result_.ndof;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b, int const c, int const d,
                 int const i) {
    return data_(idxmap(a,b,c,d),i);
  }
  T & operator()(int const a, int const b, int const c, int const d,
                 int const j, int const i) {
    return data_(idxmap(a,b,c,d),j,i);
  }
  T & operator()(int const a, int const b, int const c, int const d,
                 int const k, int const j, int const i) {
    return data_(idxmap(a,b,c,d),k,j,i);
  }
  T & operator()(int const a, int const b, int const c, int const d,
                 int const n, int const k, int const j, int const i) {
    return data_(idxmap(a,b,c,d),n,k,j,i);
  }
  T operator()(int const a, int const b, int const c, int const d,
               int const i) const {
    return data_(idxmap(a,b,c,d),i);
  }
  T operator()(int const a, int const b, int const c, int const d,
               int const j, int const i) const {
    return data_(idxmap(a,b,c,d),j,i);
  }
  T operator()(int const a, int const b, int const c, int const d,
               int const k, int const j, int const i) const {
    return data_(idxmap(a,b,c,d),k,j,i);
  }
  T operator()(int const a, int const b, int const c, int const d,
               int const n, int const k, int const j, int const i) const {
    return data_(idxmap(a,b,c,d),n,k,j,i);
  }

  // functions that initialize a tensor with shallow slice from an array
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx) {
    data_.InitWithShallowSlice(src, indx, ndof());
  }
private:
  AthenaArray<T> data_;
};

#endif
