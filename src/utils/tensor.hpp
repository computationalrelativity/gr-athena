#ifndef UTILS_TENSOR_HPP_
#define UTILS_TENSOR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tensor.hpp
//  \brief provides classes for tensor symmetries of fields described pointwise
//
//  Convention: indices a,b,c,d are tensor indices. Indices n,i,j,k are grid indices.

#include <cassert>

#include "../athena.hpp"
#include "../athena_arrays.hpp"

namespace utils
{
namespace tensor
{

// structures =================================================================

// tensor symmetries
enum class Symmetries
{
  NONE,     // no symmetries
  SYM2,     // symmetric in the last 2 indices
  ISYM2,    // symmetric in the first 2 indices
  SYM22,    // symmetric in the last 2 pairs of indices
};

// this is the abstract base class
template<typename T, Symmetries sym, int ndim, int rank>
class TensorPointwise;

// ----------------------------------------------------------------------------
// rank 0 TensorPointwise: scalar fields
template<typename T, Symmetries sym, int ndim>
class TensorPointwise<T, sym, ndim, 0> {
public:
  // the default constructor/destructor/copy operators are sufficient
  TensorPointwise() = default;
  ~TensorPointwise() = default;
  TensorPointwise(
    TensorPointwise<T, sym, ndim, 0> const &
  ) = default;
  TensorPointwise<T, sym, ndim, 0> & operator=(
    TensorPointwise<T, sym, ndim, 0> const &
  ) = default;

  // functions to allocate/de-allocate the data
  void NewTensorPointwise() {
    data_.NewAthenaArray(1);
  }

  void DeleteTensorPointwise() {
    data_.DeleteAthenaArray();
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  int idxmap() const {
    return 0;
  }
  int ndof() const {
    return 1;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()() {
    return data_(0);
  }

  T operator()() const {
    return data_(0);
  }

public:
  // debug functionality
  void print_all()
  {
    AthenaArray<T> arr = this->array();
    arr.print_all();
  }

private:
  AthenaArray<T> data_;
};

// ----------------------------------------------------------------------------
// rank 1 TensorPointwise: vector and co-vector fields
template<typename T, Symmetries sym, int ndim>
class TensorPointwise<T, sym, ndim, 1> {
public:
  // the default constructor/destructor/copy operators are sufficient
  TensorPointwise() = default;
  ~TensorPointwise() = default;
  TensorPointwise(
    TensorPointwise<T, sym, ndim, 1> const &
  ) = default;
  TensorPointwise<T, sym, ndim, 1> & operator=(
    TensorPointwise<T, sym, ndim, 1> const &
  ) = default;

  // functions to allocate/de-allocate the data
  void NewTensorPointwise() {
    data_.NewAthenaArray(ndim);
  }
  void DeleteTensorPointwise() {
    data_.DeleteAthenaArray();
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  int idxmap(int const a) const {
    return a;
  }
  int ndof() const {
    return ndim;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a) {
    return data_(a);
  }
  T operator()(int const a) const {
    return data_(a);
  }

public:
  // debug functionality
  void print_all()
  {
    AthenaArray<T> arr = this->array();
    arr.print_all();
  }

private:
  AthenaArray<T> data_;
  AthenaArray<T> slice_;
};

// ----------------------------------------------------------------------------
// rank 2 TensorPointwise, e.g., the metric or the extrinsic curvature
template<typename T, Symmetries sym, int ndim>
class TensorPointwise<T, sym, ndim, 2> {
public:
  TensorPointwise();
  // the default destructor/copy operators are sufficient
  ~TensorPointwise() = default;
  TensorPointwise(
    TensorPointwise<T, sym, ndim, 2> const &
  ) = default;
  TensorPointwise<T, sym, ndim, 2> & operator=(
    TensorPointwise<T, sym, ndim, 2> const &
  ) = default;

  // functions to allocate/de-allocate the data
  void NewTensorPointwise() {
    data_.NewAthenaArray(ndof_);
  }
  void DeleteTensorPointwise() {
    data_.DeleteAthenaArray();
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  int ndof() const {
    return ndof_;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b) {
    return data_(idxmap_[a][b]);
  }
  T operator()(int const a, int const b) const {
    return data_(idxmap_[a][b]);
  }

  // functions that initialize a tensor with shallow copy or slice from an
  // array
  void InitWithShallowCopy(AthenaArray<T> &src) {
    data_.InitWithShallowCopy(src);
  }
  void InitWithShallowSlice(AthenaArray<T> &src, const int indx) {
    data_.InitWithShallowSlice(src, indx, ndof());
  }

public:
  // debug functionality
  void print_all()
  {
    AthenaArray<T> arr = this->array();
    arr.print_all();
  }

private:
  AthenaArray<T> data_;
  AthenaArray<T> slice_;
  int idxmap_[ndim][ndim];
  int ndof_;
};

// ----------------------------------------------------------------------------
// rank 3 TensorPointwise, e.g., Christoffel symbols
template<typename T, Symmetries sym, int ndim>
class TensorPointwise<T, sym, ndim, 3> {
public:
  TensorPointwise();
  // the default destructor/copy operators are sufficient
  ~TensorPointwise() = default;
  TensorPointwise(
    TensorPointwise<T, sym, ndim, 3> const &
  ) = default;
  TensorPointwise<T, sym, ndim, 3> & operator=(
    TensorPointwise<T, sym, ndim, 3> const &
  ) = default;

  // functions to allocate/de-allocate the data
  void NewTensorPointwise() {
    data_.NewAthenaArray(ndof_);
  }
  void DeleteTensorPointwise() {
    data_.DeleteAthenaArray();
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  int idxmap(int const a, int const b, int const c) const {
    return idxmap_[a][b][c];
  }
  int ndof() const {
    return ndof_;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b, int const c) {
    return data_(idxmap_[a][b][c]);
  }
  T operator()(int const a, int const b, int const c) const {
    return data_(idxmap_[a][b][c]);
  }

public:
  // debug functionality
  void print_all()
  {
    AthenaArray<T> arr = this->array();
    arr.print_all();
  }

private:
  AthenaArray<T> data_;
  AthenaArray<T> slice_;
  int idxmap_[ndim][ndim][ndim];
  int ndof_;
};

// ----------------------------------------------------------------------------
// rank 4 TensorPointwise, e.g., mixed derivatives of the metric
template<typename T, Symmetries sym, int ndim>
class TensorPointwise<T, sym, ndim, 4> {
public:
  TensorPointwise();
  // the default destructor/copy operators are sufficient
  ~TensorPointwise() = default;
  TensorPointwise(
    TensorPointwise<T, sym, ndim, 4> const &
  ) = default;
  TensorPointwise<T, sym, ndim, 4> & operator=(
    TensorPointwise<T, sym, ndim, 4> const &
  ) = default;

  // functions to allocate/de-allocate the data
  void NewTensorPointwise() {
    data_.NewAthenaArray(ndof_);
  }

  void DeleteTensorPointwise() {
    data_.DeleteAthenaArray();
  }

  // get a reference to the array
  AthenaArray<T> & array() {
    return data_;
  }
  AthenaArray<T> const & array() const {
    return data_;
  }

  // index map
  int idxmap(int const a, int const b, int const c, int const d) const {
    return idxmap_[a][b][c][d];
  }
  int ndof() const {
    return ndof_;
  }

  // fill a tensor field with a constant value
  void Fill(T const val) { data_.Fill(val); }
  void ZeroClear() { data_.ZeroClear(); }

  // check values
  bool is_finite() { return data_.is_finite(); }
  bool is_nan() { return data_.is_nan(); }
  bool is_inf() { return data_.is_inf(); }

  // operators to access the data
  T & operator()(int const a, int const b, int const c, int const d) {
    return data_(idxmap_[a][b][c][d]);
  }

  T operator()(int const a, int const b, int const c, int const d) const {
    return data_(idxmap_[a][b][c][d]);
  }

public:
  // debug functionality
  void print_all()
  {
    AthenaArray<T> arr = this->array();
    arr.print_all();
  }

private:
  AthenaArray<T> data_;
  AthenaArray<T> slice_;
  int idxmap_[ndim][ndim][ndim][ndim];
  int ndof_;
};

// implementation details -----------------------------------------------------

template<typename T, Symmetries sym, int ndim>
TensorPointwise<T, sym, ndim, 2>::TensorPointwise() {
  switch(sym) {
    case Symmetries::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case Symmetries::SYM2:
    case Symmetries::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
    default:
      assert(false); // you shouldn't be here
      abort();
  }
}

template<typename T, Symmetries sym, int ndim>
TensorPointwise<T, sym, ndim, 3>::TensorPointwise() {
  switch(sym) {
    case Symmetries::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
      }
      break;
    case Symmetries::SYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = b; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[a][c][b] = idxmap_[a][b][c];
      }
      break;
    case Symmetries::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c) {
        idxmap_[a][b][c] = ndof_++;
        idxmap_[b][a][c] = idxmap_[a][b][c];
      }
      break;
    default:
      assert(false); // you shouldn't be here
      abort();
  }
}

template<typename T, Symmetries sym, int ndim>
TensorPointwise<T, sym, ndim, 4>::TensorPointwise() {
  switch(sym) {
    case Symmetries::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = 0; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
      }
      break;
    case Symmetries::SYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = c; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
      }
      break;
    case Symmetries::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = 0; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
      }
      break;
    case Symmetries::SYM22:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b)
      for(int c = 0; c < ndim; ++c)
      for(int d = c; d < ndim; ++d) {
        idxmap_[a][b][c][d] = ndof_++;
        idxmap_[b][a][c][d] = idxmap_[a][b][c][d];
        idxmap_[a][b][d][c] = idxmap_[a][b][c][d];
        idxmap_[b][a][d][c] = idxmap_[a][b][c][d];
      }
      break;
    default:
      assert(false); // you shouldn't be here
      abort();
  }
}

// make some further types for convenience ====================================

typedef TensorPointwise<Real, Symmetries::SYM2, 4, 2> type_sym2_4;
typedef TensorPointwise<Real, Symmetries::SYM2, 3, 2> type_sym2_3;
typedef TensorPointwise<Real, Symmetries::NONE, 4, 1> type_vec_4;
typedef TensorPointwise<Real, Symmetries::NONE, 3, 1> type_vec_3;
typedef TensorPointwise<Real, Symmetries::NONE, 4, 0> type_sca_4;
typedef TensorPointwise<Real, Symmetries::NONE, 3, 0> type_sca_3;


// prototypes =================================================================

// Kronecker delta
inline Real delta(int a, int b)
{
  return a == b ? 1.0 : 0.0;
};

// operations =================================================================

template<typename T, int N>
void contract(TensorPointwise<T, Symmetries::SYM2, N, 2> const & mat,
              TensorPointwise<T, Symmetries::NONE, N, 1> const & va,
              TensorPointwise<T, Symmetries::NONE, N, 1> & vb)
{
  for (int a = 0; a < N; ++a)
  {
    vb(a) = 0;
    for (int b = 0; b < N; ++b)
    {
      vb(a) += mat(a,b)*va(b);
    }
  }
}

template<typename T, int N>
void contract(TensorPointwise<T, Symmetries::SYM2, N, 2> const & mat,
              TensorPointwise<T, Symmetries::SYM2, N, 2> const & A,
              TensorPointwise<T, Symmetries::NONE, N, 2> & B)
{
  for (int a = 0; a < N; ++a)
  for (int b = 0; b < N; ++b)
  {
    B(a,b) = 0.0;
    for (int c = 0; c < N; ++c)
    {
      B(a,b) += mat(a,c)*A(c,b);
    }
  }
}

template<typename T, int N>
void contract2(TensorPointwise<T, Symmetries::SYM2, N, 2> const & mat,
               TensorPointwise<T, Symmetries::SYM2, N, 2> const & A,
               TensorPointwise<T, Symmetries::SYM2, N, 2> & B)
{
  for (int a = 0; a < N; ++a)
  for (int b = a; b < N; ++b)
  {
    B(a,b) = 0.0;
    for (int c = 0; c < N; ++c)
    for (int d = 0; d < N; ++d)
    {
      B(a,b) += mat(a,c) * mat(b,d) * A(c,d);
    }
  }
}

template<typename T, int N>
T dot(TensorPointwise<T, Symmetries::NONE, N, 1> const & va,
      TensorPointwise<T, Symmetries::NONE, N, 1> const & vb)
{
  T out(0);
  for (int a = 0; a < N; ++a)
  {
    out += va(a)*vb(a);
  }
  return out;
}

template<typename T, int N>
T dot(TensorPointwise<T, Symmetries::SYM2, N, 2> const & A,
      TensorPointwise<T, Symmetries::SYM2, N, 2> const & B)
{
  T out(0);
  for (int a = 0; a < N; ++a)
  for (int b = 0; b < N; ++b)
  {
    out += A(a,b) * B(a,b);
  }
  return out;
}

template<typename T, int N>
T dot(TensorPointwise<T, Symmetries::SYM2, N, 2> const & met,
      TensorPointwise<T, Symmetries::NONE, N, 1> const & va,
      TensorPointwise<T, Symmetries::NONE, N, 1> const & vb)
{
  T out(0);
  for (int a = 0; a < N; ++a)
  for (int b = 0; b < N; ++b)
  {
    out += met(a,b)*va(a)*vb(b);
  }
  return out;
}

template<typename T, int N>
T dot(TensorPointwise<T, Symmetries::SYM2, N, 2> const & met,
      TensorPointwise<T, Symmetries::SYM2, N, 2> const & A,
      TensorPointwise<T, Symmetries::SYM2, N, 2> const & B)
{
  T out(0);
  for (int a = 0; a < N; ++a)
  for (int b = 0; b < N; ++b)
  for (int c = 0; c < N; ++c)
  for (int d = 0; d < N; ++d)
  {
      out += met(a,c) * met(b,d) * A(c,d) * B(a,b);
  }
  return out;
}

} // namespace utils
} // namespace tensor

#endif // UTILS_TENSOR_HPP_
