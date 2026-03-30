//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file buffer_utils.cpp
//  \brief namespace containing buffer utilities.

// C headers

// C++ headers
#include <algorithm>  // std::min

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "buffer_utils.hpp"

namespace BufferUtility
{
//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(AthenaArray<T> &src, T *buf, int
//! sn, int en,
//                     int si, int ei, int sj, int ej, int sk, int ek, int
//                     &offset)
//  \brief pack a 4D AthenaArray into a one-dimensional buffer

template <typename T>
void PackData(AthenaArray<T>& src,
              T* buf,
              int sn,
              int en,
              int si,
              int ei,
              int sj,
              int ej,
              int sk,
              int ek,
              int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int n = sn; n <= en; ++n)
  {
    for (int k = sk; k <= ek; k++)
    {
      for (int j = sj; j <= ej; j++)
      {
#pragma omp simd
        for (int i = si; i <= ei; i++)
          buf[offset + i - si] = src(n, k, j, i);
        offset += ni;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(AthenaArray<T> &src, T *buf,
//                      int si, int ei, int sj, int ej, int sk, int ek, int
//                      &offset)
//  \brief pack a 3D AthenaArray into a one-dimensional buffer

template <typename T>
void PackData(AthenaArray<T>& src,
              T* buf,
              int si,
              int ei,
              int sj,
              int ej,
              int sk,
              int ek,
              int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; k++)
  {
    for (int j = sj; j <= ej; j++)
    {
#pragma omp simd
      for (int i = si; i <= ei; i++)
        buf[offset + i - si] = src(k, j, i);
      offset += ni;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(T *buf, AthenaArray<T> &dst, int
//! sn, int en,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 4D AthenaArray

template <typename T>
void UnpackData(T* buf,
                AthenaArray<T>& dst,
                int sn,
                int en,
                int si,
                int ei,
                int sj,
                int ej,
                int sk,
                int ek,
                int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int n = sn; n <= en; ++n)
  {
    for (int k = sk; k <= ek; ++k)
    {
      for (int j = sj; j <= ej; ++j)
      {
#pragma omp simd
        for (int i = si; i <= ei; ++i)
          dst(n, k, j, i) = buf[offset + i - si];
        offset += ni;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(T *buf, AthenaArray<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray

template <typename T>
void UnpackData(T* buf,
                AthenaArray<T>& dst,
                int si,
                int ei,
                int sj,
                int ej,
                int sk,
                int ek,
                int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = sj; j <= ej; ++j)
    {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) = buf[offset + i - si];
      offset += ni;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackDataAdd(T *buf, AthenaArray<T> &dst,
//! int sn, int en,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 4D AthenaArray additively

template <typename T>
void UnpackDataAdd(T* buf,
                   AthenaArray<T>& dst,
                   int sn,
                   int en,
                   int si,
                   int ei,
                   int sj,
                   int ej,
                   int sk,
                   int ek,
                   int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int n = sn; n <= en; ++n)
  {
    for (int k = sk; k <= ek; ++k)
    {
      for (int j = sj; j <= ej; ++j)
      {
#pragma omp simd
        for (int i = si; i <= ei; ++i)
          dst(n, k, j, i) += buf[offset + i - si];
        offset += ni;
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackDataAdd(T *buf, AthenaArray<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray additively

template <typename T>
void UnpackDataAdd(T* buf,
                   AthenaArray<T>& dst,
                   int si,
                   int ei,
                   int sj,
                   int ej,
                   int sk,
                   int ek,
                   int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = sj; j <= ej; ++j)
    {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) += buf[offset + i - si];
      offset += ni;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackDataMin(T *buf, AthenaArray<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray via
//  element-wise min

template <typename T>
void UnpackDataMin(T* buf,
                   AthenaArray<T>& dst,
                   int si,
                   int ei,
                   int sj,
                   int ej,
                   int sk,
                   int ek,
                   int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = sj; j <= ej; ++j)
    {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) = std::min(dst(k, j, i), buf[offset + i - si]);
      offset += ni;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackDataMax(T *buf, AthenaArray<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray via
//  element-wise max

template <typename T>
void UnpackDataMax(T* buf,
                   AthenaArray<T>& dst,
                   int si,
                   int ei,
                   int sj,
                   int ej,
                   int sk,
                   int ek,
                   int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = sj; j <= ej; ++j)
    {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) = std::max(dst(k, j, i), buf[offset + i - si]);
      offset += ni;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackDataAvg(T *buf, AthenaArray<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int
//                        &offset)
//  \brief unpack a one-dimensional buffer into a 3D AthenaArray via
//  element-wise average: dst = 0.5*(dst + buf)

template <typename T>
void UnpackDataAvg(T* buf,
                   AthenaArray<T>& dst,
                   int si,
                   int ei,
                   int sj,
                   int ej,
                   int sk,
                   int ek,
                   int& offset)
{
  // Buffer index computed from loop variables to avoid loop-carried
  // dependency.
  const int ni = ei - si + 1;
  for (int k = sk; k <= ek; ++k)
  {
    for (int j = sj; j <= ej; ++j)
    {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) =
          static_cast<T>(0.5) * (dst(k, j, i) + buf[offset + i - si]);
      offset += ni;
    }
  }
}

// provide explicit instantiation definitions (C++03) to allow the template
// definitions to exist outside of header file (non-inline), but still provide
// the requisite instances for other TUs during linking time (~13x files
// include "buffer_utils.hpp")

// 13x files include buffer_utils.hpp
template void UnpackData<Real>(Real*,
                               AthenaArray<Real>&,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int&);
template void UnpackData<Real>(Real*,
                               AthenaArray<Real>&,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int,
                               int&);

template void UnpackDataAdd<Real>(Real*,
                                  AthenaArray<Real>&,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int&);
template void UnpackDataAdd<Real>(Real*,
                                  AthenaArray<Real>&,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int&);

template void PackData<Real>(AthenaArray<Real>&,
                             Real*,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int,
                             int&);
template void
PackData<Real>(AthenaArray<Real>&, Real*, int, int, int, int, int, int, int&);

template void UnpackDataMin<Real>(Real*,
                                  AthenaArray<Real>&,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int&);

template void UnpackDataMax<Real>(Real*,
                                  AthenaArray<Real>&,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int&);

template void UnpackDataAvg<Real>(Real*,
                                  AthenaArray<Real>&,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int,
                                  int&);

}  // end namespace BufferUtility
