#ifndef HDF5_GUARD_HPP
#define HDF5_GUARD_HPP

// hdf5 is not by default thread-safe;
// To work around this we can make a lock inside functions that make use of
// the library. This way we don't have to wrap everything
//
// To use:
// #include "hdf5_guard.hpp"
// Then add the following at the top of whatever function calls hdf5 fcns
//
// HDF5Lock lock;
// Once the function is exit, then the lock is released

// #pragma once

#include <mutex>
#include "hdf5.h"

// One global mutex for all HDF5 calls in this process
inline std::mutex& hdf5_mutex()
{
  static std::mutex m;
  return m;
}

struct HDF5Lock
{
  HDF5Lock()
  {
    hdf5_mutex().lock();
  }
  ~HDF5Lock()
  {
    hdf5_mutex().unlock();
  }
};

inline bool is_hdf5_threadsafe()
{
#if H5_VERSION_GE(1,12,0)
  hbool_t ts = 0;
  herr_t status = H5is_library_threadsafe(&ts);
  return ts > 0;
#else
  int status = H5is_library_threadsafe();
  return status > 0;
#endif
}

#endif  // HDF5_GUARD_HPP