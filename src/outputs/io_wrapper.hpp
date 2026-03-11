#ifndef OUTPUTS_IO_WRAPPER_HPP_
#define OUTPUTS_IO_WRAPPER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus Serial Output.

// C headers

// C++ headers
#include <cstdio>

// Athena++ headers
#include "../athena.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
using  IOWrapperFile = MPI_File;
#else
using  IOWrapperFile = FILE*;
#endif

using IOWrapperSizeT = std::uint64_t;

class IOWrapper {
 public:
#ifdef MPI_PARALLEL
  IOWrapper() : fh_(nullptr), comm_(MPI_COMM_WORLD), info_(MPI_INFO_NULL) {}
  void SetCommunicator(MPI_Comm scomm) { comm_=scomm;}
  // Set MPI-IO hints for parallel file access.  Caller retains ownership of
  // the MPI_Info object (IOWrapper makes a duplicate).
  void SetMPIInfo(MPI_Info info) {
    if (info_ != MPI_INFO_NULL) MPI_Info_free(&info_);
    if (info != MPI_INFO_NULL)
      MPI_Info_dup(info, &info_);
    else
      info_ = MPI_INFO_NULL;
  }
#else
  IOWrapper() {fh_=nullptr;}
#endif
  ~IOWrapper() {
#ifdef MPI_PARALLEL
    if (info_ != MPI_INFO_NULL) MPI_Info_free(&info_);
#endif
  }
  // nested type definition of strongly typed/scoped enum in class definition
  enum class FileMode {read, write};

  // wrapper functions for basic I/O tasks
  int Open(const char* fname, FileMode rw);
  std::size_t Read(void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Read_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Read_at_all(void *buf, IOWrapperSizeT size,
                          IOWrapperSizeT count, IOWrapperSizeT offset);
  std::size_t Write(const void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Write_at_all(const void *buf, IOWrapperSizeT size,
                           IOWrapperSizeT count, IOWrapperSizeT offset);
#if defined(DBG_RST_WRITE_PER_MB)
  std::size_t Read_at(void *buf, IOWrapperSizeT size,
                      IOWrapperSizeT count, IOWrapperSizeT offset);
  std::size_t Write_at(const void *buf, IOWrapperSizeT size,
                       IOWrapperSizeT count, IOWrapperSizeT offset);
#endif // DBG_RST_WRITE_PER_MB

  int Close();
  int Seek(IOWrapperSizeT offset);
  IOWrapperSizeT GetPosition();

 private:
  IOWrapperFile fh_;
#ifdef MPI_PARALLEL
  MPI_Comm comm_;
  MPI_Info info_;
#endif
};
#endif // OUTPUTS_IO_WRAPPER_HPP_
