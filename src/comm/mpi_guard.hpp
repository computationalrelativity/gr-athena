#ifndef COMM_MPI_GUARD_HPP_
#define COMM_MPI_GUARD_HPP_
//========================================================================================
// GR-Athena++ MPI thread-safety guard
//========================================================================================
//! \file mpi_guard.hpp
//  \brief Application-level spinlock wrappers for MPI point-to-point calls.
//
//  When DBG_MPI_SPINLOCK is defined, every wrapped MPI call is serialized
//  through a global std::atomic_flag spinlock.  This works around known
//  OpenMPI ob1/BTL race conditions under MPI_THREAD_MULTIPLE where
//  concurrent MPI_Test / MPI_Isend / MPI_Irecv from multiple threads can
//  corrupt internal PML state and cause hangs.
//
//  When DBG_MPI_SPINLOCK is not defined, lock()/unlock() are empty inlines
//  that the compiler eliminates entirely -- zero overhead.
//
//  Usage:  replace  MPI_Isend(...)  with  gra::mpi_guard::MPI_Isend(...)
//          in task-list-phase code (concurrent from work-stealing threads).
//          For a whole block of setup calls, hold a scoped_lock instead.
//
//  Setup/teardown (MPI_Send_init, MPI_Recv_init, MPI_Request_free) is NOT
//  single-threaded: Mesh::Initialize finalizes CommChannels from an
//  `omp parallel for` over meshblocks, so those calls do run concurrently.
//  Under OpenMPI/UCX the first MPI_Send_init to a peer creates the UCX
//  endpoint (mca_pml_ucx_add_proc -> ucp_ep_create), and concurrent wire-up
//  corrupts the unpacked peer address ("addr_version" assertion in
//  ucp_address_unpack).  Callers must take a scoped_lock around them.
//
//  Not wrapped:  MPI_Wait (blocking, would hold the spinlock too long),
//                collectives (MPI_Allgather, MPI_Allreduce, etc.).

#include "../defs.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>

#ifdef DBG_MPI_SPINLOCK
#include <atomic>
#endif

namespace gra
{
namespace mpi_guard
{

#ifdef DBG_MPI_SPINLOCK
extern std::atomic_flag spinlock_;

inline void lock()
{
  while (spinlock_.test_and_set(std::memory_order_acquire))
  {
  }
}

inline void unlock()
{
  spinlock_.clear(std::memory_order_release);
}
#else
inline void lock()
{
}
inline void unlock()
{
}
#endif

// Holds the guard for a block of MPI calls (setup/teardown loops).
// Compiles away entirely when DBG_MPI_SPINLOCK is not defined.
struct scoped_lock
{
  scoped_lock()
  {
    lock();
  }
  ~scoped_lock()
  {
    unlock();
  }
  scoped_lock(const scoped_lock&)            = delete;
  scoped_lock& operator=(const scoped_lock&) = delete;
};

// ---------------------------------------------------------------------------
// Guarded MPI wrappers -- signature-identical to the MPI standard.
// The :: prefix calls the real MPI function from global scope.
// ---------------------------------------------------------------------------

inline int MPI_Isend(const void* buf,
                     int count,
                     MPI_Datatype datatype,
                     int dest,
                     int tag,
                     MPI_Comm comm,
                     MPI_Request* request)
{
  lock();
  int rc = ::MPI_Isend(buf, count, datatype, dest, tag, comm, request);
  unlock();
  return rc;
}

inline int MPI_Irecv(void* buf,
                     int count,
                     MPI_Datatype datatype,
                     int source,
                     int tag,
                     MPI_Comm comm,
                     MPI_Request* request)
{
  lock();
  int rc = ::MPI_Irecv(buf, count, datatype, source, tag, comm, request);
  unlock();
  return rc;
}

inline int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status)
{
  lock();
  int rc = ::MPI_Test(request, flag, status);
  unlock();
  return rc;
}

inline int MPI_Start(MPI_Request* request)
{
  lock();
  int rc = ::MPI_Start(request);
  unlock();
  return rc;
}

}  // namespace mpi_guard
}  // namespace gra

#endif  // MPI_PARALLEL
#endif  // COMM_MPI_GUARD_HPP_
