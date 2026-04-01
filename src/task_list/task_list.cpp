//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file task_list.cpp
//  \brief functions for TaskList base class
//
//  DoTaskListOneStage (GTS) uses a barrier-free work-stealing scheduler: a
//  single persistent OpenMP parallel region with per-block atomic try-locks
//  and an atomic round-robin index for block selection.  This eliminates the
//  O(nmb * niter) barrier overhead of the original "#pragma omp parallel for"
//  + while(nmb_left) design.

// C headers

// C++ headers
#include <atomic>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "task_list.hpp"

// C++ aux. headers
#ifdef DBG_TASKLIST_HANG
#include <chrono>
#include <cstdio>   // std::printf
#include <cstdlib>  // std::exit (non-MPI fallback)
#endif

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Cache-line-padded atomic flag for per-block try-locks.
// Each flag occupies its own cache line to avoid false sharing.
namespace
{
struct alignas(64) PaddedFlag
{
  std::atomic<bool> locked{ false };
};

#ifdef DBG_TASKLIST_HANG
constexpr double kHangTimeoutSeconds = 45.0;
#endif
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus TaskList::DoAllAvailableTasks
//  \brief do all tasks that can be done (are not waiting for a dependency to
//  be cleared) in this TaskList, return status.

TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock* pmb,
                                             int stage,
                                             TaskStates& ts)
{
  int skip = 0;
  TaskStatus ret;
  if (ts.num_tasks_left == 0)
    return TaskListStatus::nothing_to_do;

  for (int i = ts.indx_first_task; i < ntasks; i++)
  {
    Task& taski = task_list_[i];

    if (ts.finished_tasks.IsUnfinished(taski.task_id))
    {  // task not done
      // check if dependency clear
      if (ts.finished_tasks.CheckDependencies(taski.dependency))
      {
        if (taski.lb_time)
          pmb->StartTimeMeasurement();
        ret = (this->*task_list_[i].TaskFunc)(pmb, stage);
        if (taski.lb_time)
          pmb->StopTimeMeasurement();
        if (ret != TaskStatus::fail)
        {  // success
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip == 0)
            ts.indx_first_task++;
          if (ts.num_tasks_left == 0)
            return TaskListStatus::complete;
          if (ret == TaskStatus::next)
            continue;
          return TaskListStatus::running;
        }
      }
      skip++;  // increment number of tasks processed
    }
    else if (skip == 0)
    {  // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
    }
  }
  // there are still tasks to do but nothing can be done now
  return TaskListStatus::stuck;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//  \brief completes all tasks in this list, will not return until all are
//  tasks done
//
//  Barrier-free cross-block work-stealing scheduler.  A single persistent
//  OpenMP parallel region is used with NO implicit barriers between outer-loop
//  iterations.  Threads independently scan for MeshBlocks that need work,
//  acquiring a per-block atomic try-lock to ensure the single-thread-per-block
//  invariant.  This eliminates the O(nmb * niter) barrier overhead of the
//  original "#pragma omp parallel for" + while(nmb_left) design.

void TaskList::DoTaskListOneStage(Mesh* pmesh, int stage)
{
  int nthreads = pmesh->GetNumMeshThreads();
  (void)nthreads;
  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);

  if (nmb == 0)
    return;

  // Use the cached MeshBlock pointer array from Mesh (rebuilt on AMR regrid).
  // This eliminates the per-stage linked-list walk and heap allocation.
  const std::vector<MeshBlock*>& pmb_vec = pmesh->GetMeshBlocksCached();
  MeshBlock** pmb_array = const_cast<MeshBlock**>(pmb_vec.data());

  // clear the task states, startup the integrator and initialize mpi calls
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
  for (int i = 0; i < nmb; ++i)
  {
    pmb_array[i]->tasks.Reset(ntasks);
    StartupTaskList(pmb_array[i], stage);
  }

  // --- Barrier-free work-stealing loop ---

  // Per-block completion flags (plain char, only written under lock).
  // NOTE: std::vector<bool> is a packed bitfield; concurrent writes to
  // different indices sharing the same word would be a data race.
  // std::vector<char> avoids this by giving each flag its own byte.
  std::vector<char> completed(nmb, 0);

  // Per-block try-lock: false = unlocked, true = locked.
  std::vector<PaddedFlag> block_lock(nmb);

  // Global counters shared by all threads.
  std::atomic<int> nmb_left(nmb);
  std::atomic<int> next_idx(0);

#ifdef DBG_TASKLIST_HANG
  // Generation counter: bumped by any thread that completes a block.
  // All threads observe this to detect global stalls.
  std::atomic<int> progress_gen(0);
#endif

#pragma omp parallel num_threads(nthreads)
  {
#ifdef DBG_TASKLIST_HANG
    int local_gen    = 0;
    auto local_clock = std::chrono::steady_clock::now();
#endif

    while (nmb_left.load(std::memory_order_relaxed) > 0)
    {
      // Scan up to nmb blocks looking for one we can lock and advance.
      for (int attempt = 0; attempt < nmb; ++attempt)
      {
        // Round-robin index selection via shared atomic counter.
        int idx = next_idx.fetch_add(1, std::memory_order_relaxed) % nmb;

        if (completed[idx])
          continue;

        // Try to acquire the per-block lock (non-blocking).
        bool expected = false;
        if (!block_lock[idx].locked.compare_exchange_strong(
              expected, true, std::memory_order_acquire))
        {
          continue;  // another thread is working on this block
        }

        // Double-check completion under lock (another thread may have
        // completed it between our check above and lock acquisition).
        if (completed[idx])
        {
          block_lock[idx].locked.store(false, std::memory_order_release);
          continue;
        }

        // Execute all available tasks on this MeshBlock.
        TaskListStatus status =
          DoAllAvailableTasks(pmb_array[idx], stage, pmb_array[idx]->tasks);

        if (status == TaskListStatus::complete)
        {
          completed[idx] = true;
          nmb_left.fetch_sub(1, std::memory_order_relaxed);
#ifdef DBG_TASKLIST_HANG
          progress_gen.fetch_add(1, std::memory_order_relaxed);
#endif
        }

        // Release the per-block lock.
        block_lock[idx].locked.store(false, std::memory_order_release);
        break;  // go back to the outer while-loop to re-check nmb_left
      }

#ifdef DBG_TASKLIST_HANG
      // Detect hangs: if no block has completed for > kHangTimeoutSeconds,
      // dump diagnostic and abort.  We track a "generation" counter that any
      // thread bumps on block completion.  If the counter hasn't changed, the
      // wall-clock timer accumulates; completing any block resets it.
      int current_gen = progress_gen.load(std::memory_order_relaxed);
      if (current_gen != local_gen)
      {
        local_gen   = current_gen;
        local_clock = std::chrono::steady_clock::now();
      }
      else
      {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - local_clock;
        if (elapsed.count() > kHangTimeoutSeconds)
        {
// Only one thread should print the diagnostic.
#pragma omp critical
          {
            DumpHangDiagnostic(pmesh, stage, pmb_array, completed, nmb);
          }
        }
      }
#endif  // DBG_TASKLIST_HANG
    }
  }  // end omp parallel

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void TaskList::DumpHangDiagnostic
//  \brief Print enriched diagnostic when the task list stall detector fires,
//  then abort.  Called from inside #pragma omp critical so only one thread
//  prints at a time.

#ifdef DBG_TASKLIST_HANG
void TaskList::DumpHangDiagnostic(Mesh* pmesh,
                                  int stage,
                                  MeshBlock** pmb_array,
                                  const std::vector<char>& completed,
                                  int nmb) const
{
  std::printf(
    "[HANG] Task list stalled for > %.0f s  rank=%d  cycle=%d  time=%.15e"
    "  stage=%d\n",
    kHangTimeoutSeconds,
    Globals::my_rank,
    pmesh->ncycle,
    pmesh->time,
    stage);

  for (int i = 0; i < nmb; ++i)
  {
    if (completed[i])
      continue;

    MeshBlock* pmb = pmb_array[i];
    std::printf("  MeshBlock gid=%d lid=%d level=%d  tasks_left=%d\n",
                pmb->gid,
                pmb->lid,
                pmb->loc.level,
                pmb->tasks.num_tasks_left);

    for (int m = pmb->tasks.indx_first_task; m < ntasks; ++m)
    {
      if (!pmb->tasks.finished_tasks.IsUnfinished(task_list_[m].task_id))
        continue;

      // Compute 1-based TaskID ordinal from the single-bit position.
      std::uint64_t raw_id = task_list_[m].task_id.bitfld_[0];
      int id_ordinal       = 1;
      {
        std::uint64_t tmp = raw_id;
        while (tmp > 1)
        {
          tmp >>= 1;
          id_ordinal++;
        }
      }

      // Determine which dependency bits are still unfinished.
      std::uint64_t dep_bits   = task_list_[m].dependency.bitfld_[0];
      std::uint64_t done_bits  = pmb->tasks.finished_tasks.bitfld_[0];
      std::uint64_t blocked_by = dep_bits & ~done_bits;

      // List the individual blocking task ordinals.
      if (blocked_by != 0)
      {
        std::printf("    TaskID %d  blocked_by=[", id_ordinal);
        bool first              = true;
        std::uint64_t remaining = blocked_by;
        while (remaining != 0)
        {
          // Isolate lowest set bit.
          std::uint64_t bit = remaining & (~remaining + 1);
          remaining &= ~bit;

          int blocker_ordinal = 1;
          std::uint64_t tmp2  = bit;
          while (tmp2 > 1)
          {
            tmp2 >>= 1;
            blocker_ordinal++;
          }
          std::printf("%s%d", first ? "" : ",", blocker_ordinal);
          first = false;
        }
        std::printf("]\n");
      }
      else
      {
        std::printf(
          "    TaskID %d  blocked_by=[] (deps met, task returning fail)\n",
          id_ordinal);
      }
    }
  }

  std::printf("[HANG] Aborting.\n");
  std::fflush(stdout);

#ifdef MPI_PARALLEL
  MPI_Abort(MPI_COMM_WORLD, 1);
#else
  std::exit(EXIT_FAILURE);
#endif
}
#endif  // DBG_TASKLIST_HANG
