//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file task_list.cpp
//  \brief functions for TaskList base class


// C headers

// C++ headers
//#include <vector> // formerly needed for vector of MeshBlock ptrs in DoTaskListOneStage
#include <chrono>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../mesh/mesh.hpp"
#include "task_list.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#define DEBUG_TASK_LIST 1

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus TaskList::DoAllAvailableTasks
//  \brief do all tasks that can be done (are not waiting for a dependency to be
//  cleared) in this TaskList, return status.

TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts) {
  int skip = 0;
  TaskStatus ret;
  if (ts.num_tasks_left == 0) return TaskListStatus::nothing_to_do;

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    Task &taski = task_list_[i];

    if (ts.finished_tasks.IsUnfinished(taski.task_id)) { // task not done
      // check if dependency clear
      if (ts.finished_tasks.CheckDependencies(taski.dependency)) {
        if (taski.lb_time) pmb->StartTimeMeasurement();
        ret = (this->*task_list_[i].TaskFunc)(pmb, stage);
        if (taski.lb_time) pmb->StopTimeMeasurement();
        if (ret != TaskStatus::fail) { // success
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip == 0) ts.indx_first_task++;
          if (ts.num_tasks_left == 0) return TaskListStatus::complete;
          if (ret == TaskStatus::next) continue;
          return TaskListStatus::running;
        }
      }
      skip++; // increment number of tasks processed

    } else if (skip == 0) { // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
    }
  }
  // there are still tasks to do but nothing can be done now
  return TaskListStatus::stuck;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//  \brief completes all tasks in this list, will not return until all are tasks done

void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage) {
  int nthreads = pmesh->GetNumMeshThreads();
  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);

  // construct the MeshBlock array on this process
  MeshBlock **pmb_array = new MeshBlock*[nmb];
  MeshBlock *pmb = pmesh->pblock;
  for (int n=0; n < nmb; ++n) {
    pmb_array[n] = pmb;
    pmb = pmb->next;
  }

  // clear the task states, startup the integrator and initialize mpi calls
#pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
  for (int i=0; i<nmb; ++i) {
    pmb_array[i]->tasks.Reset(ntasks);
    StartupTaskList(pmb_array[i], stage);
  }

  // DEBUG ONLY
  #if DEBUG_TASK_LIST
  auto start = std::chrono::steady_clock::now();
  #endif

  int nmb_left = nmb;
  // cycle through all MeshBlocks and perform all tasks possible
  while (nmb_left > 0) {
    // KNOWN ISSUE: Workaround for unknown OpenMP race condition. See #183 on GitHub.
#pragma omp parallel for reduction(- : nmb_left) num_threads(nthreads) schedule(dynamic,1)
    for (int i=0; i<nmb; ++i) {
      TaskListStatus status = DoAllAvailableTasks(pmb_array[i], stage, pmb_array[i]->tasks);
      /*if (DoAllAvailableTasks(pmb_array[i],stage,pmb_array[i]->tasks)
          == TaskListStatus::complete) {
        nmb_left--;
      }*/
      if (status == TaskListStatus::complete) {
        nmb_left--;
        #if DEBUG_TASK_LIST
        start = std::chrono::steady_clock::now();
        #endif
      }
      #if DEBUG_TASK_LIST
      else if (status == TaskListStatus::stuck) {
        auto time_now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = time_now - start;
        if (elapsed.count() > 600.0) {
          std::cout << "The task list has been stuck for longer than 10 minutes.\n";
          std::cout << "  MeshBlock: " << i << "\n";
          std::cout << "  " << pmb_array[i]->tasks.num_tasks_left << " tasks remaining.\n";
          std::cout << "  Unfinished tasks: \n";
          for (int m = pmb_array[i]->tasks.indx_first_task; m < ntasks; m++) {
            if (pmb_array[i]->tasks.finished_tasks.IsUnfinished(task_list_[m].task_id)) {
              std::uint64_t id = task_list_[m].task_id.bitfld_[0];
              int k = 1;
              while (id > 1) {
                id = id >> 1;
                k++;
              }
              std::cout << "    TaskID: " << k << "\n";
            }
          }
          std::cout << "Terminating...\n";
          std::exit(EXIT_FAILURE);
        }
      }
      #endif
    }
  }
  delete [] pmb_array;
  return;
}
