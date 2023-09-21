//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_task_list.cpp
//  \brief time integrator for the wave equation (based on time_integrator.cpp)

// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../mesh/mesh.hpp"
#include "../dummy/dummy.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"

// BD TODO: Significant code duplication with time_integrator, leave decoupled

//----------------------------------------------------------------------------------------
//  DummyIntegratorTaskList constructor

DummyIntegratorTaskList::DummyIntegratorTaskList(ParameterInput *pin, Mesh *pm)
{

  // Now assemble list of tasks for each stage of wave integrator
  {using namespace DummyIntegratorTaskNames;
    AddTask(USERWORK, NONE);

    if (pm->adaptive) {
      AddTask(FLAG_AMR, USERWORK);
    } else {
      // ...
    }

  } // end of using namespace block

}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void DummyIntegratorTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace DummyIntegratorTaskNames; // NOLINT (build/namespace)
    if (id == USERWORK) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&DummyIntegratorTaskList::UserWork);
      task_list_[ntasks].lb_time = true;
    } else if (id == FLAG_AMR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&DummyIntegratorTaskList::CheckRefinement);
      task_list_[ntasks].lb_time = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTask" << std::endl
          << "Invalid Task is specified" << std::endl;
      ATHENA_ERROR(msg);
    }
    ntasks++;
    return;
}


void DummyIntegratorTaskList::StartupTaskList(MeshBlock *pmb, int stage) {

  BoundaryValues *pbval = pmb->pbval;
  return;
}

TaskStatus DummyIntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->DummyUserWorkInLoop();
  return TaskStatus::success;
}

TaskStatus DummyIntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}
