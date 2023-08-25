//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file time_integrator.cpp
//! \brief derived class for time integrator task list. Can create task lists for one
//! of many different time integrators (e.g. van Leer, RK2, RK3, etc.)

// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../gravity/gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../scalars/scalars.hpp"
#include "../m1/m1.hpp"
#include "task_list.hpp"

//----------------------------------------------------------------------------------------
//! TimeIntegratorTaskList constructor

TimeIntegratorTaskList::M1TaskList(ParameterInput *pin, Mesh *pm) {

  // Set cfl_number based on user input and time integrator CFL limit
  Real cfl_number = pin->GetReal("time", "cfl_number");
  if (cfl_number > cfl_limit) {
    std::cout << "### Warning in M1IntegratorTaskList constructor" << std::endl
              << "User CFL number " << cfl_number << " must be smaller than " << cfl_limit
              << " in " << pm->ndim << "D simulation" << std::endl << "Setting to limit" << std::endl;
    cfl_number = cfl_limit;
  }
  // Save to Mesh class
  pm->cfl_number = cfl_number;

  // Now assemble list of tasks for each stage of time integrator
  {using namespace M1IntegratorTaskNames; // NOLINT (build/namespace)
    // calculate hydro/field diffusive fluxes
    AddTask(CALC_FV, NONE);
    AddTask(CALC_CL, CALC_FV);
    AddTask(CALC_OP, CALC_CL);
    AddTask(CALC_FLX, CALC_OP|CALC_CL);
    AddTask(GR_SRC, CALC_CL);
    AddTask(CALC_UPDT, CALC_FLX);
  } // end of using namespace block
}

//----------------------------------------------------------------------------------------
//!  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//!  ntask.

void M1IntegratorTaskList::AddTask(const TaskID& id, const TaskID& dep) {
  task_list_[ntasks].task_id = id;
  task_list_[ntasks].dependency = dep;

  using namespace M1IntegratorTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_ALLBND) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&M1IntegratorTaskList::ClearAllBoundary);
    task_list_[ntasks].lb_time = false;
  } else if (id == CALC_FV) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&M1IntegratorTaskList::CalcFiducialVelocity);
    task_list_[ntasks].lb_time = true;
  } else if (id == CALC_CL) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&M1IntegratorTaskList::CalcClosure);
    task_list_[ntasks].lb_time = true;
  } else if (id == CALC_OP) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::CalcOpacities);
    task_list_[ntasks].lb_time = true;
  } else if (id == GR_SRC) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::GRSources);
    task_list_[ntasks].lb_time = true;
  } else if (id == CALC_UPDT) {
    task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&TimeIntegratorTaskList::CalcUpdate);
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

//----------------------------------------------------------------------------------------
//! Functions to end MPI communication

TaskStatus M1IntegratorTaskList::ClearAllBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundarySubset(BoundaryCommSubset::all,
                                  pmb->pbval->bvars_main_int);

  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks

TaskStatus M1IntegratorTaskList::SendM1(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->ubvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus M1IntegratorTaskList::ReceiveM1(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pm1->ubvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus M1IntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();
  return TaskStatus::success;
}

TaskStatus M1IntegratorTaskList::NewBlockTimeStep(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pm1->NewBlockTimeStep();
  return TaskStatus::success;
}

TaskStatus M1IntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Function to Calculate Fiducial Velocity

TaskStatus TimeIntegratorTaskList::CalcFiducialVelocity(MeshBlock *pmb, int stage) {

  if (stage <= nstages) {
    pm1->CalcFiducialVelocity();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Function to calculate Opacities

TaskStatus M1IntegratorTaskList::CalcOpacities(MeshBlock *pmb, int stage) {
  M1 *pm1 = pmb->pm1;

  if (stage <= nstages) {
    pm1->CalcOpacity(pm1->u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Function to calculate Closure

TaskStatus M1IntegratorTaskList::CalcClosure(MeshBlock *pmb, int stage) {
  M1 *pm1 = pmb->pm1;

  if (stage <= nstages) {
    pm1->CalcClosure(pm1->u,  pm1->u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to calculates fluxes

TaskStatus M1IntegratorTaskList::CalcFluxes(MeshBlock *pmb, int stage) {
  M1 *pm1 = pmb->pm1;

  if (stage <= nstages) {
    pm1->CalcFluxes(pm1->u,  pm1->u_rhs);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Function to add GR sources

TaskStatus TimeIntegratorTaskList::GRSources(MeshBlock *pmb, int stage) {

  M1 *pm1 = pmb->pm1;
  if (stage <= nstages) {
    pm1->GRSources(pm1->NewBlockTimeStep(), pm1->u, pm1->u_rhs);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to integrate M1 variables

TaskStatus TimeIntegratorTaskList::CalcUpdate(MeshBlock *pmb, int stage) {
  M1 *pm1 = pmb->pm1;

  if (stage <= nstages) {
    pm1->CalcUpdate(1, pm1->u, pm1->u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus TimeIntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();
  return TaskStatus::success;
}


TaskStatus TimeIntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}