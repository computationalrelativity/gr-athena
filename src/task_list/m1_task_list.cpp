//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
// #include "../bvals/bvals.hpp"
// #include "../eos/eos.hpp"
// #include "../field/field.hpp"
// #include "../field/field_diffusion/field_diffusion.hpp"
// #include "../gravity/gravity.hpp"
// #include "../hydro/hydro.hpp"
// #include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
// #include "../hydro/srcterms/hydro_srcterms.hpp"
// #include "../mesh/mesh.hpp"
// #include "../orbital_advection/orbital_advection.hpp"
// #include "../parameter_input.hpp"
// #include "../reconstruct/reconstruction.hpp"
// #include "../scalars/scalars.hpp"
#include "../m1/m1.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
// ! TimeIntegratorTaskList constructor

M1IntegratorTaskList::M1IntegratorTaskList(ParameterInput *pin, Mesh *pm)
{
  // Now assemble list of tasks for each stage of time integrator
  { using namespace M1IntegratorTaskNames; // NOLINT (build/namespace)

    AddTask(CALC_FIDU, NONE);

    AddTask(NOP, CALC_FIDU);

    AddTask(NEW_DT, NOP);
    AddTask(CLEAR_ALLBND, NEW_DT);

    /*
    AddTask(CALC_FIDU, NONE);
    AddTask(CALC_CLOSURE, CALC_FIDU);

    AddTask(CALC_OPAC, CALC_CLOSURE);
    AddTask(CALC_GRSRC, CALC_CLOSURE);

    AddTask(CALC_FLUX, CALC_CLOSURE);
    AddTask(SEND_FLUX, CALC_FLUX);
    AddTask(RECV_FLUX, CALC_FLUX);
    AddTask(ADD_FLX_DIV, RECV_FLUX);

    AddTask(CALC_UPDATE, (ADD_FLX_DIV|CALC_OPAC|CALC_GRSRC));

    AddTask(SEND, CALC_UPDATE);
    AddTask(RECV, CALC_UPDATE);

    AddTask(SETB, (RECV|CALC_UPDATE));
    if (pm->multilevel) {
      AddTask(PROLONG, (SEND|SETB));
      AddTask(PHY_BVAL, PROLONG);
    }
    else {
      AddTask(PHY_BVAL, SETB);
    }

    AddTask(USERWORK, PHY_BVAL);

    AddTask(NEW_DT, PHY_BVAL);
    if (pm->adaptive) {
      AddTask(FLAG_AMR, USERWORK);
      AddTask(CLEAR_ALLBND, FLAG_AMR);
    }
    else {
      AddTask(CLEAR_ALLBND, NEW_DT);
    }
    */

  } // end of using namespace block
}

// ----------------------------------------------------------------------------
//! Sets id and dependency for "ntask" member of task_list_ array
//! then iterates value of ntask.
void M1IntegratorTaskList::AddTask(const TaskID& id, const TaskID& dep)
{
  task_list_[ntasks].task_id = id;
  task_list_[ntasks].dependency = dep;

  using namespace M1IntegratorTaskNames; // NOLINT (build/namespace)
  if (id == CLEAR_ALLBND)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::ClearAllBoundary);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == CALC_FIDU)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcFiducialVelocity);
    task_list_[ntasks].lb_time = true;
  }
  /*
  else if (id == CALC_CLOSURE)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcClosure);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == CALC_OPAC)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcOpacity);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == CALC_FLUX)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcFlux);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == SEND_FLUX)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::SendFlux);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == RECV_FLUX)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::ReceiveAndCorrectFlux);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == ADD_FLX_DIV)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::AddFluxDivergence);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == CALC_GRSRC)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcGRSources);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == CALC_UPDATE)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::CalcUpdate);
    task_list_[ntasks].lb_time = true;
  }
  else if (id == SEND)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::Send);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == RECV)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::Receive);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == SETB)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::SetBoundaries);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == PROLONG)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::Prolongation);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == PHY_BVAL)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::PhysicalBoundary);
    task_list_[ntasks].lb_time = false;
  }
  else if (id == USERWORK)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::UserWork);
    task_list_[ntasks].lb_time = false;
  }
  */
  else if (id == NEW_DT)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::NewBlockTimeStep);
    task_list_[ntasks].lb_time = false;
  }
  // else if (id == FLAG_AMR)
  // {
  //   task_list_[ntasks].TaskFunc=
  //     static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
  //     (&M1IntegratorTaskList::CheckRefinement);
  //   task_list_[ntasks].lb_time = false;
  // }
  else if (id == NOP)
  {
    task_list_[ntasks].TaskFunc=
      static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
      (&M1IntegratorTaskList::Nop);
    task_list_[ntasks].lb_time = false;
  }
  else
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in AddTask" << std::endl
        << "Invalid Task is specified" << std::endl;
    ATHENA_ERROR(msg);
  }
  ntasks++;
  return;
}

// ----------------------------------------------------------------------------
//! Initialize the task list
void M1IntegratorTaskList::StartupTaskList(MeshBlock *pmb, int stage)
{
  /*
  if (stage == 1)
  {
    // The auxiliary variable u1 stores the solution at the beginning of the timestep
    pmb->pm1->storage.u1.DeepCopy(pmb->pm1->storage.u);
  }

  // Clear the RHS
  pmb->pm1->storage.u_rhs.ZeroClear();

  // TODO fix this
  pmb->pbval->StartReceivingSubset(BoundaryCommSubset::all, pmb->pbval->bvars_main_int);
  */
  return;
}

// ----------------------------------------------------------------------------
//! Functions to end MPI communication
TaskStatus M1IntegratorTaskList::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  // pmb->pbval->ClearBoundarySubset(BoundaryCommSubset::all,
  //                                 pmb->pbval->bvars_m1_int);

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to Calculate Fiducial Velocity
TaskStatus M1IntegratorTaskList::CalcFiducialVelocity(MeshBlock *pmb, int stage) {
  if (stage <= nstages)
  {
    pmb->pm1->CalcFiducialVelocity();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

/*
// ----------------------------------------------------------------------------
// Function to calculate Closure
TaskStatus M1IntegratorTaskList::CalcClosure(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->CalcClosure(pmb->pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to calculate Opacities
TaskStatus M1IntegratorTaskList::CalcOpacity(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    pmb->pm1->CalcOpacity(dt, pmb->pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to calculates fluxes
TaskStatus M1IntegratorTaskList::CalcFlux(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->CalcFluxes(pmb->pm1->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus M1IntegratorTaskList::SendFlux(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->ubvar.SendFluxCorrection();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to receive fluxes between MeshBlocks
TaskStatus M1IntegratorTaskList::ReceiveAndCorrectFlux(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    if (pmb->pm1->ubvar.ReceiveFluxCorrection()) {
      return TaskStatus::next;
    }
    else {
      return TaskStatus::fail;
    }
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to calculates fluxes
TaskStatus M1IntegratorTaskList::AddFluxDivergence(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->AddFluxDivergence(pmb->pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to add GR sources
TaskStatus M1IntegratorTaskList::CalcGRSources(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->GRSources(pmb->pm1->storage.u, pmb->pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to update the radiation field
TaskStatus M1IntegratorTaskList::CalcUpdate(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    pmb->pm1->CalcUpdate(dt, pmb->pm1->storage.u1, pmb->pm1->storage.u, pmb->pm1->storage.u_rhs);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to send the updated fields to the other MeshBlocks
TaskStatus M1IntegratorTaskList::Send(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->ubvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


// ----------------------------------------------------------------------------
// Function to receive updated fields from the other MeshBlocks
TaskStatus M1IntegratorTaskList::Receive(MeshBlock *pmb, int stage) {
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

// ----------------------------------------------------------------------------
// Function to set boundaries
TaskStatus M1IntegratorTaskList::SetBoundaries(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pm1->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to prolongate data
TaskStatus M1IntegratorTaskList::Prolongation(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;
  if (stage <= nstages) {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    Real t_end_stage = pmb->pmy_mesh->time + dt;
    pbval->ProlongateBoundaries(t_end_stage, dt, pmb->pbval->bvars_m1_int);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to apply the boundary conditions
TaskStatus M1IntegratorTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;
  M1 *pm1 = pmb->pm1;
  Coordinates *pco = pmb->pcoord;
  if (stage <= nstages) {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    Real t_end_stage = pmb->pmy_mesh->time + dt;

    pbval->ApplyPhysicalBoundaries(t_end_stage, dt, pmb->pbval->bvars_m1_int);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to schedule user work
TaskStatus M1IntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();
  return TaskStatus::success;
}
*/

// ----------------------------------------------------------------------------
// Determine the new timestep (used for the time adaptivity)
TaskStatus M1IntegratorTaskList::NewBlockTimeStep(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pm1->NewBlockTimeStep();
  return TaskStatus::success;
}

// // ----------------------------------------------------------------------------
// // Flag cells for MeshBlocks (de)refinement
// TaskStatus M1IntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
//   if (stage != nstages) return TaskStatus::success; // only do on last stage

//   pmb->pmr->CheckRefinementCondition();
//   return TaskStatus::success;
// }

TaskStatus M1IntegratorTaskList::Nop(MeshBlock *pmb, int stage) {
  // debug.

  return TaskStatus::success;
}