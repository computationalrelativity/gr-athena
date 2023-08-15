// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/z4c_macro.hpp"
#include "../z4c/wave_extract.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"


//----------------------------------------------------------------------------------------
//  Z4cRBCTaskList constructor

Z4cRBCTaskList::Z4cRBCTaskList(ParameterInput *pin, Mesh *pm){

  nstages = 1;

  // Now assemble list of tasks for each stage of wave integrator
  {
    using namespace Z4cRBCTaskNames;

    AddTask(SEND_RBC, NONE);
    AddTask(RECV_RBC, NONE);

    AddTask(SETB_RBC, RECV_RBC);

    // SMR should be dealt with in this function too for consistency
    if (pm->multilevel) { // SMR or AMR
      AddTask(PROL_RBC, (SEND_RBC|SETB_RBC));
      AddTask(NOP, PROL_RBC);
    } else {
      AddTask(NOP, SETB_RBC);
    }

    // have data on new grid, can do various calculations here...


    // should depend on last task
    AddTask(CLEAR_RBCBND, NOP);
  } // end of using namespace block

}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void Z4cRBCTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace Z4cRBCTaskNames; // NOLINT (build/namespace)

    if (id == NOP) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::Nop);
      task_list_[ntasks].lb_time = false;
    } else if (id == SEND_RBC) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::SendRBC);
      task_list_[ntasks].lb_time = true;
    } else if (id == RECV_RBC) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::ReceiveRBC);
      task_list_[ntasks].lb_time = false;
    } else if (id == SETB_RBC) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::SetBoundariesRBC);
      task_list_[ntasks].lb_time = true;
    } else if (id == PROL_RBC) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::ProlongRBC);
      task_list_[ntasks].lb_time = true;
    } else if (id == CLEAR_RBCBND) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cRBCTaskList::ClearAllRBCBoundary);
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


void Z4cRBCTaskList::StartupTaskList(MeshBlock *pmb, int stage)
{
  // Here we should start receiving specifically auxiliary comms
  pmb->pbval->StartReceivingRBC(BoundaryCommSubset::all);
  return;
}

TaskStatus Z4cRBCTaskList::Nop(MeshBlock *pmb, int stage)
{
  // std::cout << "in Nop" << std::endl;
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to calculate the RHS

TaskStatus Z4cRBCTaskList::SendRBC(MeshBlock *pmb, int stage) {
  // std::cout << "in SendDer" << std::endl;
#if defined(Z4C_CX_ENABLED)
  pmb->pz4c->rbvar.SendBoundaryBuffersFullRestriction();
#endif
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus Z4cRBCTaskList::ReceiveRBC(MeshBlock *pmb, int stage) {
  bool ret;

#if defined(Z4C_CX_ENABLED)
  ret = pmb->pz4c->rbvar.ReceiveBoundaryBuffers();

  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
#endif

}

// Sets boundary buffer data
TaskStatus Z4cRBCTaskList::SetBoundariesRBC(MeshBlock *pmb, int stage) {
#if defined(Z4C_CX_ENABLED)
  pmb->pz4c->rbvar.SetBoundaries();
#endif
  return TaskStatus::success;
}

TaskStatus Z4cRBCTaskList::ProlongRBC(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  // this is post-integ loop
  Real t_end_stage = pmb->pmy_mesh->time;
  pbval->ProlongateBoundaries(t_end_stage, 0);

  return TaskStatus::success;
}

TaskStatus Z4cRBCTaskList::ClearAllRBCBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundaryRBC(BoundaryCommSubset::all);
  return TaskStatus::success;
}
