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
//  Z4cAuxTaskList constructor

Z4cAuxTaskList::Z4cAuxTaskList(ParameterInput *pin, Mesh *pm){

  nstages = 1;

  // task list trigger logic --------------------------------------------------
  TaskListTriggers.wave_extraction.to_update = false;
  TaskListTriggers.wave_extraction.dt = pin->GetOrAddReal("z4c",
                                                          "dt_wave_extraction",
                                                          1.0);
  if (pin->GetOrAddInteger("z4c", "nrad_wave_extraction", 0) == 0)
  {
    TaskListTriggers.wave_extraction.dt = 0.0;
    TaskListTriggers.wave_extraction.next_time = 0.0;
    TaskListTriggers.wave_extraction.to_update = false;
  }
  else
  {
    // When initializing at restart, this procedure ensures to restart
    // extraction from right time
    int nwavecycles = static_cast<int>(
      pm->time/TaskListTriggers.wave_extraction.dt
    );
    TaskListTriggers.wave_extraction.next_time = (nwavecycles + 1)*
        TaskListTriggers.wave_extraction.dt;
  }
  // --------------------------------------------------------------------------

  // Now assemble list of tasks for each stage of wave integrator
  {
    using namespace Z4cAuxTaskNames;

    AddTask(SEND_AUX, NONE);
    AddTask(RECV_AUX, NONE);

    AddTask(SETB_AUX, RECV_AUX);

    // SMR should be dealt with in this function too for consistency
    if (pm->multilevel) { // SMR or AMR
      AddTask(PROL_AUX, (SEND_AUX|SETB_AUX));
      AddTask(PHY_BVAL, PROL_AUX);
    } else {
      AddTask(PHY_BVAL, SETB_AUX);
    }

    AddTask(NOP, PHY_BVAL);

    // have data on new grid, can do various calculations here...
    AddTask(WEYL_DECOMP, NOP);


    // should depend on last task
    AddTask(CLEAR_AUXBND, WEYL_DECOMP);
  } // end of using namespace block

}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void Z4cAuxTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace Z4cAuxTaskNames; // NOLINT (build/namespace)

    if (id == NOP) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::Nop);
      task_list_[ntasks].lb_time = false;
    } else if (id == SEND_AUX) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::SendAux);
      task_list_[ntasks].lb_time = true;
    } else if (id == RECV_AUX) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::ReceiveAux);
      task_list_[ntasks].lb_time = false;
    } else if (id == SETB_AUX) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::SetBoundariesAux);
      task_list_[ntasks].lb_time = true;
    } else if (id == PROL_AUX) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::ProlongAux);
      task_list_[ntasks].lb_time = true;
    } else if (id == PHY_BVAL) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::PhysicalBoundary);
      task_list_[ntasks].lb_time = true;
    } else if (id == WEYL_DECOMP) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::WeylDecompose);
      task_list_[ntasks].lb_time = true;
    } else if (id == CLEAR_AUXBND) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cAuxTaskList::ClearAllAuxBoundary);
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


void Z4cAuxTaskList::StartupTaskList(MeshBlock *pmb, int stage)
{
  // Here we should start receiving specifically auxiliary comms
  pmb->pbval->StartReceivingAux(BoundaryCommSubset::all);
  return;
}

TaskStatus Z4cAuxTaskList::Nop(MeshBlock *pmb, int stage)
{
  // std::cout << "in Nop" << std::endl;
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to calculate the RHS

TaskStatus Z4cAuxTaskList::SendAux(MeshBlock *pmb, int stage) {
  // std::cout << "in SendDer" << std::endl;
  pmb->pz4c->abvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus Z4cAuxTaskList::ReceiveAux(MeshBlock *pmb, int stage) {
  bool ret;

  ret = pmb->pz4c->abvar.ReceiveBoundaryBuffers();

  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

// Sets boundary buffer data
TaskStatus Z4cAuxTaskList::SetBoundariesAux(MeshBlock *pmb, int stage) {
  pmb->pz4c->abvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus Z4cAuxTaskList::ProlongAux(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  // this is post-integ loop
  Real t_end_stage = pmb->pmy_mesh->time;

  // BD: TODO: this is doing much more work than is required
  // We don't need to apply BC on ubvar_cx again
  // We do however need prol. for vars in e.g. pmr->pvars_cx_
  pbval->ProlongateBoundaries(t_end_stage, 0);

  // FCN_CC_CX_VC(
  //   pbval->ProlongateBoundaries,
  //   pbval->ProlongateCellCenteredXBoundaries,
  //   pbval->ProlongateVertexCenteredBoundaries
  // )(t_end_stage, 0);

  return TaskStatus::success;
}

TaskStatus Z4cAuxTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  if (stage <= nstages) {

  // this is post-integ loop
  Real t_end_stage = pmb->pmy_mesh->time;
  pbval->ApplyPhysicalBoundariesAux(t_end_stage, 0);

  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus Z4cAuxTaskList::ClearAllAuxBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundaryAux(BoundaryCommSubset::all);
  return TaskStatus::success;
}


TaskStatus Z4cAuxTaskList::WeylDecompose(MeshBlock *pmb, int stage)
{
  Mesh *pm = pmb->pmy_mesh;

  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction)) {
    // std::cout << "exec. Z4cAuxTaskList WeylDecompose" << std::endl;
    AthenaArray<Real> u_R;
    AthenaArray<Real> u_I;
    u_R.InitWithShallowSlice(pmb->pz4c->storage.weyl, Z4c::I_WEY_rpsi4, 1);
    u_I.InitWithShallowSlice(pmb->pz4c->storage.weyl, Z4c::I_WEY_ipsi4, 1);
    for (auto pwextr : pmb->pwave_extr_loc) {
        pwextr->Decompose_multipole(u_R,u_I);
    }
  }

  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// \!fn bool Z4cIntegratorTaskList::CurrentTimeCalculationThreshold(
//   MeshBlock *pmb, aux_NextTimeStep *variable)
// \brief Given current time / ncycles, does a specified 'dt' mean we need
//        to calculate something?
//        Secondary effect is to mutate next_time
bool Z4cAuxTaskList::CurrentTimeCalculationThreshold(
  Mesh *pm, aux_NextTimeStep *variable) {

  // this variable is not dumped / computed
  if (variable->dt == 0 )
    return false;

  Real cur_time = pm->time + pm->dt;
  if ((cur_time - pm->dt >= variable->next_time) ||
      (cur_time >= pm->tlim)) {
#pragma omp atomic write
    variable->to_update = true;
    return true;
  }

  return false;
}

void Z4cAuxTaskList::UpdateTaskListTriggers() {
  // note that for global dt > target output dt
  // next_time will 'lag'; this will need to be corrected if an integrator with dense /
  // interpolating output is used.

  if (TaskListTriggers.wave_extraction.to_update) {
    TaskListTriggers.wave_extraction.next_time += \
      TaskListTriggers.wave_extraction.dt;
    TaskListTriggers.wave_extraction.to_update = false;
  }
}