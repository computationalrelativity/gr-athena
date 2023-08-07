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
#include "../wave/wave.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"


//----------------------------------------------------------------------------------------
//  WaveDerTaskList constructor

WaveDerTaskList::WaveDerTaskList(ParameterInput *pin, Mesh *pm){

  nstages = 1;

  // Now assemble list of tasks for each stage of wave integrator
  {
    using namespace WaveDerivativeTaskNames;

    AddTask(CALC_WAVEDER, NONE);
    AddTask(SEND_DER, CALC_WAVEDER);
    AddTask(RECV_DER, NONE);

    // AddTask(SETB_DER, RECV_DER);
    // AddTask(CLEAR_DERBND, SETB_DER);


    AddTask(SETB_DER, RECV_DER);

    // SMR should be dealt with in this function too for consistency
    if (pm->multilevel) { // SMR or AMR
      AddTask(PROLDER, (SEND_DER|SETB_DER));
      AddTask(NOP, PROLDER);
    } else {
      AddTask(NOP, SETB_DER);
    }

    AddTask(CLEAR_DERBND, NOP);

    /*
    // comm. only test
    AddTask(SEND_DER, NONE);
    AddTask(RECV_DER, NONE);
    AddTask(CLEAR_DERBND, RECV_DER);
    */

  } // end of using namespace block

}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void WaveDerTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace WaveDerivativeTaskNames; // NOLINT (build/namespace)

    if (id == NOP) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::Nop);
      task_list_[ntasks].lb_time = false;
    } else if (id == CALC_WAVEDER) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::CalculateWaveDer);
    } else if (id == SEND_DER) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::SendDer);
      task_list_[ntasks].lb_time = true;
    } else if (id == RECV_DER) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::ReceiveDer);
      task_list_[ntasks].lb_time = false;
    } else if (id == SETB_DER) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::SetBoundariesDer);
      task_list_[ntasks].lb_time = true;
    } else if (id == PROLDER) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::ProlongDer);
      task_list_[ntasks].lb_time = true;
    } else if (id == CLEAR_DERBND) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::ClearAllDerBoundary);
      task_list_[ntasks].lb_time = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTask" << std::endl
          << "Invalid Task is specified" << std::endl;
      ATHENA_ERROR(msg);
    }

    /*
    if (id == CLEAR_ALLBND) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::ClearAllBoundary);
      task_list_[ntasks].lb_time = false;
    } else if (id == CALC_WAVERHS) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::CalculateWaveRHS);
      task_list_[ntasks].lb_time = true;
    } else if (id == INT_WAVE) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::IntegrateWave);
      task_list_[ntasks].lb_time = true;
    } else if (id == SEND_WAVE) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::SendWave);
      task_list_[ntasks].lb_time = true;
    } else if (id == RECV_WAVE) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::ReceiveWave);
      task_list_[ntasks].lb_time = false;
    } else if (id == SETB_WAVE) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::SetBoundariesWave);
      task_list_[ntasks].lb_time = true;
    } else if (id == PROLONG) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::Prolongation);
      task_list_[ntasks].lb_time = true;
    } else if (id == PHY_BVAL) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::PhysicalBoundary);
      task_list_[ntasks].lb_time = true;
    } else if (id == USERWORK) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::UserWork);
      task_list_[ntasks].lb_time = true;
    } else if (id == NEW_DT) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::NewBlockTimeStep);
      task_list_[ntasks].lb_time = true;
    } else if (id == FLAG_AMR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveDerTaskList::CheckRefinement);
      task_list_[ntasks].lb_time = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTask" << std::endl
          << "Invalid Task is specified" << std::endl;
      ATHENA_ERROR(msg);
    }
    */

    ntasks++;
    return;
}


void WaveDerTaskList::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pbval = pmb->pbval;
  // Here we should start receiving specifically derivative comms
  pbval->StartReceivingDer(BoundaryCommSubset::all);
  return;
}

TaskStatus WaveDerTaskList::Nop(MeshBlock *pmb, int stage)
{
  // std::cout << "in Nop" << std::endl;
  // dummy
  // pmb->pbval->ClearBoundary(BoundaryCommSubset::all);

  // pmb->pwave->vbvar_cx.bvar_index = pmb->pbval->bvars.size();
  // pmb->pbval->bvars.push_back(&pmb->pwave->vbvar_cx);
  // pmb->pbval->bvars_der.push_back(&pmb->pwave->vbvar_cx);



  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to calculate the RHS

TaskStatus WaveDerTaskList::CalculateWaveDer(MeshBlock *pmb, int stage)
{
  // std::cout << "in CalculateWaveDer" << std::endl;

  pmb->pwave->PrepareDer(pmb->pwave->u, pmb->pwave->v);

  /*
  BoundaryValues *pbval = pmb->pbval;
  pmb->pwave->WaveRHS(pmb->pwave->u);

  // application of Sommerfeld boundary conditions
  pmb->pwave->WaveBoundaryRHS(pmb->pwave->u);
  */
  return TaskStatus::success;
}


TaskStatus WaveDerTaskList::SendDer(MeshBlock *pmb, int stage) {
  // std::cout << "in SendDer" << std::endl;

  if (WAVE_CC_ENABLED)
  {
    pmb->pwave->vbvar_cc.SendBoundaryBuffers();
  }
  else if (WAVE_VC_ENABLED)
  {
    pmb->pwave->vbvar_vc.SendBoundaryBuffers();
  }
  else if (WAVE_CX_ENABLED)
  {
    pmb->pwave->vbvar_cx.SendBoundaryBuffers();
  }
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus WaveDerTaskList::ReceiveDer(MeshBlock *pmb, int stage) {
  // std::cout << "in ReceiveDer" << std::endl;

  // pmb->pwave->debug("ReceiveDer");

  // std::cout << pmb->pbval->bvars.size() << "" << pmb->pbval->bvars_der.size() << std::endl;
  // std::exit(0);

  bool ret;

  if (WAVE_CC_ENABLED)
  {
    ret = pmb->pwave->vbvar_cc.ReceiveBoundaryBuffers();
  }
  else if (WAVE_VC_ENABLED)
  {
    ret = pmb->pwave->vbvar_vc.ReceiveBoundaryBuffers();
  }
  else if (WAVE_CX_ENABLED)
  {
    ret = pmb->pwave->vbvar_cx.ReceiveBoundaryBuffers();
  }

  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

// Sets boundary buffer data
TaskStatus WaveDerTaskList::SetBoundariesDer(MeshBlock *pmb, int stage) {
  // std::cout << "in SetBoundariesDer" << std::endl;

  // AthenaArray<Real> v_0, v_1;
  // // internal dimension inferred
  // v_0.InitWithShallowSlice(pmb->pwave->v, 0, 1);
  // v_1.InitWithShallowSlice(pmb->pwave->v, 1, 1);
  // v_0.print_all("%.2e", false, false);
  // v_1.print_all("%.2e", false, false);


  if (WAVE_CC_ENABLED)
  {
    pmb->pwave->vbvar_cc.SetBoundaries();
  }
  else if (WAVE_VC_ENABLED)
  {
    pmb->pwave->vbvar_vc.SetBoundaries();
  }
  else if (WAVE_CX_ENABLED)
  {
    pmb->pwave->vbvar_cx.SetBoundaries();
  }

  // Real t_end_stage = 0;
  // Real dt = 0;

  // BoundaryValues *pbval = pmb->pbval;
  // if (WAVE_VC_ENABLED)
  // {
  //   pbval->ApplyPhysicalVertexCenteredBoundaries(t_end_stage, dt);
  // }
  // else if (WAVE_CC_ENABLED)
  // {
  //   pbval->ApplyPhysicalBoundaries(t_end_stage, dt);
  // }
  // else if (WAVE_CX_ENABLED)
  // {
  //   pbval->ApplyPhysicalCellCenteredXBoundaries(t_end_stage, dt);
  // }


  // v_0.print_all("%.2e", false, false);
  // v_1.print_all("%.2e", false, false);

  // std::exit(0);

  // now drop the vbvar
  // pmb->pbval->bvars.pop_back();
  // pmb->pbval->bvars_der.pop_back();

  return TaskStatus::success;
}

TaskStatus WaveDerTaskList::ProlongDer(MeshBlock *pmb, int stage) {
  // std::cout << "in Prolongation" << std::endl;

  BoundaryValues *pbval = pmb->pbval;

  // assume t-indept.
  pbval->ProlongateBoundaries(0, 0);

  return TaskStatus::success;
}

TaskStatus WaveDerTaskList::ClearAllDerBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundaryDer(BoundaryCommSubset::all);

  // now drop the vbvar
  // pmb->pbval->bvars.pop_back();
  // pmb->pbval->bvars_main_int_cx.pop_back();

  return TaskStatus::success;
}
