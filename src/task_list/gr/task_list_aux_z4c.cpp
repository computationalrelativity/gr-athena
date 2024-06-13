// C/C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../parameter_input.hpp"
#include "task_list.hpp"


// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskNames::GeneralRelativity::Aux_Z4c;

Aux_Z4c::Aux_Z4c(ParameterInput *pin, Mesh *pm)
{
  nstages = 1;

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic

  {
    Add(SEND_AUX, NONE, &Aux_Z4c::SendAux);
    Add(RECV_AUX, NONE, &Aux_Z4c::ReceiveAux);

    Add(SETB_AUX, RECV_AUX, &Aux_Z4c::SetBoundariesAux);

    if (multilevel)
    {
      Add(PROL_AUX, (SEND_AUX | SETB_AUX), &Aux_Z4c::ProlongAux);
      Add(PHY_BVAL,  PROL_AUX, &Aux_Z4c::PhysicalBoundary);
    }
    else
    {
      Add(PHY_BVAL, SETB_AUX, &Aux_Z4c::PhysicalBoundary);
    }

    // have data on new grid, can do various calculations here...
    Add(WEYL_DECOMP, PHY_BVAL, &Aux_Z4c::WeylDecompose);

    // should depend on last task
    Add(CLEAR_AUXBND, WEYL_DECOMP, &Aux_Z4c::ClearAllAuxBoundary);

  } // end of using namespace block

}

// ----------------------------------------------------------------------------
void Aux_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->StartReceivingAux(BoundaryCommSubset::all);
  return;
}

TaskStatus Aux_Z4c::SendAux(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;
  pz4c->abvar.SendBoundaryBuffers();
  return TaskStatus::success;
}

TaskStatus Aux_Z4c::ReceiveAux(MeshBlock *pmb, int stage)
{
  bool ret;

  ret = pmb->pz4c->abvar.ReceiveBoundaryBuffers();

  if (ret)
  {
    return TaskStatus::success;
  }
  else
  {
    return TaskStatus::fail;
  }
}

TaskStatus Aux_Z4c::SetBoundariesAux(MeshBlock *pmb, int stage)
{
  Z4c *pz4c = pmb->pz4c;
  pz4c->abvar.SetBoundaries();
  return TaskStatus::success;
}

TaskStatus Aux_Z4c::ProlongAux(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;

  // this is post-integ loop
  const Real t_end = this->t_end(stage, pmb);

  // BD: TODO fix this
#if defined(DBG_REDUCE_AUX_COMM)
  // Or just tailor to this..
  pb->ProlongateBoundariesAux(t_end, 0);
#else
  // BD: TODO: this is doing much more work than is required
  // We don't need to apply BC on ubvar_cx again
  // We do however need prol. for vars in e.g. pmr->pvars_cx_
  pb->ProlongateBoundaries(t_end, 0);
#endif // DBG_REDUCE_AUX_COMM

  // BD: should we split to something like this?
  // FCN_CC_CX_VC(
  //   pb->ProlongateBoundaries,
  //   pb->ProlongateCellCenteredXBoundaries,
  //   pb->ProlongateVertexCenteredBoundaries
  // )(t_end_stage, 0);


  return TaskStatus::success;
}

TaskStatus Aux_Z4c::PhysicalBoundary(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    BoundaryValues *pb = pmb->pbval;

    // this is post-integ loop
    const Real t_end = this->t_end(stage, pmb);
    pb->ApplyPhysicalBoundariesAux(t_end, 0);

    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

TaskStatus Aux_Z4c::ClearAllAuxBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundaryAux(BoundaryCommSubset::all);
  return TaskStatus::success;
}

TaskStatus Aux_Z4c::WeylDecompose(MeshBlock *pmb, int stage)
{
  Mesh *pm = pmb->pmy_mesh;

  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction)) {
    // std::cout << "exec. Aux_Z4c WeylDecompose" << std::endl;
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
bool Aux_Z4c::CurrentTimeCalculationThreshold(
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

void Aux_Z4c::UpdateTaskListTriggers() {
  // note that for global dt > target output dt
  // next_time will 'lag'; this will need to be corrected if an integrator with dense /
  // interpolating output is used.

  if (TaskListTriggers.wave_extraction.to_update) {
    TaskListTriggers.wave_extraction.next_time += \
      TaskListTriggers.wave_extraction.dt;
    TaskListTriggers.wave_extraction.to_update = false;
  }
}

//
// :D
//
