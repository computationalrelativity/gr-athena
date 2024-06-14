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
    // have data on new grid, can do various calculations here...
    Add(WEYL_DECOMP, NONE, &Aux_Z4c::WeylDecompose);
  } // end of using namespace block

}

// ----------------------------------------------------------------------------
void Aux_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  return;
}

TaskStatus Aux_Z4c::WeylDecompose(MeshBlock *pmb, int stage)
{
  Mesh *pm = pmb->pmy_mesh;
  Z4c *pz4c = pmb->pz4c;

  // DEBUG_TRIGGER
  if (1) // (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction))
  {
    AthenaArray<Real> u_R;
    AthenaArray<Real> u_I;
    u_R.InitWithShallowSlice(pz4c->storage.weyl, Z4c::I_WEY_rpsi4, 1);
    u_I.InitWithShallowSlice(pz4c->storage.weyl, Z4c::I_WEY_ipsi4, 1);
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
