// C/C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../scalars/scalars.hpp"
#include "../../m1/m1.hpp"
#include "../task_list.hpp"
#include "task_list.hpp"


// ----------------------------------------------------------------------------
using namespace TaskLists::M1;
using namespace TaskLists::Integrators;
using namespace TaskNames::M1::PostAMR_M1N0;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

PostAMR_M1N0::PostAMR_M1N0(ParameterInput *pin, Mesh *pm, Triggers &trgs)
  : trgs(trgs)
{
  nstages = 1;
  {
    Add(UPDATE_BG, NONE, &PostAMR_M1N0::UpdateBackground);
    Add(CALC_FIDU, UPDATE_BG, &PostAMR_M1N0::CalcFiducialVelocity);
    Add(CALC_CLOSURE, CALC_FIDU, &PostAMR_M1N0::CalcClosure);
    Add(CALC_FIDU_FRAME, CALC_CLOSURE, &PostAMR_M1N0::CalcFiducialFrame);
    Add(CALC_OPAC, CALC_FIDU_FRAME, &PostAMR_M1N0::CalcOpacity);
    Add(ANALYSIS, CALC_OPAC, &PostAMR_M1N0::Analysis);

  } // end of using namespace block

}

// ----------------------------------------------------------------------------
void PostAMR_M1N0::StartupTaskList(MeshBlock *pmb, int stage)
{
  return;
}

// ----------------------------------------------------------------------------
// Update external background / dynamical field states
TaskStatus PostAMR_M1N0::UpdateBackground(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;
  pm1->UpdateGeometry(pm1->geom, pm1->scratch);
  pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to Calculate Fiducial Velocity
TaskStatus PostAMR_M1N0::CalcFiducialVelocity(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;
  pm1->CalcFiducialVelocity();
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to calculate Closure
TaskStatus PostAMR_M1N0::CalcClosure(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;
  pm1->CalcClosure(pm1->storage.u);
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
TaskStatus PostAMR_M1N0::CalcFiducialFrame(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  pm1->CalcFiducialFrame(pm1->storage.u);
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to calculate Opacities
TaskStatus PostAMR_M1N0::CalcOpacity(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::success;
  }

  const Real dt = 0;
  pm1->CalcOpacity(0, pm1->storage.u);

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
TaskStatus PostAMR_M1N0::Analysis(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  // ...

  return TaskStatus::success;
}

//
// :D
//
