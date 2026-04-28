// C/C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/z4c.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskNames::GeneralRelativity::PostAMR_Z4c;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

PostAMR_Z4c::PostAMR_Z4c(ParameterInput* pin, Mesh* pm, Triggers& trgs)
    : trgs(trgs)
{
  nstages = 1;
  {
    // Z4c derivatives are already computed by FinalizeZ4cADMGhosts
    // inside Initialize(regrid) and the z4c_derivs_initialized flag
    // prevents recomputation.  The task below is therefore a no-op.
    // Add(PREP_Z4C_DERIV, NONE, &PostAMR_Z4c::PrepareZ4cDerivatives);

    // ADM constraints are already computed by CalculateZ4cInitDiagnostics()
    // inside Initialize(regrid), using derivatives freshly set by
    // FinalizeZ4cADMGhosts.  No current pgen modifies Z4c state in the
    // PrePostAMR hook, so the recomputation here is redundant.
    // Add(ADM_CONSTR, PREP_Z4C_DERIV, &PostAMR_Z4c::ADM_Constraints);

    // Recompute Weyl on new blocks for output consistency.
    Add(Z4C_WEYL, NONE, &PostAMR_Z4c::Z4c_Weyl);

  }  // end of using namespace block
}

// ----------------------------------------------------------------------------
void PostAMR_Z4c::StartupTaskList(MeshBlock* pmb, int stage)
{
  return;
}

// Pre-compute conformal derivative 3D arrays ---------------------------------
TaskStatus PostAMR_Z4c::PrepareZ4cDerivatives(MeshBlock* pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::next;
  }

  Z4c* pz4c = pmb->pz4c;
  // Use InitializeZ4cDerivatives (guarded by z4c_derivs_initialized flag)
  // to avoid redundant recomputation when FinalizeZ4cADMGhosts already called
  // it.
  pz4c->InitializeZ4cDerivatives(pz4c->storage.u);
  return TaskStatus::next;
}

TaskStatus PostAMR_Z4c::ADM_Constraints(MeshBlock* pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::next;
  }

  Z4c* pz4c = pmb->pz4c;

  pz4c->ADMConstraints(
    pz4c->storage.con, pz4c->storage.adm, pz4c->storage.mat, pz4c->storage.u);

  return TaskStatus::next;
}

TaskStatus PostAMR_Z4c::Z4c_Weyl(MeshBlock* pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::next;
  }

  Z4c* pz4c = pmb->pz4c;

  pz4c->Z4cWeyl(pz4c->storage.adm, pz4c->storage.mat, pz4c->storage.weyl);
  return TaskStatus::next;
}

//
// :D
//
