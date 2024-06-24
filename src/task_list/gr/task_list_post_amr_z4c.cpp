// C/C++ headers

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../parameter_input.hpp"
#include "task_list.hpp"


// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskNames::GeneralRelativity::PostAMR_Z4c;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

PostAMR_Z4c::PostAMR_Z4c(ParameterInput *pin, Mesh *pm, Triggers &trgs)
  : trgs(trgs)
{
  nstages = 1;
  {
    // Recompute constraints
    Add(ADM_CONSTR, NONE, &PostAMR_Z4c::ADM_Constraints);

    // Recompute Weyl
    Add(Z4C_WEYL, NONE, &PostAMR_Z4c::Z4c_Weyl);

  } // end of using namespace block

}

// ----------------------------------------------------------------------------
void PostAMR_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  return;
}

TaskStatus PostAMR_Z4c::ADM_Constraints(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;

  // BD: TODO - could add a trigger check here, note however that time has
  // been updated cf. earlier task-lists
  pz4c->ADMConstraints(pz4c->storage.con,
                       pz4c->storage.adm,
                       pz4c->storage.mat,
                       pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus PostAMR_Z4c::Z4c_Weyl(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    return TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;

  // BD: TODO - could add a trigger check here, note however that time has
  // been updated cf. earlier task-lists
  pz4c->Z4cWeyl(pz4c->storage.adm,
                pz4c->storage.mat,
                pz4c->storage.weyl);
  return TaskStatus::success;
}

//
// :D
//
