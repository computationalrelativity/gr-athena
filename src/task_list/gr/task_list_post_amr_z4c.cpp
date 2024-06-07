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
using namespace TaskNames::GeneralRelativity::PostAMR_Z4c;

PostAMR_Z4c::PostAMR_Z4c(ParameterInput *pin, Mesh *pm)
{
  nstages = 1;
  {
    // Algebraic constraints, prepare ADM
    Add(ALG_CONSTR, NONE,       &PostAMR_Z4c::EnforceAlgConstr);
    Add(Z4C_TO_ADM, ALG_CONSTR, &PostAMR_Z4c::Z4cToADM);

    if (FLUID_ENABLED || MAGNETIC_FIELDS_ENABLED)
    {
      // ADM sources need updating (fluid populated via R/P on cons.)
      Add(UPDATE_SRC, Z4C_TO_ADM, &PostAMR_Z4c::UpdateSource);
      Add(ADM_CONSTR, UPDATE_SRC, &PostAMR_Z4c::ADM_Constraints);
    }
    else
    {
      // vacuum
      Add(ADM_CONSTR, Z4C_TO_ADM, &PostAMR_Z4c::ADM_Constraints);
    }

    // Recompute Weyl (strictly only needed for 3d dump)
    Add(Z4C_WEYL, Z4C_TO_ADM, &PostAMR_Z4c::Z4c_Weyl);

  } // end of using namespace block

}

// ----------------------------------------------------------------------------
void PostAMR_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  return;
}

TaskStatus PostAMR_Z4c::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);
  return TaskStatus::success;
}

TaskStatus PostAMR_Z4c::Z4cToADM(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;
  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
  return TaskStatus::success;
}

TaskStatus PostAMR_Z4c::UpdateSource(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    TaskStatus::success;
  }

  Hydro *ph = pmb->phydro;
  Z4c *pz4c = pmb->pz4c;

#if USETM
  pz4c->GetMatter(
    pz4c->storage.mat,
    pz4c->storage.adm,
    ph->w,
    pmb->pscalars->r,
    pmb->pfield->bcc
  );
#else
  pz4c->GetMatter(
    pz4c->storage.mat,
    pz4c->storage.adm,
    ph->w,
    pmb->pfield->bcc
  );
#endif  // USETM

  return TaskStatus::success;
}

TaskStatus PostAMR_Z4c::ADM_Constraints(MeshBlock *pmb, int stage)
{
  if (!pmb->IsNewFromAMR())
  {
    TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;

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
    TaskStatus::success;
  }

  Z4c *pz4c = pmb->pz4c;

  pz4c->Z4cWeyl(pz4c->storage.adm,
                pz4c->storage.mat,
                pz4c->storage.weyl);
  return TaskStatus::success;
}

//
// :D
//
