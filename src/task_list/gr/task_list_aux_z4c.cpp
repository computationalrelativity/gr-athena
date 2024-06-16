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

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

Aux_Z4c::Aux_Z4c(ParameterInput *pin, Mesh *pm, Triggers &trgs)
  : trgs(trgs)
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

  if (trgs.IsSatisfied(TriggerVariant::Z4c_Weyl))
  {
    AthenaArray<Real> u_R;
    AthenaArray<Real> u_I;
    u_R.InitWithShallowSlice(pz4c->storage.weyl, Z4c::I_WEY_rpsi4, 1);
    u_I.InitWithShallowSlice(pz4c->storage.weyl, Z4c::I_WEY_ipsi4, 1);
    for (auto pwextr : pmb->pwave_extr_loc)
    {
        pwextr->Decompose_multipole(u_R,u_I);
    }
  }

  return TaskStatus::success;
}

//
// :D
//
