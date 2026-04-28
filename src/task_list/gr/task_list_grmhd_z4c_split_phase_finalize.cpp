// C/C++ headers
#include <iostream>  // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/z4c.hpp"
#include "task_list.hpp"
#include "task_names.hpp"

// #if M1_ENABLED
// #include "../../m1/m1.hpp"
// #endif

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GRMHD_Z4c_Phase_Finalize::GRMHD_Z4c_Phase_Finalize(ParameterInput* pin,
                                                   Mesh* pm,
                                                   Triggers& trgs)
    : LowStorage(pin, pm), trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Finalize;

  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  // Finalization logic -------------------------------------------------------
  Add(CONS2PRIMG, NONE, &GRMHD_Z4c_Phase_Finalize::PrimitivesGhosts);
  Add(UPDATE_SRC, CONS2PRIMG, &GRMHD_Z4c_Phase_Finalize::UpdateSource);

  // Note: conformal derivative 3D arrays were already filled by
  // Phase_Z4c::PREP_Z4C_DERIV on the final substep and persist on the
  // MeshBlock, so ADM_CONSTR and Z4C_WEYL can read them directly.
  Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c_Phase_Finalize::ADM_Constraints);
  Add(Z4C_WEYL, UPDATE_SRC, &GRMHD_Z4c_Phase_Finalize::Z4c_Weyl);

  Add(USERWORK, UPDATE_SRC, &GRMHD_Z4c_Phase_Finalize::UserWork);

  // only depend on geometry, which isn't affected by the above
  Add(NEW_DT, NONE, &GRMHD_Z4c_Phase_Finalize::NewBlockTimeStep);

  if (adaptive)
    Add(FLAG_AMR, NONE, &GRMHD_Z4c_Phase_Finalize::CheckRefinement);
}

// ----------------------------------------------------------------------------
void GRMHD_Z4c_Phase_Finalize::StartupTaskList(MeshBlock* pmb, int stage)
{
  return;
}

TaskStatus GRMHD_Z4c_Phase_Finalize::PrimitivesGhosts(MeshBlock* pmb,
                                                      int stage)
{
  // Construct primitives from conserved on the whole MeshBlock.
  if (stage <= nstages)
  {
    Hydro* ph             = pmb->phydro;
    Field* pf             = pmb->pfield;
    PassiveScalars* ps    = pmb->pscalars;
    EquationOfState* peos = pmb->peos;

    int il = 0, iu = pmb->ncells1 - 1;
    int jl = 0, ju = pmb->ncells2 - 1;
    int kl = 0, ku = pmb->ncells3 - 1;

    static const int coarseflag     = 0;
    static const bool skip_physical = true;
    peos->ConservedToPrimitive(ph->u,
                               ph->w1,
                               ph->w,
                               ps->s,
                               ps->r,
                               pf->bcc,
                               pmb->pcoord,
                               il,
                               iu,
                               jl,
                               ju,
                               kl,
                               ku,
                               coarseflag,
                               skip_physical);

    if (pmb->peos->smooth_temperature)
    {
      peos->SmoothTemperatureAndRecompute(ph->w,
                                          ph->w1,
                                          ph->derived_ms,
                                          ps->r,
                                          il,
                                          iu,
                                          jl,
                                          ju,
                                          kl,
                                          ku,
                                          pmb->precon->xorder_use_aux_cs2,
                                          pmb->precon->xorder_use_aux_s);
    }

    // Update w1 to have the state of w
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_Finalize::UserWork(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->UserWorkInLoop();

  // TODO: BD- this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_Finalize::CheckRefinement(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_Finalize::Z4c_Weyl(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Mesh* pm  = pmb->pmy_mesh;
  Z4c* pz4c = pmb->pz4c;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_Weyl))
  {
    pmb->pz4c->Z4cWeyl(
      pmb->pz4c->storage.adm, pmb->pz4c->storage.mat, pmb->pz4c->storage.weyl);
  }

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_Finalize::ADM_Constraints(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Mesh* pm  = pmb->pmy_mesh;
  Z4c* pz4c = pmb->pz4c;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_ADM_constraints))
  {
    pz4c->ADMConstraints(pz4c->storage.con,
                         pz4c->storage.adm,
                         pz4c->storage.mat,
                         pz4c->storage.u);
  }
  return TaskStatus::next;
}

// new dt ---------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_Finalize::NewBlockTimeStep(MeshBlock* pmb,
                                                      int stage)
{
  // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@T:Fin\n", "@E:Fin\n");

  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  // NB using the Z4C version of this fn rather than fluid - potential issue?
  Z4c* pz4c = pmb->pz4c;
  pz4c->NewBlockTimeStep();

  return TaskStatus::next;
}

// Recouple ADM sources -------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_Finalize::UpdateSource(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    Hydro* ph = pmb->phydro;
    Field* pf = pmb->pfield;

    PassiveScalars* ps = pmb->pscalars;

    pz4c->GetMatter(
      pz4c->storage.mat, pz4c->storage.adm, ph->w, ps->r, pf->bcc);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//
// :D
//
