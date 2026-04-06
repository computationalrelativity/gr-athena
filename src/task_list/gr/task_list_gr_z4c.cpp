// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/z4c_macro.hpp"
#if CCE_ENABLED
#include "../../z4c/cce/cce.hpp"
#endif
#include "../../comm/comm_registry.hpp"
#include "../../parameter_input.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;
using namespace TaskNames::GeneralRelativity::GR_Z4c;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GR_Z4c::GR_Z4c(ParameterInput* pin, Mesh* pm, Triggers& trgs)
    : integrators(pin, pm), trgs(trgs)
{
  // Take the number of stages from the integrator
  nstages = integrators::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  {
    Add(INIT_Z4C_DERIV, NONE, &GR_Z4c::InitializeZ4cDerivatives);
    Add(CALC_Z4CRHS, INIT_Z4C_DERIV, &GR_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C, CALC_Z4CRHS, &GR_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GR_Z4c::SendZ4c);
    Add(RECV_Z4C, NONE, &GR_Z4c::ReceiveZ4c);

    Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GR_Z4c::SetBoundariesZ4c);

    if (multilevel)
    {
      Add(PROLONG, (SEND_Z4C | SETB_Z4C), &GR_Z4c::Prolongation);
      Add(PHY_BVAL, PROLONG, &GR_Z4c::PhysicalBoundary);
    }
    else
    {
      Add(PHY_BVAL, SETB_Z4C, &GR_Z4c::PhysicalBoundary);
    }

    Add(ALG_CONSTR, PHY_BVAL, &GR_Z4c::EnforceAlgConstr);

    // Pre-compute conformal derivative 3D arrays from post-communication z4c.u
    Add(PREP_Z4C_DERIV, ALG_CONSTR, &GR_Z4c::PrepareZ4cDerivatives);

    Add(Z4C_TO_ADM, PREP_Z4C_DERIV, &GR_Z4c::Z4cToADM);

    Add(ADM_CONSTR, (Z4C_TO_ADM | PREP_Z4C_DERIV), &GR_Z4c::ADM_Constraints);
    Add(Z4C_WEYL, (Z4C_TO_ADM | PREP_Z4C_DERIV), &GR_Z4c::Z4c_Weyl);

#if CCE_ENABLED
    Add(CCE_DUMP, Z4C_TO_ADM, &GR_Z4c::CCEDump);
#endif

    Add(USERWORK, ADM_CONSTR, &GR_Z4c::UserWork);
    Add(NEW_DT, USERWORK, &GR_Z4c::NewBlockTimeStep);

    // We are done for the z4c phase
    if (adaptive)
    {
      Add(FLAG_AMR, USERWORK, &GR_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GR_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &GR_Z4c::ClearAllBoundary);
    }

    // Add(ASSERT_FIN, CLEAR_ALLBND, &GR_Z4c::AssertFinite);
  }  // namespace
}

// ----------------------------------------------------------------------------
void GR_Z4c::StartupTaskList(MeshBlock* pmb, int stage)
{
  Z4c* pz4c = pmb->pz4c;

  integrators::Initialize(
    stage, pmb, pz4c->bt_k, pz4c->storage.u, pz4c->storage.rhs);

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GR_Z4c::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with GR" << std::endl;
      ATHENA_ERROR(msg);
    }

    if (is_lowstorage)
    {
      // Auxiliary var u1 needs 0-init at the beginning of each cycle
      pz4c->storage.u1.ZeroClear();
    }
  }

  // Post persistent receives for Z4c ghost exchange.
  pmb->pcomm->StartReceiving(comm::CommGroup::Z4c);
  return;
}

//-----------------------------------------------------------------------------
// Non-blocking clear of outstanding Z4c ghost sends and channel flag reset.
// Uses MPI_Test (wait=false) so that the per-block work-stealing lock is
// released immediately when sends are still in flight, preventing deadlocks
// where all threads block inside MPI_Wait simultaneously.
TaskStatus GR_Z4c::ClearAllBoundary(MeshBlock* pmb, int stage)
{
  if (!pmb->pcomm->ClearBoundary(
        comm::CommGroup::Z4c, comm::CommTarget::All, false))
    return TaskStatus::fail;
  return TaskStatus::next;
}

//-----------------------------------------------------------------------------
// Functions to calculate the RHS
TaskStatus GR_Z4c::CalculateZ4cRHS(MeshBlock* pmb, int stage)
{
  Z4c* pz4c = pmb->pz4c;

  // PunctureTracker: interpolate beta at puncture position before evolution
  if (stage == 1)
  {
    for (auto ptracker : pmb->pmy_mesh->pz4c_tracker)
    {
      ptracker->InterpolateShift(pmb, pz4c->storage.u);
    }
  }

  if (stage <= nstages)
  {
    pz4c->Z4cRHS(pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

    // Sommerfeld boundary conditions
    pz4c->Z4cBoundaryRHS(
      pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to integrate variables
TaskStatus GR_Z4c::IntegrateZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;

    if (is_lowstorage)
    {
      Real ave_wghts[3];
      ave_wghts[0] = 1.0;
      ave_wghts[1] = ls->stage_wghts[stage - 1].delta;
      ave_wghts[2] = 0.0;
      pz4c->WeightedAve(
        pz4c->storage.u1, pz4c->storage.u, pz4c->storage.u2, ave_wghts);

      ave_wghts[0] = ls->stage_wghts[stage - 1].gamma_1;
      ave_wghts[1] = ls->stage_wghts[stage - 1].gamma_2;
      ave_wghts[2] = ls->stage_wghts[stage - 1].gamma_3;

      pz4c->WeightedAve(
        pz4c->storage.u, pz4c->storage.u1, pz4c->storage.u2, ave_wghts);

      const Real dt_scaled = this->dt_scaled(stage, pmb);
      pz4c->AddZ4cRHS(pz4c->storage.rhs, dt_scaled, pz4c->storage.u);
    }
    else
    {
      const Real dt = pmb->pmy_mesh->dt;

      if (stage < nstages)
      {
        bt->SumBT_ak(pmb, stage + 1, dt, pz4c->bt_k, pz4c->storage.u);
      }

      if (stage == nstages)
      {
        bt->SumBT_bk(pmb, dt, pz4c->bt_k, pz4c->storage.u);
      }
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Pack and send Z4c ghost-zone data to neighbors.
TaskStatus GR_Z4c::SendZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Group-level send handles all Z4c channels.
    // Restriction into coarse buffers (for cross-level neighbors) is embedded
    // inside SendBoundaryBuffers when mesh is multilevel.
    pmb->pcomm->SendBoundaryBuffers(comm::CommGroup::Z4c);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Poll for received Z4c ghost-zone data from neighbors.
TaskStatus GR_Z4c::ReceiveZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Poll all Z4c channels.  Returns true when every channel has received
    // all expected messages from same-level, coarser, and finer neighbors.
    bool done = pmb->pcomm->ReceiveBoundaryBuffers(comm::CommGroup::Z4c);
    if (done)
      return TaskStatus::next;
    return TaskStatus::fail;
  }
  return TaskStatus::fail;
}

TaskStatus GR_Z4c::SetBoundariesZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Unpack received data for all Z4c channels into state arrays.
    pmb->pcomm->SetBoundaries(comm::CommGroup::Z4c);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GR_Z4c::Prolongation(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c                 = pmb->pz4c;
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Z4c uses module-specific coarse indices from MB_info (may differ from
    // pmb->cis/cie when the sampling mode has a different ghost width).
    const int cil = pz4c->mbi.cil, ciu = pz4c->mbi.ciu;
    const int cjl = pz4c->mbi.cjl, cju = pz4c->mbi.cju;
    const int ckl = pz4c->mbi.ckl, cku = pz4c->mbi.cku;
    const int cng = pz4c->mbi.cng;

    // Apply coarse-level physical BCs then prolongate each Z4c channel.
    pcomm->ProlongateAndApplyPhysicalBCs(comm::CommGroup::Z4c,
                                         t_end,
                                         dt_scaled,
                                         cil,
                                         ciu,
                                         cjl,
                                         cju,
                                         ckl,
                                         cku,
                                         cng);
  }
  else
  {
    return TaskStatus::fail;
  }

  return TaskStatus::next;
}

TaskStatus GR_Z4c::PhysicalBoundary(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Apply fine-level physical BCs for every Z4c channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::Z4c, t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

TaskStatus GR_Z4c::UserWork(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->UserWorkInLoop();

  // BD: TODO - this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::next;
}

TaskStatus GR_Z4c::EnforceAlgConstr(MeshBlock* pmb, int stage)
{
  Z4c* pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);

  return TaskStatus::next;
}

TaskStatus GR_Z4c::Z4cToADM(MeshBlock* pmb, int stage)
{
  // BD: this map is only required on the final stage
  if (stage != nstages)
    return TaskStatus::next;

  Z4c* pz4c = pmb->pz4c;
  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
  return TaskStatus::next;
}

TaskStatus GR_Z4c::Z4c_Weyl(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Mesh* pm  = pmb->pmy_mesh;
  Z4c* pz4c = pmb->pz4c;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_Weyl))
  {
    pz4c->Z4cWeyl(pz4c->storage.adm, pz4c->storage.mat, pz4c->storage.weyl);
  }

  return TaskStatus::next;
}

#if CCE_ENABLED
TaskStatus GR_Z4c::CCEDump(MeshBlock* pmb, int stage)
{
  // only do on last stage
  if (stage != nstages)
    return TaskStatus::next;

  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  Mesh* pm = pmb->pmy_mesh;

  for (auto cce : pm->pcce)
  {
    // BD: TODO- double check the following
    const Real dt_cce = trgs.GetTrigger_dt(tvar::Z4c_CCE, ovar::user);

    if (dt_cce > 0)
    {
      const Real time = pm->time;

      // Trigger when time is (within tolerance) an integer multiple of dt_cce
      if (std::fabs(std::fmod(time, dt_cce)) > 1e-12)
      {
        continue;
      }

      cce->Interpolate(pmb);
    }
  }

  return TaskStatus::next;
}
#endif

TaskStatus GR_Z4c::ADM_Constraints(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Mesh* pm  = pmb->pmy_mesh;
  Z4c* pz4c = pmb->pz4c;

  // DEBUG_TRIGGER
  // if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con) ||
  //     CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con_hst))
  // {      // Time at the end of stage for (u, b) register pair

  if (trgs.IsSatisfied(TriggerVariant::Z4c_ADM_constraints))
  {
    pz4c->ADMConstraints(pz4c->storage.con,
                         pz4c->storage.adm,
                         pz4c->storage.mat,
                         pz4c->storage.u);
  }
  return TaskStatus::next;
}

TaskStatus GR_Z4c::NewBlockTimeStep(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  Z4c* pz4c = pmb->pz4c;
  pz4c->NewBlockTimeStep();
  return TaskStatus::next;
}

TaskStatus GR_Z4c::CheckRefinement(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::next;
}

// Pre-compute conformal derivative 3D arrays ---------------------------------
TaskStatus GR_Z4c::PrepareZ4cDerivatives(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    pz4c->PrepareZ4cDerivatives(pz4c->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// One-time initialization of derivative 3D arrays for fresh MeshBlocks -------
TaskStatus GR_Z4c::InitializeZ4cDerivatives(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    pz4c->InitializeZ4cDerivatives(pz4c->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

/*
TaskStatus GR_Z4c::AssertFinite(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::next;

  Mesh *pm = pmb->pmy_mesh;

  // check last stage is actually at a relevant time
  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.assert_is_finite))
{ pmb->pz4c->assert_is_finite_adm(); pmb->pz4c->assert_is_finite_con();
    pmb->pz4c->assert_is_finite_mat();
    pmb->pz4c->assert_is_finite_z4c();
  }

  return TaskStatus::next;
}
*/

//
// :D
//
