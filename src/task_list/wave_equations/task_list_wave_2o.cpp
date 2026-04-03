// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../comm/comm_registry.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../wave/wave.hpp"
#include "../task_list.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
using namespace TaskLists::WaveEquations;
using namespace TaskLists::Integrators;
using namespace TaskNames::WaveEquations::WE_2O;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

Wave_2O::Wave_2O(ParameterInput* pin, Mesh* pm, Triggers& trgs)
    : integrators(pin, pm), trgs(trgs)
{
  // Take the number of stages from the integrator
  nstages = integrators::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  {
    Add(CALC_WAVERHS, NONE, &Wave_2O::CalculateWaveRHS);

    Add(INT_WAVE, CALC_WAVERHS, &Wave_2O::IntegrateWave);

    Add(SEND_WAVE, INT_WAVE, &Wave_2O::SendWave);
    Add(RECV_WAVE, NONE, &Wave_2O::ReceiveWave);

    Add(SETB_WAVE, (RECV_WAVE | INT_WAVE), &Wave_2O::SetBoundariesWave);

    if (pm->multilevel)
    {  // SMR or AMR
      Add(PROLONG, (SEND_WAVE | SETB_WAVE), &Wave_2O::Prolongation);
      Add(PHY_BVAL, PROLONG, &Wave_2O::PhysicalBoundary);
    }
    else
    {
      Add(PHY_BVAL, SETB_WAVE, &Wave_2O::PhysicalBoundary);
    }

    Add(USERWORK, PHY_BVAL, &Wave_2O::UserWork);
    Add(NEW_DT, USERWORK, &Wave_2O::NewBlockTimeStep);

    if (pm->adaptive)
    {
      Add(FLAG_AMR, USERWORK, &Wave_2O::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &Wave_2O::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &Wave_2O::ClearAllBoundary);
    }
  }  // namespace
}

// ----------------------------------------------------------------------------
void Wave_2O::StartupTaskList(MeshBlock* pmb, int stage)
{
  Wave* pwave = pmb->pwave;

  integrators::Initialize(stage, pmb, pwave->bt_k, pwave->u, pwave->rhs);

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in Wave_2O::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with Wave_2O" << std::endl;
      ATHENA_ERROR(msg);
    }

    if (is_lowstorage)
    {
      // Auxiliary var u1 needs 0-init at the beginning of each cycle
      pwave->u1.ZeroClear();
    }
  }

  // Post persistent receives for Wave ghost exchange.
  pmb->pcomm->StartReceiving(comm::CommGroup::Wave);
  return;
}

// ----------------------------------------------------------------------------
// Non-blocking clear of outstanding Wave ghost sends and channel flag reset.
// Uses MPI_Test (wait=false) so that the per-block work-stealing lock is
// released immediately when sends are still in flight, preventing deadlocks
// where all threads block inside MPI_Wait simultaneously.
TaskStatus Wave_2O::ClearAllBoundary(MeshBlock* pmb, int stage)
{
  if (!pmb->pcomm->ClearBoundary(
        comm::CommGroup::Wave, comm::CommTarget::All, false))
    return TaskStatus::fail;
  return TaskStatus::next;
}

// ----------------------------------------------------------------------------
// Functions to calculate the RHS
TaskStatus Wave_2O::CalculateWaveRHS(MeshBlock* pmb, int stage)
{
  Wave* pwave = pmb->pwave;

  if (stage <= nstages)
  {
    pwave->WaveRHS(pwave->u);

    // application of Sommerfeld boundary conditions
    pwave->WaveBoundaryRHS(pwave->u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Functions to integrate variables
TaskStatus Wave_2O::IntegrateWave(MeshBlock* pmb, int stage)
{
  Wave* pwave = pmb->pwave;

  if (stage <= nstages)
  {
    if (is_lowstorage)
    {
      if (WAVE_CC_ENABLED)
      {
        Real ave_wghts[3];
        ave_wghts[0] = 1.0;
        ave_wghts[1] = ls->stage_wghts[stage - 1].delta;
        ave_wghts[2] = 0.0;
        pmb->WeightedAveCC(pwave->u1, pwave->u, pwave->u2, ave_wghts);

        ave_wghts[0] = ls->stage_wghts[stage - 1].gamma_1;
        ave_wghts[1] = ls->stage_wghts[stage - 1].gamma_2;
        ave_wghts[2] = ls->stage_wghts[stage - 1].gamma_3;

        pmb->WeightedAveCC(pwave->u, pwave->u1, pwave->u2, ave_wghts);
      }
      else if (WAVE_VC_ENABLED)
      {
        Real ave_wghts[3];
        ave_wghts[0] = 1.0;
        ave_wghts[1] = ls->stage_wghts[stage - 1].delta;
        ave_wghts[2] = 0.0;
        pmb->WeightedAveVC(pwave->u1, pwave->u, pwave->u2, ave_wghts);

        ave_wghts[0] = ls->stage_wghts[stage - 1].gamma_1;
        ave_wghts[1] = ls->stage_wghts[stage - 1].gamma_2;
        ave_wghts[2] = ls->stage_wghts[stage - 1].gamma_3;

        pmb->WeightedAveVC(pwave->u, pwave->u1, pwave->u2, ave_wghts);
      }
      else if (WAVE_CX_ENABLED)
      {
        Real ave_wghts[3];
        ave_wghts[0] = 1.0;
        ave_wghts[1] = ls->stage_wghts[stage - 1].delta;
        ave_wghts[2] = 0.0;
        pmb->WeightedAveCX(pwave->u1, pwave->u, pwave->u2, ave_wghts);

        ave_wghts[0] = ls->stage_wghts[stage - 1].gamma_1;
        ave_wghts[1] = ls->stage_wghts[stage - 1].gamma_2;
        ave_wghts[2] = ls->stage_wghts[stage - 1].gamma_3;

        pmb->WeightedAveCX(pwave->u, pwave->u1, pwave->u2, ave_wghts);
      }

      pwave->AddWaveRHS(ls->stage_wghts[stage - 1].beta, pwave->u);
    }
    else
    {
      const Real dt = pmb->pmy_mesh->dt;

      if (stage < nstages)
      {
        bt->SumBT_ak(pmb, stage + 1, dt, pwave->bt_k, pwave->u);
      }

      if (stage == nstages)
      {
        bt->SumBT_bk(pmb, dt, pwave->bt_k, pwave->u);
      }
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Pack and send Wave ghost-zone data to neighbors.
TaskStatus Wave_2O::SendWave(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Group-level send handles the Wave channel (CC, CX, or VC depending on
    // compile-time config).  Restriction is embedded for multilevel.
    pmb->pcomm->SendBoundaryBuffers(comm::CommGroup::Wave);
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

// ----------------------------------------------------------------------------
// Poll for received Wave ghost-zone data from neighbors.
TaskStatus Wave_2O::ReceiveWave(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    bool done = pmb->pcomm->ReceiveBoundaryBuffers(comm::CommGroup::Wave);
    if (done)
      return TaskStatus::next;
    return TaskStatus::fail;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus Wave_2O::SetBoundariesWave(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Unpack received data for the Wave channel into state arrays.
    pmb->pcomm->SetBoundaries(comm::CommGroup::Wave);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Functions for everything else
TaskStatus Wave_2O::Prolongation(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Wave* pwave               = pmb->pwave;
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Wave uses module-specific coarse indices from MB_info (may differ from
    // pmb->cis/cie when the sampling mode has a different ghost width).
    const int cil = pwave->mbi.cil, ciu = pwave->mbi.ciu;
    const int cjl = pwave->mbi.cjl, cju = pwave->mbi.cju;
    const int ckl = pwave->mbi.ckl, cku = pwave->mbi.cku;
    const int cng = pwave->mbi.cng;

    // Apply coarse-level physical BCs then prolongate each Wave channel.
    pcomm->ProlongateAndApplyPhysicalBCs(comm::CommGroup::Wave,
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

// ----------------------------------------------------------------------------
TaskStatus Wave_2O::PhysicalBoundary(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Apply fine-level physical BCs for every Wave channel.
    pcomm->ApplyPhysicalBCs(comm::CommGroup::Wave, t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }

  return TaskStatus::next;
}

// ----------------------------------------------------------------------------
TaskStatus Wave_2O::UserWork(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->WaveUserWorkInLoop();

  // BD: TODO - this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::next;
}

TaskStatus Wave_2O::NewBlockTimeStep(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->pwave->NewBlockTimeStep();
  return TaskStatus::next;
}

TaskStatus Wave_2O::CheckRefinement(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;  // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::next;
}