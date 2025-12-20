// C/C++ headers
#include <iostream>   // endl
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../hydro/srcterms/hydro_srcterms.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "task_list.hpp"
#include "task_names.hpp"
#if CCE_ENABLED
#include "../../z4c/cce/cce.hpp"
#endif

// #if M1_ENABLED
// #include "../../m1/m1.hpp"
// #endif

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GRMHD_Z4c_Phase_Z4c::GRMHD_Z4c_Phase_Z4c(ParameterInput *pin,
                                         Mesh *pm,
                                         Triggers &trgs)
  : LowStorage(pin, pm),
    trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_Z4c;

  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

#ifdef USE_COMM_DEPENDENCY
  // Accumulate MPI communication tasks:
  TaskID COMM = NONE;
#endif

  // Z4c sub-system logic ---------------------------------------------------
  Add(CALC_Z4CRHS, NONE,        &GRMHD_Z4c_Phase_Z4c::CalculateZ4cRHS);
  Add(INT_Z4C,     CALC_Z4CRHS, &GRMHD_Z4c_Phase_Z4c::IntegrateZ4c);

  // Should be able to do this
  Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c_Phase_Z4c::SendZ4c);
  Add(RECV_Z4C, NONE, &GRMHD_Z4c_Phase_Z4c::ReceiveZ4c);
#ifdef USE_COMM_DEPENDENCY
  COMM = COMM | SEND_Z4C | RECV_Z4C;
#endif

  Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GRMHD_Z4c_Phase_Z4c::SetBoundariesZ4c);

  if (multilevel)
  {
    Add(PROLONG_Z4C,  SETB_Z4C, &GRMHD_Z4c_Phase_Z4c::Prolongation_Z4c);
    Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c_Phase_Z4c::PhysicalBoundary_Z4c);
  }
  else
  {
    Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c_Phase_Z4c::PhysicalBoundary_Z4c);
  }

  Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c_Phase_Z4c::EnforceAlgConstr);
  Add(Z4C_TO_ADM, ALG_CONSTR,   &GRMHD_Z4c_Phase_Z4c::Z4cToADM);

#if CCE_ENABLED
    Add(CCE_DUMP, Z4C_TO_ADM,   &GRMHD_Z4c_Phase_Z4c::CCEDump);
#endif

#ifdef USE_COMM_DEPENDENCY
  // We are done with MPI communication
  Add(CLEAR_ALLBND, COMM, &GRMHD_Z4c_Phase_Z4c::ClearAllBoundary);
#else
  // We are done for the z4c phase
  Add(CLEAR_ALLBND, Z4C_TO_ADM, &GRMHD_Z4c_Phase_Z4c::ClearAllBoundary);
#endif

}

// ----------------------------------------------------------------------------
void GRMHD_Z4c_Phase_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb   = pmb->pbval;
  Z4c            *pz4c = pmb->pz4c;

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GRMHD_Z4c_Phase_Z4c::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with GRMHD"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // Auxiliary var u1 needs 0-init at the beginning of each cycle
    pz4c->storage.u1.ZeroClear();
  }

  pb->StartReceiving(BoundaryCommSubset::z4c);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c_Phase_Z4c::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundary(BoundaryCommSubset::z4c);

  // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@T:Z4c\n", "@E:Z4c\n");
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c_Phase_Z4c::CalculateZ4cRHS(MeshBlock *pmb, int stage)
{
  Z4c * pz4c = pmb->pz4c;

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
    pz4c->Z4cBoundaryRHS(pz4c->storage.u,
                         pz4c->storage.mat,
                         pz4c->storage.rhs);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to integrate variables
TaskStatus GRMHD_Z4c_Phase_Z4c::IntegrateZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;

    // See IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pz4c->WeightedAve(pz4c->storage.u1,
                      pz4c->storage.u,
                      pz4c->storage.u2,
                      ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    pz4c->WeightedAve(pz4c->storage.u,
                      pz4c->storage.u1,
                      pz4c->storage.u2,
                      ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pz4c->AddZ4cRHS(pz4c->storage.rhs,
                    dt_scaled,
                    pz4c->storage.u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_Z4c::SendZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    pz4c->ubvar.SendBoundaryBuffers();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_Z4c::ReceiveZ4c(MeshBlock *pmb, int stage)
{
  bool ret;

  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    ret = pz4c->ubvar.ReceiveBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }

  if (ret)
  {
    return TaskStatus::success;
  }
  else
  {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c_Phase_Z4c::SetBoundariesZ4c(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;

    pz4c->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_Z4c::Prolongation_Z4c(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Prolongate z4c vars
    pbval->ProlongateBoundariesZ4c(t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c_Phase_Z4c::PhysicalBoundary_Z4c(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;
    Z4c *pz4c = pmb->pz4c;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pbval->ApplyPhysicalBoundaries(
      t_end, dt_scaled,
      pbval->GetBvarsZ4c(),
      pz4c->mbi.il, pz4c->mbi.iu,
      pz4c->mbi.jl, pz4c->mbi.ju,
      pz4c->mbi.kl, pz4c->mbi.ku,
      pz4c->mbi.ng);

  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c_Phase_Z4c::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
#ifndef DBG_ALGCONSTR_ALL
  if (stage != nstages) return TaskStatus::success; // only do on last stage
#endif // DBG_ALGCONSTR_ALL

  Z4c *pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c_Phase_Z4c::Z4cToADM(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c *pz4c = pmb->pz4c;
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

#if CCE_ENABLED
TaskStatus GRMHD_Z4c_Phase_Z4c::CCEDump(MeshBlock *pmb, int stage)
{
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  Mesh *pm = pmb->pmy_mesh;

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

  return TaskStatus::success;
}
#endif

//
// :D
//
