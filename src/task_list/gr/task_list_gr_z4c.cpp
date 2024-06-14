// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../z4c/z4c.hpp"
#include "../../z4c/z4c_macro.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../trackers/extrema_tracker.hpp"
#if CCE_ENABLED
#include "../../z4c/cce/cce.hpp"
#endif
#include "../../z4c/puncture_tracker.hpp"
#include "../../parameter_input.hpp"
#include "task_list.hpp"


// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;
using namespace TaskNames::GeneralRelativity::GR_Z4c;

GR_Z4c::GR_Z4c(ParameterInput *pin, Mesh *pm)
  : LowStorage(pin, pm)
{
  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------
  // Output frequency control (on task-list)
  TaskListTriggers.assert_is_finite.next_time = pm->time;
  TaskListTriggers.assert_is_finite.dt = pin->GetOrAddReal(
    "task_triggers",
    "dt_assert_is_finite", 0.0);

  // For constraint calculation
  TaskListTriggers.con.next_time = pm->time;
  // Seed TaskListTriggers.con.dt in main

  // Initialize dt for history output to calculate the constraint
  InputBlock *pib = pin->pfirst_block;
  std::string aux;

  TaskListTriggers.con_hst.next_time = 1000000.;
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 6, "output") == 0) {
      aux = pin->GetOrAddString(pib->block_name,"file_type","none");
      if (std::strcmp(aux.c_str(),"hst") == 0) {
        TaskListTriggers.con_hst.dt = pin->GetOrAddReal(pib->block_name,"dt",0.);
        break;
      }
    }
    pib = pib->pnext;
  }
  if (TaskListTriggers.con_hst.dt > 0) TaskListTriggers.con_hst.next_time = std::floor(pm->time/TaskListTriggers.con_hst.dt)*TaskListTriggers.con_hst.dt;

  TaskListTriggers.wave_extraction.to_update = false;

  TaskListTriggers.wave_extraction.dt = pin->GetOrAddReal(
    "task_triggers", "dt_psi4_extraction", 1.0);
  if (pin->GetOrAddInteger("psi4_extraction", "num_radii", 0) == 0) {
    TaskListTriggers.wave_extraction.dt = 0.0;
    TaskListTriggers.wave_extraction.next_time = 0.0;
    TaskListTriggers.wave_extraction.to_update = false;
  }
  else {
    // When initializing at restart, this procedure ensures to restart
    // extraction from right time
    int nwavecycles = static_cast<int>(pm->time/TaskListTriggers.wave_extraction.dt);
    TaskListTriggers.wave_extraction.next_time = (nwavecycles + 1)*
        TaskListTriggers.wave_extraction.dt;
  }
  
#if CCE_ENABLED
  // CCE 
  TaskListTriggers.cce_dump.dt = pin->GetOrAddReal("cce", "dump_every_dt", 1.0);
  if (pin->GetOrAddInteger("cce", "num_radii", 0) == 0) {
    TaskListTriggers.cce_dump.dt = 0.0;
    TaskListTriggers.cce_dump.next_time = 0.0;
    TaskListTriggers.cce_dump.to_update = false;
  }
  else {
    // we need to write at t = 0.
    // ensuring there is no duplicated iteration a bookkeeping system is used.
    int ncycles = static_cast<int>(pm->time/TaskListTriggers.cce_dump.dt);
    TaskListTriggers.cce_dump.next_time = 
      (ncycles == 0) ? 0.0: (ncycles+1)*TaskListTriggers.cce_dump.dt;
  }
#endif

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  {
    Add(CALC_Z4CRHS, NONE,        &GR_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C,     CALC_Z4CRHS, &GR_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GR_Z4c::SendZ4c);
    Add(RECV_Z4C, NONE,    &GR_Z4c::ReceiveZ4c);

    Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GR_Z4c::SetBoundariesZ4c);

    if (multilevel)
    {
      Add(PROLONG, (SEND_Z4C | SETB_Z4C), &GR_Z4c::Prolongation);
      Add(PHY_BVAL, PROLONG,            &GR_Z4c::PhysicalBoundary);
    }
    else
    {
      Add(PHY_BVAL, SETB_Z4C, &GR_Z4c::PhysicalBoundary);
    }

    Add(ALG_CONSTR, PHY_BVAL,   &GR_Z4c::EnforceAlgConstr);

    Add(Z4C_TO_ADM, ALG_CONSTR, &GR_Z4c::Z4cToADM);

    Add(ADM_CONSTR, Z4C_TO_ADM, &GR_Z4c::ADM_Constraints);
    Add(Z4C_WEYL,   Z4C_TO_ADM, &GR_Z4c::Z4c_Weyl);

#if CCE_ENABLED
    Add(CCE_DUMP, Z4C_TO_ADM, &GR_Z4c::CCEDump);
#endif

    Add(USERWORK, ADM_CONSTR, &GR_Z4c::UserWork);
    Add(NEW_DT,   USERWORK,   &GR_Z4c::NewBlockTimeStep);

    if (adaptive)
    {
      Add(FLAG_AMR,     USERWORK, &GR_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GR_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT,   &GR_Z4c::ClearAllBoundary);
    }

    // Add(ASSERT_FIN, CLEAR_ALLBND, &GR_Z4c::AssertFinite);
  } // namespace
}

// ----------------------------------------------------------------------------
void GR_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb   = pmb->pbval;
  Z4c            *pz4c = pmb->pz4c;

  // Application of Sommerfeld boundary conditions
  pz4c->Z4cBoundaryRHS(pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

  const Real t_end = this->t_end(stage, pmb);
  const Real dt_scaled = this->dt_scaled(stage, pmb);

  FCN_CC_CX_VC(
    pb->ApplyPhysicalBoundaries,
    pb->ApplyPhysicalCellCenteredXBoundaries,
    pb->ApplyPhysicalVertexCenteredBoundaries
  )(t_end, dt_scaled);

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GR_Z4c::StartupTaskList\n"
          << "integrator=" << integrator << " is currently incompatible with GR"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // Auxiliary var u1 needs 0-init at the beginning of each cycle
    pz4c->storage.u1.ZeroClear();
  }

  pb->StartReceiving(BoundaryCommSubset::all);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GR_Z4c::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundary(BoundaryCommSubset::all);
  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to calculate the RHS

TaskStatus GR_Z4c::CalculateZ4cRHS(MeshBlock *pmb, int stage)
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
TaskStatus GR_Z4c::IntegrateZ4c(MeshBlock *pmb, int stage)
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

    // BD: TODO - why does this give a slightly different result?
    // if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
    // {
    //   // pz4c->storage.u.SwapAthenaArray(pz4c->storage.u1);
    //   std::swap(pz4c->storage.u, pz4c->storage.u1);
    // }
    // else
    // {
    //   pz4c->WeightedAve(pz4c->storage.u,
    //                     pz4c->storage.u1,
    //                     pz4c->storage.u2,
    //                     ave_wghts);
    // }

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
TaskStatus GR_Z4c::SendZ4c(MeshBlock *pmb, int stage)
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
TaskStatus GR_Z4c::ReceiveZ4c(MeshBlock *pmb, int stage)
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

TaskStatus GR_Z4c::SetBoundariesZ4c(MeshBlock *pmb, int stage)
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
TaskStatus GR_Z4c::Prolongation(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pbval->ProlongateBoundaries(t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus GR_Z4c::PhysicalBoundary(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    BoundaryValues *pbval = pmb->pbval;

    const Real t_end = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // switch based on sampling
    FCN_CC_CX_VC(
        pbval->ApplyPhysicalBoundaries,
        pbval->ApplyPhysicalCellCenteredXBoundaries,
        pbval->ApplyPhysicalVertexCenteredBoundaries
    )(t_end, dt_scaled);

  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus GR_Z4c::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();

  // TODO: BD- this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::success;
}

TaskStatus GR_Z4c::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
#ifndef DBG_ALGCONSTR_ALL
  if (stage != nstages) return TaskStatus::success; // only do on last stage
#endif // DBG_ALGCONSTR_ALL

  Z4c *pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus GR_Z4c::Z4cToADM(MeshBlock *pmb, int stage)
{
  // BD: this map is only required on the final stage
  if (stage != nstages) return TaskStatus::success;

  Z4c *pz4c = pmb->pz4c;
  pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
  return TaskStatus::success;
}

TaskStatus GR_Z4c::Z4c_Weyl(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm   = pmb->pmy_mesh;
  Z4c  *pz4c = pmb->pz4c;

  // DEBUG_TRIGGER
  // if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction))
  if (1)
  {
    pmb->pz4c->Z4cWeyl(pmb->pz4c->storage.adm,
                       pmb->pz4c->storage.mat,
                       pmb->pz4c->storage.weyl);
  }

  return TaskStatus::success;
}

#if CCE_ENABLED
TaskStatus GR_Z4c::CCEDump(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;

  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.cce_dump))
  {
    for (auto cce : pm->pcce)
    {
      cce->Interpolate(pmb);
    }
  }

  return TaskStatus::success;
}
#endif

TaskStatus GR_Z4c::ADM_Constraints(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm   = pmb->pmy_mesh;
  Z4c  *pz4c = pmb->pz4c;

  // DEBUG_TRIGGER
  // if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con) ||
  //     CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con_hst))
  // {      // Time at the end of stage for (u, b) register pair

  if (1) {
    pz4c->ADMConstraints(pz4c->storage.con, pz4c->storage.adm,
                         pz4c->storage.mat, pz4c->storage.u);

  }
  return TaskStatus::success;
}

TaskStatus GR_Z4c::NewBlockTimeStep(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  Z4c *pz4c = pmb->pz4c;
  pz4c->NewBlockTimeStep();
  return TaskStatus::success;
}

TaskStatus GR_Z4c::CheckRefinement(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}

/*
TaskStatus GR_Z4c::AssertFinite(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;

  // check last stage is actually at a relevant time
  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.assert_is_finite)) {
    pmb->pz4c->assert_is_finite_adm();
    pmb->pz4c->assert_is_finite_con();
    pmb->pz4c->assert_is_finite_mat();
    pmb->pz4c->assert_is_finite_z4c();
  }

  return TaskStatus::success;
}
*/

//----------------------------------------------------------------------------------------
// \!fn bool GR_Z4c::CurrentTimeCalculationThreshold(
//   MeshBlock *pmb, aux_NextTimeStep *variable)
// \brief Given current time / ncycles, does a specified 'dt' mean we need
//        to calculate something?
//        Secondary effect is to mutate next_time
bool GR_Z4c::CurrentTimeCalculationThreshold(
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

//----------------------------------------------------------------------------------------
// \!fn void GR_Z4c::UpdateTaskListTriggers()
// \brief Update 'next_time' outside task list to avoid race condition
void GR_Z4c::UpdateTaskListTriggers() {
  // note that for global dt > target output dt
  // next_time will 'lag'; this will need to be corrected if an integrator with dense /
  // interpolating output is used.

  if (TaskListTriggers.adm.to_update) {
    TaskListTriggers.adm.next_time += TaskListTriggers.adm.dt;
    TaskListTriggers.adm.to_update = false;
  }

  if (TaskListTriggers.con.to_update) {
    TaskListTriggers.con.next_time += TaskListTriggers.con.dt;
    TaskListTriggers.con.to_update = false;
  }

  if (TaskListTriggers.con_hst.to_update) {
    TaskListTriggers.con_hst.next_time += TaskListTriggers.con_hst.dt;
    TaskListTriggers.con_hst.to_update = false;
  }

  if (TaskListTriggers.assert_is_finite.to_update) {
    TaskListTriggers.assert_is_finite.next_time += \
      TaskListTriggers.assert_is_finite.dt;
    TaskListTriggers.assert_is_finite.to_update = false;
  }

  if (TaskListTriggers.wave_extraction.to_update) {
    TaskListTriggers.wave_extraction.next_time += \
      TaskListTriggers.wave_extraction.dt;
    TaskListTriggers.wave_extraction.to_update = false;
  }

  if (TaskListTriggers.cce_dump.to_update) {
    TaskListTriggers.cce_dump.next_time += TaskListTriggers.cce_dump.dt;
    TaskListTriggers.cce_dump.to_update = false;
  }
}


//
// :D
//
