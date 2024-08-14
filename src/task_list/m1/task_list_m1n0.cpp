// C headers

// C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../scalars/scalars.hpp"
#include "../../m1/m1.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../task_list.hpp"
#include "task_list.hpp"

// ----------------------------------------------------------------------------
using namespace TaskLists::M1;
using namespace TaskLists::Integrators;
using namespace TaskNames::M1::M1N0;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

M1N0::M1N0(ParameterInput *pin, Mesh *pm, Triggers &trgs)
  : LowStorage(pin, pm),
    trgs(trgs)
{
  // Fix the number of stages based on internal M1+N0 method
  nstages = 2;

  // Now assemble list of tasks for each stage of time integrator
  {
    Add(UPDATE_BG, NONE, &M1N0::UpdateBackground);
    Add(CALC_FIDU, UPDATE_BG, &M1N0::CalcFiducialVelocity);
    Add(CALC_CLOSURE, CALC_FIDU, &M1N0::CalcClosure);

    // FIDU_FRAME is computed on-the-fly during the updates
    // Add(CALC_FIDU_FRAME, CALC_CLOSURE, &M1N0::CalcClosure);
    // Add(CALC_OPAC, CALC_FIDU_FRAME, &M1N0::CalcFiducialFrame);
    Add(CALC_OPAC, CALC_CLOSURE, &M1N0::CalcOpacity);

    Add(CALC_FLUX, CALC_OPAC, &M1N0::CalcFlux);

    Add(SEND_FLUX, CALC_FLUX, &M1N0::SendFlux);
    Add(RECV_FLUX, NONE, &M1N0::ReceiveAndCorrectFlux);

    Add(ADD_GRSRC, CALC_CLOSURE, &M1N0::AddGRSources);

    Add(ADD_FLX_DIV, (CALC_FLUX|ADD_GRSRC), &M1N0::AddFluxDivergence);

    Add(CALC_UPDATE, (CALC_OPAC|ADD_FLX_DIV), &M1N0::CalcUpdate);

    Add(SEND, CALC_UPDATE, &M1N0::Send);
    Add(RECV, CALC_UPDATE, &M1N0::Receive);

    Add(UPDATE_COUPLING, RECV, &M1N0::UpdateCoupling);

    Add(SETB, RECV, &M1N0::SetBoundaries);

    if (pm->multilevel) {
      Add(PROLONG, (SEND|SETB), &M1N0::Prolongation);
      Add(PHY_BVAL, PROLONG, &M1N0::PhysicalBoundary);
    }
    else {
      Add(PHY_BVAL, SETB, &M1N0::PhysicalBoundary);
    }

    Add(USERWORK, (PHY_BVAL|SEND_FLUX|CALC_UPDATE), &M1N0::UserWork);
    Add(NEW_DT, PHY_BVAL, &M1N0::NewBlockTimeStep);

    if (pm->adaptive)
    {
      Add(FLAG_AMR, USERWORK, &M1N0::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &M1N0::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &M1N0::ClearAllBoundary);
    }
  } // namespace
}

// ----------------------------------------------------------------------------
//! Initialize the task list
void M1N0::StartupTaskList(MeshBlock *pmb, int stage)
{
  if (stage == 1)
  {
    // u1 stores the solution at the beginning of the timestep
    pmb->pm1->storage.u1 = pmb->pm1->storage.u;
    // pmb->pm1->storage.u1.DeepCopy(pmb->pm1->storage.u);
  }

  // Clear the RHS
  pmb->pm1->storage.u_rhs.ZeroClear();
  pmb->pbval->StartReceiving(BoundaryCommSubset::m1);
  return;
}

// ----------------------------------------------------------------------------
//! Functions to end MPI communication
TaskStatus M1N0::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  pmb->pbval->ClearBoundary(BoundaryCommSubset::m1);
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Update external background / dynamical field states
TaskStatus M1N0::UpdateBackground(MeshBlock *pmb, int stage)
{
  // Task-list is not interspersed with external field evolution -
  // only need to do this on the first step.
  if (stage <= 1)
  {
    pmb->pm1->UpdateGeometry(pmb->pm1->geom, pmb->pm1->scratch);
    pmb->pm1->UpdateHydro(pmb->pm1->hydro, pmb->pm1->geom, pmb->pm1->scratch);
  }
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to Calculate Fiducial Velocity
TaskStatus M1N0::CalcFiducialVelocity(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->CalcFiducialVelocity();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to calculate Closure
TaskStatus M1N0::CalcClosure(MeshBlock *pmb, int stage)
{
  if (stage <= nstages) {
    pmb->pm1->CalcClosure(pmb->pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, P_dd) to (J, H_d)
TaskStatus M1N0::CalcFiducialFrame(MeshBlock *pmb, int stage)
{
  if (stage <= nstages) {
    pmb->pm1->CalcFiducialFrame(pmb->pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Function to calculate Opacities
TaskStatus M1N0::CalcOpacity(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    pmb->pm1->CalcOpacity(dt, pmb->pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::CalcFlux(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->CalcFluxes(pmb->pm1->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus M1N0::SendFlux(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->ubvar.SendFluxCorrection();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Receive fluxes between MeshBlocks
TaskStatus M1N0::ReceiveAndCorrectFlux(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    if (pmb->pm1->ubvar.ReceiveFluxCorrection())
    {
      return TaskStatus::next;
    }
    else {
      return TaskStatus::fail;
    }
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::AddFluxDivergence(MeshBlock *pmb, int stage)
{
  if (stage <= nstages) {
    pmb->pm1->AddFluxDivergence(pmb->pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::AddGRSources(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->AddSourceGR(pmb->pm1->storage.u,
                          pmb->pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Update the state vector
TaskStatus M1N0::CalcUpdate(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    pmb->pm1->CalcUpdate(dt,
                         pmb->pm1->storage.u1,
                         pmb->pm1->storage.u,
                         pmb->pm1->storage.u_rhs,
                         pmb->pm1->storage.u_sources);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Coupling to z4c_matter
TaskStatus M1N0::UpdateCoupling(MeshBlock *pmb, int stage)
{
  if (stage == nstages)
  {
    if (pmb->pm1->opt.couple_sources_hydro)
    {
      pmb->pm1->CoupleSourcesHydro(pmb->phydro->u);
    }

#if NSCALARS > 0
    if (pmb->pm1->opt.couple_sources_Y_e)
    {
      if (pmb->pm1->N_SPCS != 3)
      #pragma omp critical
      {
        std::cout << "M1: couple_sources_Y_e supported for 3 species \n";
        std::exit(0);
      }
      const Real mb = pmb->peos->GetEOS().GetRawBaryonMass();
      pmb->pm1->CoupleSourcesYe(mb, pmb->pscalars->s);
    }
#endif

    // N.B.
    // Conserved variables and ADM matter fields updated in
    // `Mesh::ScatterMatter`
    //
    // Source coupling in e.g. `CoupleSourcesHydro` acts on physical nodes
    // therefore further communication is required for consistency.
    return TaskStatus::next;
  } else {
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::Send(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->ubvar.SendBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


// ----------------------------------------------------------------------------
TaskStatus M1N0::Receive(MeshBlock *pmb, int stage)
{
  bool ret;
  if (stage <= nstages)
  {
    ret = pmb->pm1->ubvar.ReceiveBoundaryBuffers();
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

// ----------------------------------------------------------------------------
TaskStatus M1N0::SetBoundaries(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pm1->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::Prolongation(MeshBlock *pmb, int stage)
{
  BoundaryValues *pbval = pmb->pbval;
  if (stage <= nstages) {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    Real t_end_stage = pmb->pmy_mesh->time + dt;
    pbval->ProlongateBoundariesM1(t_end_stage, dt);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::PhysicalBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pbval = pmb->pbval;
  // Task-list vs M1 class (need :: prefix)
  ::M1::M1 *pm1 = pmb->pm1;
  Coordinates *pco = pmb->pcoord;

  if (stage <= nstages)
  {
    Real const dt = pmb->pmy_mesh->dt * dt_fac[stage - 1];
    Real t_end_stage = pmb->pmy_mesh->time + dt;

    pm1->enable_user_bc = true;
    pbval->ApplyPhysicalBoundariesM1(t_end_stage, dt);
    pm1->enable_user_bc = false;
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();

#if !Z4C_ENABLED
  // TODO: BD- this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();
#endif

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Determine the new timestep (used for the time adaptivity)
TaskStatus M1N0::NewBlockTimeStep(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pm1->NewBlockTimeStep();
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Flag cells for MeshBlocks (de)refinement
TaskStatus M1N0::CheckRefinement(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}
