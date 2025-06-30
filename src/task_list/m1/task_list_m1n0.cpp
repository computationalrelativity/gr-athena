// C headers

// C++ headers
#include <cstdio>
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
#include "../../z4c/z4c.hpp"
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

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  // Now assemble list of tasks for each stage of time integrator
  {
    Add(UPDATE_BG,    NONE,      &M1N0::UpdateBackground);
    Add(CALC_FIDU,    UPDATE_BG, &M1N0::CalcFiducialVelocity);
    Add(CALC_CLOSURE, CALC_FIDU, &M1N0::CalcClosure);

    // Closure is not guaranteed to compute fiducial frame quantities;
    // these are required for e.g. rad.sc_n from lab.sc_nG after closure.
    // This example enters weak-rates in opacities.
    Add(CALC_FIDU_FRAME, CALC_CLOSURE, &M1N0::CalcFiducialFrame);
    Add(CALC_OPAC, CALC_FIDU_FRAME, &M1N0::CalcOpacity);


    Add(ADD_GRSRC,   CALC_CLOSURE, &M1N0::AddGRSources);
    Add(CALC_FLUX, (CALC_OPAC|ADD_GRSRC), &M1N0::CalcFlux);

    Add(ADD_FLX_DIV, CALC_FLUX,
                     &M1N0::AddFluxDivergence);

    if (multilevel)
    {
      Add(SEND_FLUX, CALC_FLUX, &M1N0::SendFluxCorrection);
      Add(RECV_FLUX, CALC_FLUX, &M1N0::ReceiveAndCorrectFlux);
      Add(CALC_UPDATE, (RECV_FLUX|ADD_FLX_DIV),
                       &M1N0::CalcUpdate);

    }
    else
    {
      Add(CALC_UPDATE, ADD_FLX_DIV, &M1N0::CalcUpdate);
    }

    Add(SEND, CALC_UPDATE, &M1N0::SendM1);
    Add(RECV, CALC_UPDATE, &M1N0::ReceiveM1);

    Add(SETB, RECV, &M1N0::SetBoundaries);

    Add(UPDATE_COUPLING, SETB, &M1N0::UpdateCoupling);

    if (multilevel)
    {
      Add(PROLONG,  SETB,    &M1N0::Prolongation);
      Add(PHY_BVAL, PROLONG, &M1N0::PhysicalBoundary);
    }
    else
    {
      Add(PHY_BVAL, SETB, &M1N0::PhysicalBoundary);
    }

    Add(ANALYSIS, PHY_BVAL, &M1N0::Analysis);

    Add(USERWORK, ANALYSIS, &M1N0::UserWork);
    Add(NEW_DT,   USERWORK, &M1N0::NewBlockTimeStep);

    if (adaptive)
    {
      Add(FLAG_AMR,     USERWORK, &M1N0::CheckRefinement);
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
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage == 1)
  {
    pm1->ResetEvolutionStrategy();

    // u1 stores the solution at the beginning of the timestep
    pm1->storage.u1 = pm1->storage.u;
    // pm1->storage.u1.DeepCopy(pm1->storage.u);
  }

  // Clear the RHS
  pm1->storage.u_rhs.ZeroClear();
  pmb->pbval->StartReceiving(BoundaryCommSubset::m1);
  return;
}

// ----------------------------------------------------------------------------
//! Functions to end MPI communication
TaskStatus M1N0::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  Mesh * pm = pmb->pmy_mesh;
  ::M1::M1 * pm1 = pmb->pm1;

  pmb->pbval->ClearBoundary(BoundaryCommSubset::m1);
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Update external background / dynamical field states
TaskStatus M1N0::UpdateBackground(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  // Task-list is not interspersed with external field evolution -
  // only need to do this on the first step.
  if (stage == 1)
  {
    pm1->UpdateGeometry(pm1->geom, pm1->scratch);
    pm1->UpdateHydro(pm1->hydro, pm1->geom, pm1->scratch);
  }

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to Calculate Fiducial Velocity
TaskStatus M1N0::CalcFiducialVelocity(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  // Task-list is not interspersed with external field evolution -
  // only need to do this on the first step.
  if (stage == 1)
  {
    pm1->CalcFiducialVelocity();
  }

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to calculate Closure
TaskStatus M1N0::CalcClosure(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    pm1->CalcClosure(pm1->storage.u);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Map (closed) Eulerian fields (E, F_d, nG) to (J, H_d, n)
// Needed on first step for weak-rates
// Use frame in flux calculation
TaskStatus M1N0::CalcFiducialFrame(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  // if (stage == 1)
  if (stage <= nstages)
  {
    pm1->CalcFiducialFrame(pm1->storage.u);
  }
  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
// Function to calculate Opacities
TaskStatus M1N0::CalcOpacity(MeshBlock *pmb, int stage)
{
  Mesh * pm = pmb->pmy_mesh;
  ::M1::M1 * pm1 = pmb->pm1;

  // opacities are kept fixed throughout the implicit time integration
  // if (stage <= nstages)
  if (stage == 1)
  {
    Real const dt = pm->dt;
    // Real const dt = pm->dt * dt_fac[stage - 1];
    pm1->CalcOpacity(dt, pm1->storage.u);
    return TaskStatus::success;
  }
  else if (stage <= nstages)
  {
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::CalcFlux(MeshBlock *pmb, int stage)
{
  Mesh * pm = pmb->pmy_mesh;
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    if (pm1->opt.flux_limiter_multicomponent)
    {
      pm1->CalcFluxLimiter(pm1->storage.u);
    }

    pm1->CalcFluxes(pm1->storage.u, false);

    if (pm1->opt.flux_lo_fallback)
    {
      pm1->CalcFluxes(pm1->storage.u, true);

      // Zero property preservation mask
      // Do prior to evo. as mask is modified on pp enforcement.
      pm1->ev_strat.masks.pp.Fill(0.0);

      // construct candidate solution -----------------------------------------
      // need to add divF to inhomogeneity; subtract off after solution known

      pm1->AddFluxDivergence(pm1->storage.u_rhs);

      Real const dt = pm->dt * dt_fac[stage - 1];

      if (stage == 1)
      {
        pm1->PrepareEvolutionStrategy(dt);
      }

      // Construct candidate state
      pm1->CalcUpdate(stage,
                      dt,
                      pm1->storage.u1,
                      pm1->storage.u,
                      pm1->storage.u_rhs,
                      pm1->storage.u_sources);

      // Update status
      M1_MLOOP3(k, j, i)
      {
        const Real is_lo = pm1->ev_strat.masks.pp(k, j, i) == 1.0;
        pm1->ev_strat.status.num_lo_reversions += is_lo;
      }
      // Revert inhomogeneity
      // WARNING: masks have been adjusted within CalcUpdate; to properly
      // compensate flux addition we should not use the hybridization mask
      // there.
      pm1->SubFluxDivergence(pm1->storage.u_rhs);

      // hybridize fluxes based on pp mask ------------------------------------
      pm1->HybridizeLOFlux(pm1->storage.u);
      // Flip mask to execute next CalcUpdate on LO points
      pm1->AdjustMaskPropertyPreservation();
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus M1N0::SendFluxCorrection(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    pm1->ubvar.SendFluxCorrection();
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Receive fluxes between MeshBlocks
TaskStatus M1N0::ReceiveAndCorrectFlux(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    if (pm1->ubvar.ReceiveFluxCorrection())
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
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    pm1->AddFluxDivergence(pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::AddGRSources(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    pm1->AddSourceGR(pm1->storage.u, pm1->storage.u_rhs);
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Update the state vector
TaskStatus M1N0::CalcUpdate(MeshBlock *pmb, int stage)
{
  Mesh * pm = pmb->pmy_mesh;
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    Real const dt = pm->dt * dt_fac[stage - 1];

    if ((stage == 1) && (!pm1->opt.flux_lo_fallback))
    {
      pm1->PrepareEvolutionStrategy(dt);
    }


    // pm1->opt.flux_lo_fallback = false;
    pm1->CalcUpdate(stage,
                    dt,
                    pm1->storage.u1,
                    pm1->storage.u,
                    pm1->storage.u_rhs,
                    pm1->storage.u_sources);

    // pm1->opt.flux_lo_fallback = true;

    // if (stage == 2)
    // {
    //   pm1->ResetEvolutionStrategy();
    // }

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
// Coupling to z4c_matter
TaskStatus M1N0::UpdateCoupling(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage == nstages)
  {
    if (pm1->opt.zero_fix_sources)
    {
      pm1->EnforceSourcesFinite();
    }

    if (pm1->opt.couple_sources_hydro)
    {
      pm1->CoupleSourcesHydro(pmb->phydro->u);
    }

#if NSCALARS > 0
    if (pm1->opt.couple_sources_Y_e)
    {
      if (pm1->N_SPCS != 3)
      #pragma omp critical
      {
        std::cout << "M1: couple_sources_Y_e supported for 3 species \n";
        std::exit(0);
      }
      // const Real mb = pmb->peos->GetEOS().GetBaryonMass();
      const Real mb = pmb->peos->GetEOS().GetRawBaryonMass();
      pm1->CoupleSourcesYe(mb, pmb->pscalars->s);
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
TaskStatus M1N0::SendM1(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;
  if (stage <= nstages)
  {
    pm1->ubvar.SendBoundaryBuffers();
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


// ----------------------------------------------------------------------------
TaskStatus M1N0::ReceiveM1(MeshBlock *pmb, int stage)
{
  ::M1::M1 * pm1 = pmb->pm1;

  bool ret;
  if (stage <= nstages)
  {
    ret = pm1->ubvar.ReceiveBoundaryBuffers();
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
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage <= nstages)
  {
    pm1->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::Prolongation(MeshBlock *pmb, int stage)
{
  BoundaryValues *pbval = pmb->pbval;
  Mesh * pm = pmb->pmy_mesh;

  if (stage <= nstages) {
    Real const dt = pm->dt * dt_fac[stage - 1];
    Real t_end_stage = pm->time + dt;
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

    pbval->ApplyPhysicalBoundaries(
      t_end_stage, dt,
      pbval->GetBvarsM1(),
      pm1->mbi.il, pm1->mbi.iu,
      pm1->mbi.jl, pm1->mbi.ju,
      pm1->mbi.kl, pm1->mbi.ku,
      pm1->mbi.ng);

    pm1->enable_user_bc = false;
    return TaskStatus::success;
  }

  return TaskStatus::fail;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::Analysis(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pm1->PerformAnalysis();

  return TaskStatus::success;
}

// ----------------------------------------------------------------------------
TaskStatus M1N0::UserWork(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->M1UserWorkInLoop();

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
  ::M1::M1 * pm1 = pmb->pm1;

  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pm1->NewBlockTimeStep();
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
