//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
/////////////////////////////////// Athena++ Main Program
///////////////////////////////////
//! \file main.cpp
//  \brief Athena++ main program
//
// Based on the Athena MHD code (Cambridge version), originally written in
// 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important
// contributions by many other developers after that, i.e. 2005-2014.
//
// Athena++ was started in Jan 2014.  The core design was finished during
// 4-7/2014 at the KITP by Jim Stone.  GR was implemented by Chris White and
// AMR by Kengo Tomida during 2014-2016.  Contributions from many others have
// continued to the present.
//========================================================================================

// C headers

// C++ headers
#include <cmath>  // sqrt()
#include <csignal>  // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
#include <cstdint>    // int64_t
#include <cstdio>     // sscanf()
#include <cstdlib>    // strtol
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <exception>  // exception
#include <iomanip>    // setprecision()
#include <iostream>   // cout, endl
#include <limits>     // max_digits10
#include <new>        // bad_alloc
#include <string>     // string

// Athena++ headers
#include "defs.hpp"
#include "globals.hpp"
#include "main.hpp"
#include "main_triggers.hpp"

//-----------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Athena++ main program

int main(int argc, char* argv[])
{
  gra::Pathing Pathing;
  gra::Flags Flags;
  Flags.res   = 0;
  Flags.narg  = 0;
  Flags.iarg  = 0;
  Flags.mesh  = 0;
  Flags.wtlim = 0;

  std::uint64_t mbcnt = 0;

  //--- Step 1. ---------------------------------------------------------------
  // Initialize MPI environment, if necessary
  gra::parallelism::Init(argc, argv);
  gra::PrintRankZero("Running GR-Athena++ @ " + std::string(GIT_HASH));
  gra::PrintRankZero("Step 01: Parallel environment initialized.");

  //--- Step 2. ---------------------------------------------------------------
  // Check for command line options and respond.
  gra::PrintRankZero("Step 02: Parsing CMD-line...");
  gra::ParseCommandLine(argc, argv, &Pathing, &Flags);

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && Flags.wtlim > 0)
  {
    SignalHandler::SetWallTimeAlarm(Flags.wtlim);
  }

  //--- Step 3. ---------------------------------------------------------------
  // Construct object to store input parameters, then parse input file and
  // command line. With MPI, the input is read by every process in parallel
  // using MPI-IO.
  gra::PrintRankZero("Step 03: Parsing inputs...");

  ParameterInput* pinput = new ParameterInput;
  IOWrapper infile, restartfile;
  gra::ParseInputs(argc, argv, &Pathing, &Flags, pinput, infile, restartfile);

  //--- Step 4. ---------------------------------------------------------------
  // Construct and initialize Mesh
  gra::PrintRankZero("Step 04: Initializing Mesh...");
  Mesh* pmesh = gra::InitMesh(&Flags, pinput, restartfile);

  //--- Step 5. ---------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading
  // restart file
  gra::PrintRankZero("Step 05: Initializing Mesh data...");
  gra::InitMeshData(&Flags, pinput, pmesh);

  //--- Step 6. ---------------------------------------------------------------
  // Change to run directory, initialize outputs object, and make output of ICs
  gra::PrintRankZero("Step 06: Initializing Outputs...");
  Outputs* pouts = gra::InitOutputs(&Flags, &Pathing, pinput, pmesh);

  //--- Step 7. ---------------------------------------------------------------
  // Construct and initialize Triggers / TaskLists
  gra::PrintRankZero("Step 07: Initializing Triggers / Tasklists...");

  // could shift assembly to header, leave here now
  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  Triggers trgs(pmesh, pinput, pouts);
  const bool allow_rescale_dt =
    pinput->GetOrAddBoolean("task_triggers", "adjust_mesh_dt", true);

  // deal uniquely with hst
  trgs.Add(ovar::hst, true, allow_rescale_dt);

  trgs.Add(tvar::tracker_extrema, ovar::user, true, allow_rescale_dt);

#if defined(TWO_PUNCTURES)
  trgs.Add(tvar::Z4c_tracker_punctures, ovar::user, true, allow_rescale_dt);
#endif

  trgs.Add(tvar::Z4c_ADM_constraints, ovar::hst, true, allow_rescale_dt);
  trgs.Add(tvar::Z4c_ADM_constraints, ovar::data, true, allow_rescale_dt);

  trgs.Add(tvar::Z4c_Weyl, ovar::user, false, allow_rescale_dt);
  trgs.Add(tvar::Z4c_Weyl, ovar::data, true, allow_rescale_dt);

  trgs.Add(tvar::Z4c_AHF, ovar::user, true, allow_rescale_dt);

  trgs.Add(tvar::Z4c_RWZ, ovar::user, true, allow_rescale_dt);

#if CCE_ENABLED
  trgs.Add(tvar::Z4c_CCE, ovar::user, true, allow_rescale_dt);
#endif

#if defined(EJECTA_ENABLED)
  trgs.Add(tvar::ejecta, ovar::user, true, allow_rescale_dt);
#endif

  trgs.Add(tvar::global_extrema, ovar::user, true, allow_rescale_dt);

  gra::mesh::surfaces::InitSurfaceTriggers(trgs, pmesh->psurfs);

  // now populate requisite task-lists
  gra::tasklist::Collection ptlc{ trgs };
  gra::tasklist::PopulateCollection(ptlc, pmesh, pinput);

  //=== Step 8. === START OF MAIN INTEGRATION LOOP ============================
  // For performance, there is no error handler protecting this step
  // (except outputs)
  gra::PrintRankZero("Step 08: Entering main integration loop...");
  gra::timing::Clocks* pclk = new gra::timing::Clocks(pmesh);

  while ((pmesh->time < pmesh->tlim) &&
         (pmesh->nlim < 0 || pmesh->ncycle < pmesh->nlim))
  {
    // Adjust pmesh->dt if allowed by trigger, record if it happens.
    // This is to allow increase of pmesh->NewTimeStep to be unlimited & avoid
    // getting stuck
    bool mesh_dt_adjusted =
      (allow_rescale_dt) ? trgs.AdjustFromAny_mesh_dt() : false;

    // Any Mesh-level logic to be called prior to next iter of integration loop
    pmesh->UserWorkBeforeLoop(pinput);

    // Check user-enrolled break condition (e.g. post-bounce short-circuit)
    if (pmesh->CheckUserMainLoopBreak(pinput))
      break;

    if (Globals::my_rank == 0)
    {
      // pmesh->evo_rate = (
      //   pmesh->time - pmesh->start_time
      // ) / pclk->Elapsed_hours();
      // pmesh->evo_rate = std::isfinite(pmesh->evo_rate) ? pmesh->evo_rate :
      // 0;

      pmesh->evo_rate = pclk->evo_rate();
      pmesh->OutputCycleDiagnostics();
    }

    // Reset per-block dt to sentinel before any subsystem computes its CFL dt.
    // Each subsystem's NewBlockTimeStep() will min-reduce against this value.
    pmesh->ResetAllBlockDt();

    if (WAVE_ENABLED)
    {
      gra::evolve::Wave_2O(ptlc, pmesh);
    }

    if (Z4C_ENABLED)
    {
      if (FLUID_ENABLED)
      {
        if (M1_ENABLED &&
            pinput->GetOrAddBoolean("problem", "M1_enabled", true))
        {
          gra::evolve::Z4c_GRMHD_M1N0(ptlc, pmesh);
        }
        else
        {
          gra::evolve::Z4c_GRMHD(ptlc, pmesh);
        }
      }
      else
      {
        gra::evolve::Z4c_Vacuum(ptlc, pmesh);
      }

      gra::evolve::Z4c_DerivedQuantities(ptlc, trgs, pmesh, pouts);
    }
    else if (M1_ENABLED)
    {
      gra::evolve::M1N0(ptlc, pmesh);
    }

    gra::evolve::TrackerExtrema(ptlc, trgs, pmesh);
    const bool is_final = false;
    gra::evolve::SurfaceReductions(ptlc, trgs, pmesh, is_final);

    //-------------------------------------------------------------------------
    // Update triggers as required
    trgs.Update();
    //-------------------------------------------------------------------------

    pmesh->UserWorkInLoop(pinput);
    pmesh->ncycle++;
    pmesh->time += pmesh->dt;
    mbcnt += pmesh->nbtotal;
    pmesh->step_since_lb++;  // steps since load-balance

    pclk->Elapsed_pause();  // ignore regrid time in evo/hr estimate

    const Mesh::AMRStatus amr_status =
      pmesh->LoadBalancingAndAdaptiveMeshRefinement(pinput);

    // Any redistribution (AMR or LB-only)
    if (amr_status != Mesh::AMRStatus::unchanged)
    {
      pmesh->InitializePostMainUpdatedMesh(pinput);

      // surface interpolators hold stale MeshBlock* pointers and must be
      // rebuilt.
      for (auto psurf : pmesh->psurfs)
      {
        psurf->ReinitializeSurfaces(pmesh->ncycle, pmesh->time);
      }

      // GridThetaPhi pools hold stale MeshBlock* pointers; tear down so that
      // the next Calculate() call lazily rebuilds them.
      for (WaveExtractRWZ* prwz : pmesh->pwave_extr_rwz)
        prwz->grid_.TearDown();
#ifdef EJECTA_ENABLED
      for (Ejecta* pej : pmesh->pej_extract)
        pej->grid_.TearDown();
#endif
    }

    // Post-AMR pgen hook: only needed when mesh topology changed
    // (refinement/coarsening creates new blocks whose state the pgen hook
    // may need to customize).
    if (amr_status == Mesh::AMRStatus::refined)
    {
      pmesh->ApplyUserWorkMeshUpdatedPrePostAMRHooks(pinput);

      // Recompute Weyl scalars on new blocks for output consistency.
      // Weyl is AMR-registered (transferred during redistribution), so
      // only newly refined blocks (with prolongated data) need
      // recomputation.
      if (Z4C_ENABLED)
      {
        // TimeExceedsNextOutputTime uses substring matching so e.g.
        // geom.weyl* is matched
        if (pouts->TimeExceedsNextOutputTime("geom", pmesh->time) ||
            pouts->TimeExceedsNextOutputTime("hst", pmesh->time))
        {
          ptlc.postamr_z4c->DoTaskListOneStage(pmesh, 1);
        }
      }
    }

    // M1 derived quantities (fiducial-frame fields, opacities) are not
    // AMR-registered and are not recomputed by Initialize(regrid) /
    // FinalizeM1.  Recompute them after any redistribution so that
    // outputs on this cycle see consistent data.
    if (amr_status != Mesh::AMRStatus::unchanged)
    {
      if (M1_ENABLED && pinput->GetOrAddBoolean("problem", "M1_enabled", true))
      {
        if (pouts->TimeExceedsNextOutputTime("M1", pmesh->time) ||
            pouts->TimeExceedsNextOutputTime("hst", pmesh->time))
        {
          ptlc.postamr_m1n0->DoTaskListOneStage(pmesh, 1);
        }
      }
    }

    // Clear new_from_amr flags after any redistribution.
    if (amr_status != Mesh::AMRStatus::unchanged)
    {
      pmesh->FinalizePostAMR();
      // reset the timer to exclude recent regridding
      // pclk->evo_rate_reset();
    }

    pclk->Elapsed_resume();  // finish timer suspension

    // If a trigger adjusted pmesh->dt then do not limit dt rescaling.
    // Begin posts MPI_Iallreduce (non-blocking); the reduction overlaps
    // with file I/O below.  Finish waits for the result and clamps to tlim.
    pmesh->NewTimeStepBegin(!mesh_dt_adjusted);
    mesh_dt_adjusted = false;

    // Dump all the outputs ---------------------------------------------------
    // (MPI_Iallreduce for dt progresses in the background)
    if (pmesh->time < pmesh->tlim)
    {
      const bool is_final = false;
      gra::MakeOutputs(is_final, pinput, pmesh, pouts);
    }

    // Complete the non-blocking dt reduction before next cycle.
    pmesh->NewTimeStepFinish();

    // signals (i.e. -t) ------------------------------------------------------
    if (SignalHandler::CheckSignalFlags() != 0)
    {
      break;
    }

  }  // END OF MAIN INTEGRATION LOOP
     // ===========================================

  if (Globals::my_rank == 0 && Flags.wtlim > 0)
    SignalHandler::CancelWallTimeAlarm();

  // If the main loop broke due to a user hook that requested a tagged restart,
  // write it now (the pgen stored the tag in ParameterInput).
  if (pinput->DoesParameterExist("problem", "restart_tag"))
  {
    const std::string tag = pinput->GetString("problem", "restart_tag");
    // Clear before writing so the tag is NOT serialized into the restart file.
    // Otherwise restarting from the tagged file would produce another spurious
    // tagged restart on every subsequent exit.
    pinput->SetString("problem", "restart_tag", "");
    if (!tag.empty())
    {
      gra::MakeOutputRestart(pinput, pmesh, pouts, tag);
    }
  }

  //--- Step 9. --------------------------------------------------------------
  // Make the final outputs; post-loop work.
  gra::PrintRankZero("Step 09: Preparing final outputs...");

  {
    const bool is_final = true;
    gra::MakeOutputs(is_final, pinput, pmesh, pouts);
    gra::evolve::SurfaceReductions(ptlc, trgs, pmesh, is_final);
  }

  // BD: TODO - what is even the logic here?
#ifdef TWO_PUNCTURES
  // In case of two punctures this function has to be called only if not
  // restarting simulation
  if (!Flags.res)
    pmesh->UserWorkAfterLoop(pinput);
#else
  pmesh->UserWorkAfterLoop(pinput);
#endif

  //--- Step 10. --------------------------------------------------------------
  // Print diagnostic messages related to the end of the simulation
  gra::PrintRankZero("Step 10: Dumping diagnostic info...");
  gra::PrintDiagnostics(mbcnt, pclk, pmesh);

  //--- Step 11. --------------------------------------------------------------
  // Cleanup
  gra::PrintRankZero("Step 11: Cleanup...");

  delete pclk;
  delete pinput;
  delete pmesh;
  delete pouts;

  gra::tasklist::TearDown(ptlc);
  gra::parallelism::Teardown();

  return (0);
}
