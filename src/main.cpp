//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
/////////////////////////////////// Athena++ Main Program ////////////////////////////////
//! \file main.cpp
//  \brief Athena++ main program
//
// Based on the Athena MHD code (Cambridge version), originally written in 2002-2005 by
// Jim Stone, Tom Gardiner, and Peter Teuben, with many important contributions by many
// other developers after that, i.e. 2005-2014.
//
// Athena++ was started in Jan 2014.  The core design was finished during 4-7/2014 at the
// KITP by Jim Stone.  GR was implemented by Chris White and AMR by Kengo Tomida during
// 2014-2016.  Contributions from many others have continued to the present.
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <csignal>    // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
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
#include "main.hpp"
#include "main_triggers.hpp"

#include "z4c/wave_extract.hpp"
#include "z4c/puncture_tracker.hpp"
#include "trackers/extrema_tracker.hpp"
#include "z4c/ahf.hpp"
// #include "z4c/ejecta.hpp"
#if CCE_ENABLED
#include "z4c/cce/cce.hpp"
#endif

//-----------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Athena++ main program

int main(int argc, char *argv[])
{
  gra::Pathing Pathing;
  gra::Flags Flags;
  Flags.res  = 0;
  Flags.narg = 0;
  Flags.iarg = 0;
  Flags.mesh = 0;
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

  ParameterInput *pinput = new ParameterInput;
  IOWrapper infile, restartfile;
  gra::ParseInputs(argc, argv, &Pathing, &Flags, pinput, infile, restartfile);

  //--- Step 4. ---------------------------------------------------------------
  // Construct and initialize Mesh
  gra::PrintRankZero("Step 04: Initializing Mesh...");
  Mesh *pmesh = gra::InitMesh(&Flags, pinput, restartfile);

  //--- Step 5. ---------------------------------------------------------------
  // Set initial conditions by calling problem generator, or reading
  // restart file
  gra::PrintRankZero("Step 05: Initializing Mesh data...");
  gra::InitMeshData(&Flags, pinput, pmesh);

  //--- Step 6. ---------------------------------------------------------------
  // Change to run directory, initialize outputs object, and make output of ICs
  gra::PrintRankZero("Step 06: Initializing Outputs...");
  Outputs *pouts = gra::InitOutputs(&Flags, &Pathing, pinput, pmesh);

  //--- Step 7. ---------------------------------------------------------------
  // Construct and initialize Triggers / TaskLists
  gra::PrintRankZero("Step 07: Initializing Triggers / Tasklists...");

  // could shift assembly to header, leave here now
  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  Triggers trgs(pmesh, pinput, pouts);
  trgs.Add(tvar::tracker_extrema,     ovar::user, true, true);

  trgs.Add(tvar::Z4c_ADM_constraints, ovar::hst,  true, true);
  trgs.Add(tvar::Z4c_ADM_constraints, ovar::data, true, true);

  trgs.Add(tvar::Z4c_Weyl, ovar::user, true, true);
  trgs.Add(tvar::Z4c_Weyl, ovar::data, true, true);

  // BD: TODO - move to collection etc.
  WaveIntegratorTaskList *pwlist = nullptr;
  M1IntegratorTaskList *pm1list = nullptr;

  // now populate requisite task-lists
  gra::tasklist::Collection ptlc { trgs };
  gra::tasklist::PopulateCollection(ptlc, pmesh, pinput);

  // BD: TODO - move to collection
  if (M1_ENABLED)
  {
    pm1list = new M1IntegratorTaskList(pinput, pmesh);
  }

  //=== Step 8. === START OF MAIN INTEGRATION LOOP ============================
  // For performance, there is no error handler protecting this step
  // (except outputs)
  gra::PrintRankZero("Step 08: Entering main integration loop...");
  gra::timing::Clocks * pclk = new gra::timing::Clocks();

  const bool trgs_can_adjust_mesh_dt = pinput->GetOrAddBoolean(
    "task_triggers", "adjust_mesh_dt", true
  );

  while ((pmesh->time < pmesh->tlim) &&
         (pmesh->nlim < 0 || pmesh->ncycle < pmesh->nlim))
  {

    // Adjust pmesh->dt if allowed by trigger, record if it happens.
    // This is to allow increase of pmesh->NewTimeStep to be unlimited & avoid
    // getting stuck
    bool mesh_dt_adjusted = (trgs_can_adjust_mesh_dt)
      ? trgs.AdjustFromAny_mesh_dt()
      : false;

    // After state vector propagated, derived diagnostics (i.e. GW, trackers)
    // are at the new time-step ...
    const Real time_end_stage   = pmesh->time+pmesh->dt;
    const Real ncycle_end_stage = pmesh->ncycle+1;

    if (Globals::my_rank == 0)
    {
      pmesh->OutputCycleDiagnostics();
    }

    if (WAVE_ENABLED)
    {
      for (int stage=1; stage<=pwlist->nstages; ++stage)
      {
        pwlist->DoTaskListOneStage(pmesh, stage);
      }
    }

    if (Z4C_ENABLED)
    {
      if (!FLUID_ENABLED)
      {
        for (int stage=1; stage<=ptlc.gr_z4c->nstages; ++stage)
        {
          ptlc.gr_z4c->DoTaskListOneStage(pmesh, stage);
          // Iterate bnd comm. as required
          pmesh->CommunicateIteratedZ4c(Z4C_CX_NUM_RBC);
        }
      }
      else
      { //FLUID_ENABLED && Z4C_ENABLED
        for (int stage=1; stage<=ptlc.grmhd_z4c->nstages; ++stage)
        {
          ptlc.grmhd_z4c->DoTaskListOneStage(pmesh, stage);
          // Iterate bnd comm. as required
          pmesh->CommunicateIteratedZ4c(Z4C_CX_NUM_RBC);
        }

      } //FLUID_ENABLED

      // Auxiliary variable logic
      // Currently this handles Weyl communication & decomposition
      if (trgs.IsSatisfied(tvar::Z4c_Weyl))
      {
        // May be required to prevent task-list overlaps
        // gra::parallelism::Barrier();
        pmesh->CommunicateAuxZ4c();
        ptlc.aux_z4c->DoTaskListOneStage(pmesh, 1);  // only 1 stage
      }

      if (trgs.IsSatisfied(tvar::Z4c_Weyl, ovar::user))
      {
        for (auto pwextr : pmesh->pwave_extr)
        {
          pwextr->ReduceMultipole();
          pwextr->Write(ncycle_end_stage, time_end_stage);
        }
      }

// BD: TODO - CCE needs to be cleaned up & tested
#if CCE_ENABLED
      bool cce_update;
      Real cce_dt;

      if (!FLUID_ENABLED)
      {
        cce_update = ptlc.gr_z4c->TaskListTriggers.cce_dump.to_update;
        cce_dt = ptlc.gr_z4c->TaskListTriggers.cce_dump.dt;
      } else {
        cce_update = ptlc.grmhd_z4c->TaskListTriggers.cce_dump.to_update;
        cce_dt = ptlc.grmhd_z4c->TaskListTriggers.cce_dump.dt;
      }
      // only do a CCE dump if NextTime threshold cleared (updated below)
      if (cce_update) {
        // gather all interpolation values from all processors to the root proc.
        for (auto cce : pmesh->pcce)
        {
          cce->ReduceInterpolation();
        }
        // number of cce iteration(dump)
        int cce_iter = static_cast<int>(pmesh->time / 
                                       cce_dt);
        int w_iter = 0; // write iter

        // update the bookkeeping file to ensure resuming from the correct iter number 
        // after a restart. note: for a duplicated iter, hdf5 writer gives an error!
        if (CCE::BookKeeping(pinput,cce_iter,w_iter))
        {
          for (auto cce : pmesh->pcce)
          {
            cce->DecomposeAndWrite(w_iter);
          }
        }
      }
#endif

      for (auto pah_f : pmesh->pah_finder)
      {
        if (pah_f->CalculateMetricDerivatives(ncycle_end_stage,
                                              time_end_stage))
        {
          break;
        }
      }
      for (auto pah_f : pmesh->pah_finder)
      {
        pah_f->Find(ncycle_end_stage, time_end_stage);
        pah_f->Write(ncycle_end_stage, time_end_stage);
      }
      for (auto pah_f : pmesh->pah_finder)
      {
        if (pah_f->DeleteMetricDerivatives(ncycle_end_stage, time_end_stage))
        {
          break;
        }
      }
/* WC: Temporarily remove ejecta - to be tested/reincluded
      for (auto pej : pmesh->pej_extract) {
        pej->Calculate(pmesh->ncycle, pmesh->time);
      }
*/
      for (auto ptracker : pmesh->pz4c_tracker)
      {
        ptracker->EvolveTracker();
      }

      if (trgs.IsSatisfied(tvar::Z4c_tracker_punctures, ovar::user))
      for (auto ptracker : pmesh->pz4c_tracker)
      {
        ptracker->WriteTracker(ncycle_end_stage, time_end_stage);
      }

    }

    // Trace AMR state
    pmesh->ptracker_extrema->ReduceTracker();
    pmesh->ptracker_extrema->EvolveTracker();

    if (trgs.IsSatisfied(tvar::tracker_extrema, ovar::user))
    {
      pmesh->ptracker_extrema->WriteTracker(ncycle_end_stage, time_end_stage);
    }

    //-------------------------------------------------------------------------
    // Update triggers as required
    trgs.Update();
    //-------------------------------------------------------------------------

    // BD: TODO - shift to correct place
    // BD: Shift to Evolve:: style header as previously
    if (M1_ENABLED)
    {
      for (int stage=1; stage<=pm1list->nstages; ++stage)
      {
        pm1list->DoTaskListOneStage(pmesh, stage);
      }
    }

    pmesh->UserWorkInLoop();
    pmesh->ncycle++;
    pmesh->time += pmesh->dt;
    mbcnt += pmesh->nbtotal;
    pmesh->step_since_lb++;   // steps since load-balance

    bool mesh_updated = pmesh->LoadBalancingAndAdaptiveMeshRefinement(pinput);

    // Post AMR hook;
    // While state vectors are suitably populated on the blocks, other derived
    // quantities such as e.g. ADM constraints, Weyl scalar are not.
    //
    // Some quantities, e.g. ADM constraints also _should not_ be interpolated.
    //
    // To rectify, we add a final task-list dealing with such quantities before
    // output.

    if (mesh_updated)
    {
      // N.B.
      // At this stage Mesh::Initialize(2, pinputs) has been called
      if (Z4C_ENABLED)
      {
        ptlc.postamr_z4c->DoTaskListOneStage(pmesh, 1);  // only 1 stage
      }

      pmesh->FinalizePostAMR();
    }

    // If a trigger adjusted pmesh->dt then do not limit dt rescaling
    pmesh->NewTimeStep(!mesh_dt_adjusted);
    mesh_dt_adjusted = false;

    // Dump all the outputs ---------------------------------------------------
    if (pmesh->time < pmesh->tlim)
    {
      const bool is_final = false;
      gra::MakeOutputs(is_final, pinput, pmesh, pouts);
    }

    // signals (i.e. -t) ------------------------------------------------------
    if (SignalHandler::CheckSignalFlags() != 0)
    {
      break;
    }

  } // END OF MAIN INTEGRATION LOOP ===========================================

  if (Globals::my_rank == 0 && Flags.wtlim > 0)
    SignalHandler::CancelWallTimeAlarm();

  //--- Step 9. --------------------------------------------------------------
  // Make the final outputs; post-loop work.
  gra::PrintRankZero("Step 09: Preparing final outputs...");

  {
    const bool is_final = true;
    gra::MakeOutputs(is_final, pinput, pmesh, pouts);
  }

  // BD: TODO - what is even the logic here?
#ifdef TWO_PUNCTURES
  // In case of two punctures this function has to be called only if not
  // restarting simulation
  if (!Flags.res) pmesh->UserWorkAfterLoop(pinput);
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

  // BD: TODO - move to TearDown
  if (M1_ENABLED)
  {
    delete pm1list;
  }

  gra::tasklist::TearDown(ptlc);
  gra::parallelism::Teardown();

  return(0);
}
