//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_task_list.cpp
//  \brief time integrator for the z4c system (based on time_integrator.cpp)

// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../z4c/z4c.hpp"
#include "../z4c/wave_extract.hpp"
#include "../z4c/puncture_tracker.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"

// BD TODO: Significant code duplication with time_integrator, leave decoupled

//----------------------------------------------------------------------------------------
//  Z4cIntegratorTaskList constructor

Z4cIntegratorTaskList::Z4cIntegratorTaskList(ParameterInput *pin, Mesh *pm){
  //! \note
  //! First, define each time-integrator by setting weights for each step of
  //! the algorithm and the CFL number stability limit when coupled to the single-stage
  //! spatial operator.
  //! Currently, the explicit, multistage time-integrators must be expressed as 2S-type
  //! algorithms as in Ketcheson (2010) Algorithm 3, which incudes 2N (Williamson) and 2R
  //! (van der Houwen) popular 2-register low-storage RK methods. The 2S-type integrators
  //! depend on a bidiagonally sparse Shu-Osher representation; at each stage l:
  //! \f[
  //!   U^{l} = a_{l,l-2}*U^{l-2} + a_{l-1}*U^{l-1}
  //!         + b_{l,l-2}*dt*Div(F_{l-2}) + b_{l,l-1}*dt*Div(F_{l-1}),
  //! \f]
  //! where \f$U^{l-1}\f$ and \f$U^{l-2}\f$ are previous stages and
  //! \f$a_{l,l-2}\f$, \f$a_{l,l-1}=(1-a_{l,l-2})\f$,
  //! and \f$b_{l,l-2}\f$, \f$b_{l,l-1}\f$
  //! are weights that are different for each stage and
  //! integrator. Previous timestep \f$U^{0} = U^n\f$ is given, and the integrator solves
  //! for \f$U^{l}\f$ for 1 <= l <= nstages.
  //!
  //! \note
  //! The 2x RHS evaluations of Div(F) and source terms per stage is avoided by adding
  //! another weighted average / caching of these terms each stage. The API and framework
  //! is extensible to three register 3S* methods,
  //! although none are currently implemented.
  //!
  //! \note
  //! Notation: exclusively using "stage", equivalent in lit. to "substage" or "substep"
  //! (infrequently "step"), to refer to the intermediate values of U^{l} between each
  //! "timestep" = "cycle" in explicit, multistage methods. This is to disambiguate the
  //! temporal integration from other iterative sequences;  generic
  //! "Step" is often used for sequences in code, e.g. main.cpp: "Step 1: MPI"
  //!
  //! \note
  //! main.cpp invokes the tasklist in a for () loop from stage=1 to stage=ptlist->nstages
  //!
  //! \todo (felker):
  //! - validate Field and Hydro diffusion with RK3, RK4, SSPRK(5,4)
  integrator = pin->GetOrAddString("time", "integrator", "vl2");

  if (integrator == "vl2") {
    //! \note `integrator == "vl2"`
    //! - VL: second-order van Leer integrator (Stone & Gardiner, NewA 14, 139 2009)
    //! - Simple predictor-corrector scheme similar to MUSCL-Hancock
    //! - Expressed in 2S or 3S* algorithm form

    // set number of stages and time coeff.
    nstages_main = 2;
    nstages = nstages_main;
    stage_wghts[0].sbeta = 0.0;
    stage_wghts[0].ebeta = 0.5;
    stage_wghts[1].sbeta = 0.5;
    stage_wghts[1].ebeta = 1.0;
    stage_wghts[0].beta = 0.5;
    stage_wghts[1].beta = 1.0;
    cfl_limit = 1.0;
    // Modify VL2 stability limit in 2D, 3D
    if (pm->ndim == 2) cfl_limit = 0.5;
    if (pm->ndim == 3) cfl_limit = 0.5;

    // set delta and gamma at each stage
    int n_main = 0;
    for (int n=0; n<nstages; n++) {
      if (n_main == 0) {
        stage_wghts[n].delta = 1.0; // required for consistency
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 1) {
        stage_wghts[n].delta = 0.0;
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      }
    }
  } else if (integrator == "rk1") {
    //! \note `integrator == "rk1"`
    //! - RK1: first-order Runge-Kutta / the forward Euler (FE) method

    // set number of stages and time coeff.
    nstages_main = 1;
    nstages = nstages_main;
    stage_wghts[0].sbeta = 0.0;
    stage_wghts[0].ebeta = 1.0;
    stage_wghts[0].beta = 1.0;
    cfl_limit = 1.0;

    // set delta and gamma at each stage
    int n_main = 0;
    for (int n=0; n<nstages; n++) {
      if (n_main == 0) {
        stage_wghts[n].delta = 1.0;
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      }
    }
  } else if (integrator == "rk2") {
    //! \note `integrator == "rk2"`
    //! - Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
    //! - Optimal (in error bounds) explicit two-stage, second-order SSPRK

    // set number of stages and time coeff.
    nstages_main = 2;
    nstages = nstages_main;
    stage_wghts[0].sbeta = 0.0;
    stage_wghts[0].ebeta = 1.0;
    stage_wghts[1].sbeta = 1.0;
    stage_wghts[1].ebeta = 1.0;
    stage_wghts[0].beta = 1.0;
    stage_wghts[1].beta = 0.5;
    cfl_limit = 1.0;  // c_eff = c/nstages = 1/2 (Gottlieb (2009), pg 271)

    // set delta and gamma at each stage
    int n_main = 0;
    for (int n=0; n<nstages; n++) {
      if (n_main == 0) {
        stage_wghts[n].delta = 1.0;
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 1) {
        stage_wghts[n].delta = 0.0;
        stage_wghts[n].gamma_1 = 0.5;
        stage_wghts[n].gamma_2 = 0.5;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      }
    }
  } else if (integrator == "rk3") {
    //! \note `integrator == "rk3"`
    //! - SSPRK (3,3): Gottlieb (2009) equation 3.2
    //! - Optimal (in error bounds) explicit three-stage, third-order SSPRK

    // set number of stages and time coeff.
    nstages_main = 3;
    nstages = nstages_main;
    stage_wghts[0].sbeta = 0.0;
    stage_wghts[0].ebeta = 1.0;
    stage_wghts[1].sbeta = 1.0;
    stage_wghts[1].ebeta = 0.5;
    stage_wghts[2].sbeta = 0.5;
    stage_wghts[2].ebeta = 1.0;
    stage_wghts[0].beta = 1.0;
    stage_wghts[1].beta = 0.25;
    stage_wghts[2].beta = TWO_3RD;
    cfl_limit = 1.0;  // c_eff = c/nstages = 1/3 (Gottlieb (2009), pg 271)

    // set delta and gamma at each stage
    int n_main = 0;
    for (int n=0; n<nstages; n++) {
      if (n_main == 0) {
        stage_wghts[n].delta = 1.0;
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 1) {
        stage_wghts[n].delta = 0.0;
        stage_wghts[n].gamma_1 = 0.25;
        stage_wghts[n].gamma_2 = 0.75;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 2) {
        stage_wghts[n].delta = 0.0;
        stage_wghts[n].gamma_1 = TWO_3RD;
        stage_wghts[n].gamma_2 = ONE_3RD;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      }
    }
  } else if (integrator == "rk4") {
    //! \note `integorator == "rk4"`
    //! - RK4()4[2S] from Table 2 of Ketcheson (2010)
    //! - Non-SSP, explicit four-stage, fourth-order RK
    //! - Stability properties are similar to classical (non-SSP) RK4
    //!   (but ~2x L2 principal error norm).
    //! - Refer to Colella (2011) for linear stability analysis of constant
    //!   coeff. advection of classical RK4 + 4th or 1st order (limiter engaged) fluxes
    nstages_main = 4;
    cfl_limit = 1.3925; // Colella (2011) eq 101; 1st order flux is most severe constraint

    nstages = nstages_main;
    stage_wghts[0].beta = 1.193743905974738;
    stage_wghts[1].beta = 0.099279895495783;
    stage_wghts[2].beta = 1.131678018054042;
    stage_wghts[3].beta = 0.310665766509336;

    // set delta and gamma at each stage
    int n_main = 0;
    for (int n=0; n<nstages; n++) {
      if (n_main == 0) {
        stage_wghts[n].delta = 1.0;
        stage_wghts[n].gamma_1 = 0.0;
        stage_wghts[n].gamma_2 = 1.0;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 1) {
        stage_wghts[n].delta = 0.217683334308543;
        stage_wghts[n].gamma_1 = 0.121098479554482;
        stage_wghts[n].gamma_2 = 0.721781678111411;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 2) {
        stage_wghts[n].delta = 1.065841341361089;
        stage_wghts[n].gamma_1 = -3.843833699660025;
        stage_wghts[n].gamma_2 = 2.121209265338722;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      } else if (n_main == 3) {
        stage_wghts[n].delta = 0.0;
        stage_wghts[n].gamma_1 = 0.546370891121863;
        stage_wghts[n].gamma_2 = 0.198653035682705;
        stage_wghts[n].gamma_3 = 0.0;
        n_main++;
      }
    }

    // set sbeta & ebeta
    Real temp = 0.0;
    Real temp_prev = 0.0;
    stage_wghts[0].sbeta = 0.0;
    for (int l=0; l<nstages-1; l++) {
      temp_prev = temp;
      temp = temp_prev + stage_wghts[l].delta*stage_wghts[l].sbeta;
      stage_wghts[l].ebeta = stage_wghts[l].gamma_1*temp_prev
                             + stage_wghts[l].gamma_2*temp
                             + stage_wghts[l].gamma_3*0.0
                             + stage_wghts[l].beta;
      stage_wghts[l+1].sbeta = stage_wghts[l].ebeta;
    }
    stage_wghts[nstages-1].ebeta = 1.0;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in TimeIntegratorTaskList constructor" << std::endl
        << "integrator=" << integrator << " not valid time integrator" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Set cfl_number based on user input and time integrator CFL limit
  Real cfl_number = pin->GetReal("time", "cfl_number");
  if (cfl_number > cfl_limit
      && pm->fluid_setup == FluidFormulation::evolve) {
    std::cout << "### Warning in TimeIntegratorTaskList constructor" << std::endl
              << "User CFL number " << cfl_number << " must be smaller than " << cfl_limit
              << " for integrator=" << integrator << " in " << pm->ndim
              << "D simulation" << std::endl << "Setting to limit" << std::endl;
    cfl_number = cfl_limit;
  }
  // Save to Mesh class
  pm->cfl_number = cfl_number;

  //---------------------------------------------------------------------------
  // Output frequency control (on task-list)
  TaskListTriggers.assert_is_finite.next_time = pm->time;
  TaskListTriggers.assert_is_finite.dt = pin->GetOrAddReal("z4c",
    "dt_assert_is_finite", 0.0);

  // For constraint calculation
  TaskListTriggers.con.next_time = pm->time;
  // Seed TaskListTriggers.con.dt in main

  TaskListTriggers.wave_extraction.dt = pin->GetOrAddReal("z4c", "dt_wave_extraction", 1.0);
  if (pin->GetOrAddInteger("z4c", "nrad_wave_extraction", 0) == 0) {
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
  //---------------------------------------------------------------------------

  // Now assemble list of tasks for each stage of z4c integrator
  {using namespace Z4cIntegratorTaskNames;
    AddTask(CALC_Z4CRHS, NONE);                // CalculateZ4cRHS
    AddTask(INT_Z4C, CALC_Z4CRHS);             // IntegrateZ4c

    AddTask(SEND_Z4C, INT_Z4C);                // SendZ4c
    AddTask(RECV_Z4C, NONE);                   // ReceiveZ4c

    AddTask(SETB_Z4C, (RECV_Z4C|INT_Z4C));     // SetBoundariesZ4c
    if (pm->multilevel) { // SMR or AMR
      AddTask(PROLONG, (SEND_Z4C|SETB_Z4C));   // Prolongation
      AddTask(PHY_BVAL, PROLONG);              // PhysicalBoundary
    } else {
      AddTask(PHY_BVAL, SETB_Z4C);             // PhysicalBoundary
    }

    AddTask(ALG_CONSTR, PHY_BVAL);             // EnforceAlgConstr
    AddTask(Z4C_TO_ADM, ALG_CONSTR);           // Z4cToADM
    AddTask(ADM_CONSTR, Z4C_TO_ADM);           // ADM_Constraints
    AddTask(Z4C_WEYL, Z4C_TO_ADM);             // Calc Psi4
    AddTask(WAVE_EXTR, Z4C_WEYL);              // Project Psi4 multipoles
    AddTask(USERWORK, ADM_CONSTR);             // UserWork

    AddTask(NEW_DT, USERWORK);                 // NewBlockTimeStep
    if (pm->adaptive) {
      AddTask(FLAG_AMR, USERWORK);             // CheckRefinement
      AddTask(CLEAR_ALLBND, FLAG_AMR);         // ClearAllBoundary
    } else {
      AddTask(CLEAR_ALLBND, NEW_DT);           // ClearAllBoundary
    }

    AddTask(ASSERT_FIN, CLEAR_ALLBND);         // AssertFinite
  } // end of using namespace block
}

//---------------------------------------------------------------------------------------
//  Sets id and dependency for "ntask" member of task_list_ array, then iterates value of
//  ntask.

void Z4cIntegratorTaskList::AddTask(const TaskID& id, const TaskID& dep) {
    task_list_[ntasks].task_id = id;
    task_list_[ntasks].dependency = dep;

    using namespace Z4cIntegratorTaskNames; // NOLINT (build/namespace)

    if (id == CLEAR_ALLBND) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::ClearAllBoundary);
      task_list_[ntasks].lb_time = false;
    } else if (id == CALC_Z4CRHS) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::CalculateZ4cRHS);
      task_list_[ntasks].lb_time = true;
    } else if (id == INT_Z4C) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::IntegrateZ4c);
      task_list_[ntasks].lb_time = true;
    } else if (id == SEND_Z4C) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::SendZ4c);
      task_list_[ntasks].lb_time = true;
    } else if (id == RECV_Z4C) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::ReceiveZ4c);
      task_list_[ntasks].lb_time = false;
    } else if (id == SETB_Z4C) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::SetBoundariesZ4c);
      task_list_[ntasks].lb_time = true;
    } else if (id == PROLONG) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::Prolongation);
      task_list_[ntasks].lb_time = true;
    } else if (id == PHY_BVAL) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::PhysicalBoundary);
      task_list_[ntasks].lb_time = true;
    } else if (id == ALG_CONSTR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::EnforceAlgConstr);
      task_list_[ntasks].lb_time = true;
    } else if (id == Z4C_TO_ADM) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::Z4cToADM);
      task_list_[ntasks].lb_time = true;
    } else if (id == USERWORK) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::UserWork);
      task_list_[ntasks].lb_time = true;
    } else if (id == ADM_CONSTR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::ADM_Constraints);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == Z4C_WEYL) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::Z4c_Weyl);
      task_list_[ntasks].lb_time = true;
    } else if (id == WAVE_EXTR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::WaveExtract);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == NEW_DT) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::NewBlockTimeStep);
      task_list_[ntasks].lb_time = true;
    } else if (id == FLAG_AMR) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::CheckRefinement);
      task_list_[ntasks].lb_time = true;
    }
    else if (id == ASSERT_FIN) {
      task_list_[ntasks].TaskFunc=
        static_cast<TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&Z4cIntegratorTaskList::AssertFinite);
      task_list_[ntasks].lb_time = false;
    }
    else {
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTask" << std::endl
          << "Invalid Task is specified" << std::endl;
      ATHENA_ERROR(msg);
    }

    ntasks++;
    return;
}


void Z4cIntegratorTaskList::StartupTaskList(MeshBlock *pmb, int stage) {
  if (stage == 1) {
    // Auxiliar var u1 needs to be initialized to 0 at the beginning of each cycle
    // Change to emulate PassiveScalars logic
    pmb->pz4c->storage.u1.ZeroClear();
  }

  pmb->pbval->StartReceivingSubset(BoundaryCommSubset::all, pmb->pbval->bvars_main_int_vc);
  return;
}

//----------------------------------------------------------------------------------------
// Functions to end MPI communication

TaskStatus Z4cIntegratorTaskList::ClearAllBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundarySubset(BoundaryCommSubset::all, pmb->pbval->bvars_main_int_vc);
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to calculate the RHS

TaskStatus Z4cIntegratorTaskList::CalculateZ4cRHS(MeshBlock *pmb, int stage) {
  // PunctureTracker: interpolate beta at puncture position before evolution
  if (stage == 1) {
    for (auto ptracker : pmb->pmy_mesh->pz4c_tracker) {
      ptracker->InterpolateShift(pmb, pmb->pz4c->storage.u);
    }
  }

  if (stage <= nstages) {
    pmb->pz4c->Z4cRHS(pmb->pz4c->storage.u,
                      pmb->pz4c->storage.mat,
                      pmb->pz4c->storage.rhs);

    // application of Sommerfeld boundary conditions
    pmb->pz4c->Z4cBoundaryRHS(pmb->pz4c->storage.u,
                              pmb->pz4c->storage.mat,
                              pmb->pz4c->storage.rhs);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to integrate variables

TaskStatus Z4cIntegratorTaskList::IntegrateZ4c(MeshBlock *pmb, int stage) {
  Z4c *pz4c = pmb->pz4c;

  if (stage <= nstages) {
    // This time-integrator-specific averaging operation logic is identical
    // to IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pz4c->WeightedAve(pz4c->storage.u1, pz4c->storage.u,
                      pz4c->storage.u2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    pz4c->WeightedAve(pz4c->storage.u, pz4c->storage.u1,
                      pz4c->storage.u2, ave_wghts);
    pz4c->AddZ4cRHS(pz4c->storage.rhs, stage_wghts[stage-1].beta,
                    pz4c->storage.u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}
//----------------------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks

TaskStatus Z4cIntegratorTaskList::SendZ4c(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pz4c->ubvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus Z4cIntegratorTaskList::ReceiveZ4c(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pz4c->ubvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus Z4cIntegratorTaskList::SetBoundariesZ4c(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pz4c->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

//--------------------------------------------------------------------------------------
// Functions for everything else

TaskStatus Z4cIntegratorTaskList::Prolongation(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time
                       + stage_wghts[(stage-1)].ebeta*pmb->pmy_mesh->dt;
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    pbval->ProlongateBoundaries(t_end_stage, dt, pmb->pbval->bvars_main_int_vc);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time
                       + stage_wghts[(stage-1)].ebeta*pmb->pmy_mesh->dt;
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);

    pbval->ApplyPhysicalVertexCenteredBoundaries(t_end_stage, dt);

  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();
  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::EnforceAlgConstr(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pz4c->AlgConstr(pmb->pz4c->storage.u);
  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::Z4cToADM(MeshBlock *pmb, int stage) {
  // BD: this map is only required on the final stage
  if (stage != nstages) return TaskStatus::success;

  pmb->pz4c->Z4cToADM(pmb->pz4c->storage.u, pmb->pz4c->storage.adm);
  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::Z4c_Weyl(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;
  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction)) {
    pmb->pz4c->Z4cWeyl(pmb->pz4c->storage.adm, pmb->pz4c->storage.mat,
                       pmb->pz4c->storage.weyl);
  }

  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::WaveExtract(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;

  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction)) {
    AthenaArray<Real> u_R;
    AthenaArray<Real> u_I;
    u_R.InitWithShallowSlice(pmb->pz4c->storage.weyl, Z4c::I_WEY_rpsi4, 1);
    u_I.InitWithShallowSlice(pmb->pz4c->storage.weyl, Z4c::I_WEY_ipsi4, 1);
    for (auto pwextr : pmb->pwave_extr_loc) {
        pwextr->Decompose_multipole(u_R,u_I);
    }
  }

  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::ADM_Constraints(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;

  // check last stage is actually at a relevant time
  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con)) {
    pmb->pz4c->ADMConstraints(pmb->pz4c->storage.con, pmb->pz4c->storage.adm,
                              pmb->pz4c->storage.mat, pmb->pz4c->storage.u);
  }

  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::NewBlockTimeStep(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pz4c->NewBlockTimeStep();
  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}

TaskStatus Z4cIntegratorTaskList::AssertFinite(MeshBlock *pmb, int stage) {
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


//----------------------------------------------------------------------------------------
// \!fn bool Z4cIntegratorTaskList::CurrentTimeCalculationThreshold(
//   MeshBlock *pmb, aux_NextTimeStep *variable)
// \brief Given current time / ncycles, does a specified 'dt' mean we need
//        to calculate something?
//        Secondary effect is to mutate next_time
bool Z4cIntegratorTaskList::CurrentTimeCalculationThreshold(
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
// \!fn void Z4cIntegratorTaskList::UpdateTaskListTriggers()
// \brief Update 'next_time' outside task list to avoid race condition
void Z4cIntegratorTaskList::UpdateTaskListTriggers() {
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
}
