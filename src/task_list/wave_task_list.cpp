//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_task_list.cpp
//  \brief time integrator for the wave equation (based on time_integrator.cpp)

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
#include "../wave/wave.hpp"
#include "../wave/wave_extract.hpp"
#include "../z4c/z4c.hpp"
#include "../parameter_input.hpp"
#include "task_list.hpp"
#include "wave_task_list.hpp"

//----------------------------------------------------------------------------------------
//  WaveIntegratorTaskList constructor
WaveIntegratorTaskList::WaveIntegratorTaskList(ParameterInput *pin, Mesh *pm)
  : TaskList(pm)
{
  // First, define each time-integrator by setting weights for each step of the algorithm
  // and the CFL number stability limit when coupled to the single-stage spatial operator.
  // Currently, the time-integrators must be expressed as 2S-type algorithms as in
  // Ketchenson (2010) Algorithm 3, which incudes 2N (Williamson) and 2R (van der Houwen)
  // popular 2-register low-storage RK methods. The 2S-type integrators depend on a
  // bidiagonally sparse Shu-Osher representation; at each stage l:
  //
  //    U^{l} = a_{l,l-2}*U^{l-2} + a_{l-1}*U^{l-1}
  //          - b_{l,l-2}*dt*RHS(U_{l-2}) - b_{l,l-1}*dt*RHS(U_{l-1}),
  //
  // where U^{l-1} and U^{l-2} are previous stages and a_{l,l-2}, a_{l,l-1}=(1-a_{l,l-2}),
  // and b_{l,l-2}, b_{l,l-1} are weights that are different for each stage and integrator
  //
  // The 2x RHS evaluations of per stage is avoided by adding another weighted
  // average / caching of these terms each stage. The API and framework is
  // extensible to three register 3S* methods.

  // Notation: exclusively using "stage", equivalent in lit. to "substage" or "substep"
  // (sometimes "step"), to refer to intermediate values between "timesteps" = "cycles"
  // "Step" is often used for generic sequences in code, e.g. main.cpp: "Step 1: MPI"

  integrator = pin->GetOrAddString("time","integrator","vl2");
  int dim = 1;
  if (pm->mesh_size.nx2 > 1) dim = 2;
  if (pm->mesh_size.nx3 > 1) dim = 3;

  if (integrator == "vl2") {
    // VL: second-order van Leer integrator (Stone & Gardiner, NewA 14, 139 2009)
    // Simple predictor-corrector scheme similar to MUSCL-Hancock
    // Expressed in 2S or 3S* algorithm form
    nstages = 2;
    cfl_limit = 1.0;
    // Modify VL2 stability limit in 2D, 3D
    if (dim == 2) cfl_limit = 0.5;
    if (dim == 3) cfl_limit = ONE_3RD;

    stage_wghts[0].delta = 1.0; // required for consistency
    stage_wghts[0].gamma_1 = 0.0;
    stage_wghts[0].gamma_2 = 1.0;
    stage_wghts[0].gamma_3 = 0.0;
    stage_wghts[0].beta = 0.5;

    stage_wghts[1].delta = 0.0;
    stage_wghts[1].gamma_1 = 0.0;
    stage_wghts[1].gamma_2 = 1.0;
    stage_wghts[1].gamma_3 = 0.0;
    stage_wghts[1].beta = 1.0;
  } else if (integrator == "rk2") {
    // Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
    // Optimal (in error bounds) explicit two-stage, second-order SSPRK
    nstages = 2;
    cfl_limit = 1.0;
    stage_wghts[0].delta = 1.0;
    stage_wghts[0].gamma_1 = 0.0;
    stage_wghts[0].gamma_2 = 1.0;
    stage_wghts[0].gamma_3 = 0.0;
    stage_wghts[0].beta = 1.0;

    stage_wghts[1].delta = 0.0;
    stage_wghts[1].gamma_1 = 0.5;
    stage_wghts[1].gamma_2 = 0.5;
    stage_wghts[1].gamma_3 = 0.0;
    stage_wghts[1].beta = 0.5;
  } else if (integrator == "rk3") {
    // SSPRK (3,3): Gottlieb (2009) equation 3.2
    // Optimal (in error bounds) explicit three-stage, third-order SSPRK
    nstages = 3;
    cfl_limit = 1.0;
    stage_wghts[0].delta = 1.0;
    stage_wghts[0].gamma_1 = 0.0;
    stage_wghts[0].gamma_2 = 1.0;
    stage_wghts[0].gamma_3 = 0.0;
    stage_wghts[0].beta = 1.0;

    stage_wghts[1].delta = 0.0;
    stage_wghts[1].gamma_1 = 0.25;
    stage_wghts[1].gamma_2 = 0.75;
    stage_wghts[1].gamma_3 = 0.0;
    stage_wghts[1].beta = 0.25;

    stage_wghts[2].delta = 0.0;
    stage_wghts[2].gamma_1 = TWO_3RD;
    stage_wghts[2].gamma_2 = ONE_3RD;
    stage_wghts[2].gamma_3 = 0.0;
    stage_wghts[2].beta = TWO_3RD;
    //} else if (integrator == "ssprk5_3") {
    //} else if (integrator == "ssprk10_4") {
  } else if (integrator == "rk4") {
    // RK4()4[2S] from Table 2 of Ketchenson (2010)
    // Non-SSP, explicit four-stage, fourth-order RK
    nstages = 4;
    // Stability properties are similar to classical RK4
    // Refer to Colella (2011) for constant advection with 4th order fluxes
    // linear stability analysis
    cfl_limit = 1.3925;
    stage_wghts[0].delta = 1.0;
    stage_wghts[0].gamma_1 = 0.0;
    stage_wghts[0].gamma_2 = 1.0;
    stage_wghts[0].gamma_3 = 0.0;
    stage_wghts[0].beta = 1.193743905974738;

    stage_wghts[1].delta = 0.217683334308543;
    stage_wghts[1].gamma_1 = 0.121098479554482;
    stage_wghts[1].gamma_2 = 0.721781678111411;
    stage_wghts[1].gamma_3 = 0.0;
    stage_wghts[1].beta = 0.099279895495783;

    stage_wghts[2].delta = 1.065841341361089;
    stage_wghts[2].gamma_1 = -3.843833699660025;
    stage_wghts[2].gamma_2 = 2.121209265338722;
    stage_wghts[2].gamma_3 = 0.0;
    stage_wghts[2].beta = 1.131678018054042;

    stage_wghts[3].delta = 0.0;
    stage_wghts[3].gamma_1 = 0.546370891121863;
    stage_wghts[3].gamma_2 = 0.198653035682705;
    stage_wghts[3].gamma_3 = 0.0;
    stage_wghts[3].beta = 0.310665766509336;
  } else if (integrator == "ssprk5_4") {
    // SSPRK (5,4): Gottlieb (2009) section 3.1
    // Optimal (in error bounds) explicit five-stage, fourth-order SSPRK
    // 3N method, but there is no 3S* formulation due to irregular sparsity
    // of Shu-Osher form matrix, alpha
    nstages = 5;
    cfl_limit = 1.3925;
    stage_wghts[0].delta = 1.0;
    stage_wghts[0].gamma_1 = 0.0;
    stage_wghts[0].gamma_2 = 1.0;
    stage_wghts[0].gamma_3 = 0.0;
    stage_wghts[0].beta = 0.391752226571890;

    stage_wghts[1].delta = 0.0; // u1 = u^n
    stage_wghts[1].gamma_1 = 0.555629506348765;
    stage_wghts[1].gamma_2 = 0.444370493651235;
    stage_wghts[1].gamma_3 = 0.0;
    stage_wghts[1].beta = 0.368410593050371;

    stage_wghts[2].delta = 0.0;
    stage_wghts[2].gamma_1 = 0.379898148511597;
    stage_wghts[2].gamma_2 = 0.0;
    stage_wghts[2].gamma_3 = 0.620101851488403; // u2 = u^n
    stage_wghts[2].beta = 0.251891774271694;

    stage_wghts[3].delta = 0.0;
    stage_wghts[3].gamma_1 = TWO_3RD;
    stage_wghts[3].gamma_2 = ONE_3RD;
    stage_wghts[3].gamma_3 = 0.178079954393132; // u2 = u^n
    stage_wghts[3].beta = 0.544974750228521;

    stage_wghts[4].delta = 0.0;
    stage_wghts[4].gamma_1 = 0.386708617503268; // from Gottlieb (2009), u^(4) coeff.
    stage_wghts[4].gamma_2 = ONE_3RD;
    stage_wghts[4].gamma_3 = 0.0;
    stage_wghts[4].beta = 0.226007483236906; // from Gottlieb (2009), F(u^(4)) coeff.
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in CreateTimeIntegrator" << std::endl
        << "integrator=" << integrator << " not valid time integrator" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // Set cfl_number based on user input and time integrator CFL limit
  Real cfl_number = pin->GetReal("time","cfl_number");
  if (cfl_number > cfl_limit) {
    std::cout << "### Warning in CreateTimeIntegrator" << std::endl
        << "User CFL number " << cfl_number << " must be smaller than " << cfl_limit
        << " for integrator=" << integrator << " in "
        << dim << "D simulation" << std::endl << "Setting to limit" << std::endl;
    cfl_number = cfl_limit;
  }
  // Save to Mesh class
  pm->cfl_number = cfl_number;

  // Now assemble list of tasks for each stage of time integrator
  {using namespace WaveIntegratorTaskNames;
    AddWaveIntegratorTask(STARTUP_INT, NONE);
    AddWaveIntegratorTask(START_ALLRECV, STARTUP_INT);
    AddWaveIntegratorTask(CALC_WAVERHS, START_ALLRECV);
    AddWaveIntegratorTask(INT_WAVE, CALC_WAVERHS);
    AddWaveIntegratorTask(SEND_WAVE, INT_WAVE);
    AddWaveIntegratorTask(RECV_WAVE, START_ALLRECV);
    if (pm->multilevel) { // SMR or AMR
      AddWaveIntegratorTask(PROLONG, (SEND_WAVE|RECV_WAVE));
      AddWaveIntegratorTask(PHY_BVAL, PROLONG);
    }
    else {
      AddWaveIntegratorTask(PHY_BVAL, (SEND_WAVE|RECV_WAVE));
    }
    AddWaveIntegratorTask(WAVE_EXTR, PHY_BVAL);
    AddWaveIntegratorTask(USERWORK, PHY_BVAL);
    AddWaveIntegratorTask(NEW_DT, USERWORK);
    if (pm->adaptive==true) {
      AddWaveIntegratorTask(AMR_FLAG, USERWORK);
      AddWaveIntegratorTask(CLEAR_ALLBND, AMR_FLAG);
    } else {
      AddWaveIntegratorTask(CLEAR_ALLBND, NEW_DT);
    }
  } // using namespace WaveIntegratorTaskNames
}

void WaveIntegratorTaskList::AddWaveIntegratorTask(uint64_t id, uint64_t dep) {
  task_list_[ntasks].task_id=id;
  task_list_[ntasks].dependency=dep;

  using namespace WaveIntegratorTaskNames;
  switch((id)) {
    case (START_ALLRECV):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::StartAllReceive);
      break;
    case (CLEAR_ALLBND):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::ClearAllBoundary);
      break;
    case (CALC_WAVERHS):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::CalculateWaveRHS);
      break;
    case (INT_WAVE):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::WaveIntegrate);
      break;
    case (SEND_WAVE):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::WaveSend);
      break;
    case (RECV_WAVE):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::WaveReceive);
      break;
    case (PROLONG):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::Prolongation);
      break;
    case (PHY_BVAL):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::PhysicalBoundary);
      break;
    case (USERWORK):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::UserWork);
      break;
    case (NEW_DT):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::NewBlockTimeStep);
      break;
    case (AMR_FLAG):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::CheckRefinement);
      break;
    case (STARTUP_INT):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::StartupIntegrator);
      break;
    case (WAVE_EXTR):
      task_list_[ntasks].TaskFunc=
        static_cast<enum TaskStatus (TaskList::*)(MeshBlock*,int)>
        (&WaveIntegratorTaskList::WaveExtract);

    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in AddTimeIntegratorTask" << std::endl
          << "Invalid Task "<< id << " is specified" << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }
  ntasks++;
  return;
}

//----------------------------------------------------------------------------------------
// Functions to start/end MPI communication

enum TaskStatus WaveIntegratorTaskList::StartAllReceive(MeshBlock *pmb, int stage) {
  Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
  Real time = pmb->pmy_mesh->time+dt;
  pmb->pbval->StartReceivingAll(time);
  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::ClearAllBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundaryAll();
  return TASK_SUCCESS;
}

//----------------------------------------------------------------------------------------
// Functions to calculate the RHS

enum TaskStatus WaveIntegratorTaskList::CalculateWaveRHS(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pwave->WaveRHS(pmb->pwave->u);
    pmb->pwave->WaveBoundaryRHS(pmb->pwave->u);
    return TASK_NEXT;
  }
  return TASK_FAIL;
}

//----------------------------------------------------------------------------------------
// Functions to integrate conserved variables

enum TaskStatus WaveIntegratorTaskList::WaveIntegrate(MeshBlock *pmb, int stage)
{
    Wave *pwave = pmb->pwave;

    if (stage <= nstages) {
      // This time-integrator-specific averaging operation logic is identical to HydroInt

      Real ave_wghts[3];
      ave_wghts[0] = 1.0;
      ave_wghts[1] = stage_wghts[stage-1].delta;
      ave_wghts[2] = 0.0;
      pwave->WeightedAve(pwave->u1,pwave->u,pwave->u2,ave_wghts);

      ave_wghts[0] = stage_wghts[stage-1].gamma_1;
      ave_wghts[1] = stage_wghts[stage-1].gamma_2;
      ave_wghts[2] = stage_wghts[stage-1].gamma_3;
      pwave->WeightedAve(pwave->u,pwave->u1,pwave->u2,ave_wghts);
      pwave->AddWaveRHS(stage_wghts[stage-1].beta,pwave->u);

      return TASK_NEXT;
    }
    return TASK_FAIL;
}

//----------------------------------------------------------------------------------------
// Functions to communicate variables between MeshBlocks

enum TaskStatus WaveIntegratorTaskList::WaveSend(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
      pmb->pbval->SendCellCenteredBoundaryBuffers(pmb->pwave->u, WAVE_SOL);
  } else {
    return TASK_FAIL;
  }
  return TASK_SUCCESS;
}

//----------------------------------------------------------------------------------------
// Functions to receive variables between MeshBlocks

enum TaskStatus WaveIntegratorTaskList::WaveReceive(MeshBlock *pmb, int step) {
  bool ret;
  if(step <= nstages) {
      ret=pmb->pbval->ReceiveCellCenteredBoundaryBuffers(pmb->pwave->u, WAVE_SOL);
  } else {
    return TASK_FAIL;
  }

  if(ret == true) {
    return TASK_SUCCESS;
  } else {
    return TASK_FAIL;
  }
}

//--------------------------------------------------------------------------------------
// Functions for everything else

enum TaskStatus WaveIntegratorTaskList::Prolongation(MeshBlock *pmb, int stage) {
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;
  Wave *pwave=pmb->pwave;
  Z4c *pz4c=pmb->pz4c;
  BoundaryValues *pbval=pmb->pbval;

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    pbval->ProlongateBoundaries(phydro->w,  phydro->u, pwave->u, pz4c->storage.u,
                                pfield->b,  pfield->bcc, t_end_stage, dt);
  } else {
    return TASK_FAIL;
  }

  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::PhysicalBoundary(MeshBlock *pmb, int stage) {
  Hydro *phydro=pmb->phydro;
  Field *pfield=pmb->pfield;
  Wave *pwave=pmb->pwave;
  Z4c *pz4c=pmb->pz4c;
  BoundaryValues *pbval=pmb->pbval;

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    pbval->ApplyPhysicalBoundaries(phydro->w,  phydro->u, pwave->u, pz4c->storage.u,
                                   pfield->b,  pfield->bcc, t_end_stage, dt);
  } else {
    return TASK_FAIL;
  }

  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::WaveExtract(MeshBlock * pmb, int stage) {
  if (stage != nstages) return TASK_SUCCESS; // only do on last stage
  AthenaArray<Real> u;
  u.InitWithShallowSlice(pmb->pwave->u, 0, 1);
  pmb->pwave_extr_loc->Decompose(u);
}

enum TaskStatus WaveIntegratorTaskList::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TASK_SUCCESS; // only do on last stage
  pmb->UserWorkInLoop();
  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::NewBlockTimeStep(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TASK_SUCCESS; // only do on last stage
  pmb->pwave->NewBlockTimeStep();
  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TASK_SUCCESS; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TASK_SUCCESS;
}

enum TaskStatus WaveIntegratorTaskList::StartupIntegrator(MeshBlock *pmb, int stage) {
  // Initialize time-integrator only on first stage
  if (stage != 1) {
    return TASK_SUCCESS;
  } else {
    // For each Meshblock, initialize time abscissae of each memory register pair (u,b)
    // at stage=0 to correspond to the beginning of the interval [t^n, t^{n+1}]
    pmb->stage_abscissae[0][0] = 0.0;
    pmb->stage_abscissae[0][1] = 0.0; // u1 advances to u1 = 0*u1 + 1.0*u in stage=1
    pmb->stage_abscissae[0][2] = 0.0; // u2 = u cached for all stages in 3S* methods

    // Given overall timestep dt, precompute the time abscissae for all registers, stages
    for (int l=1; l<=nstages; l++) {
      // Update the dt abscissae of each memory register to values at end of this stage
      const IntegratorWeight w = stage_wghts[l-1];

      // u1 = u1 + delta*u
      pmb->stage_abscissae[l][1] = pmb->stage_abscissae[l-1][1]
          + w.delta*pmb->stage_abscissae[l-1][0];
      // u = gamma_1*u + gamma_2*u1 + gamma_3*u2 + beta*dt*F(u)
      pmb->stage_abscissae[l][0] = w.gamma_1*pmb->stage_abscissae[l-1][0]
          + w.gamma_2*pmb->stage_abscissae[l][1]
          + w.gamma_3*pmb->stage_abscissae[l-1][2]
          + w.beta*pmb->pmy_mesh->dt;
      // u2 = u^n
      pmb->stage_abscissae[l][2] = 0.0;
    }

    // Auxiliar var u1 needs to be initialized to 0 at the beginning of each cycle
    Wave *pwave = pmb->pwave;
    Real ave_wghts[3];
    ave_wghts[0] = 0.0;
    ave_wghts[1] = 0.0;
    ave_wghts[2] = 0.0;
    pwave->WeightedAve(pwave->u1,pwave->u,pwave->u,ave_wghts);

    return TASK_SUCCESS;
  }
}
