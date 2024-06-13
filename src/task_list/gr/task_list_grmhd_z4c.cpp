// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../bvals/bvals.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../gravity/gravity.hpp"
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


// ----------------------------------------------------------------------------

using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;
using namespace TaskNames::GeneralRelativity::GRMHD_Z4c;

GRMHD_Z4c::GRMHD_Z4c(ParameterInput *pin, Mesh *pm)
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

  {

    if (!STS_ENABLED)
    {
      Add(DIFFUSE_HYD, NONE, &GRMHD_Z4c::DiffuseHydro);

      if (MAGNETIC_FIELDS_ENABLED)
      {
        Add(DIFFUSE_FLD, NONE, &GRMHD_Z4c::DiffuseField);

        // compute hydro fluxes, integrate hydro variables
        Add(CALC_HYDFLX,
            (DIFFUSE_HYD | DIFFUSE_FLD),
            &GRMHD_Z4c::CalculateHydroFlux);
      } else { // Hydro
        Add(CALC_HYDFLX,
            DIFFUSE_HYD,
            &GRMHD_Z4c::CalculateHydroFlux);
      }

      if (NSCALARS > 0)
      {
        Add(DIFFUSE_SCLR, NONE, &GRMHD_Z4c::DiffuseScalars);
        Add(CALC_SCLRFLX, (CALC_HYDFLX | DIFFUSE_SCLR), &GRMHD_Z4c::CalculateScalarFlux);
      }

    }
    else
    {
      // STS enabled:
      Add(CALC_HYDFLX, NONE, &GRMHD_Z4c::CalculateHydroFlux);
      if (NSCALARS > 0)
        Add(CALC_SCLRFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateScalarFlux);
    }
    if (pm->multilevel)
    { // SMR or AMR
      Add(SEND_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::SendHydroFlux);
      Add(RECV_HYDFLX, CALC_HYDFLX, &GRMHD_Z4c::ReceiveAndCorrectHydroFlux);
      Add(INT_HYD, RECV_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }
    else
    {
      Add(INT_HYD, CALC_HYDFLX, &GRMHD_Z4c::IntegrateHydro);
    }

    Add(SRCTERM_HYD, INT_HYD, &GRMHD_Z4c::AddSourceTermsHydro);
    Add(SEND_HYD, SRCTERM_HYD, &GRMHD_Z4c::SendHydro);
    Add(RECV_HYD, INT_HYD, &GRMHD_Z4c::ReceiveHydro);
    Add(SETB_HYD, (RECV_HYD | SRCTERM_HYD), &GRMHD_Z4c::SetBoundariesHydro);

    if (SHEARING_BOX)
    { // Shearingbox BC for Hydro
      Add(SEND_HYDSH, SETB_HYD, &GRMHD_Z4c::SendHydroShear);
      Add(RECV_HYDSH, SETB_HYD, &GRMHD_Z4c::ReceiveHydroShear);
    }

    if (NSCALARS > 0)
    {
      if (pm->multilevel)
      {
        Add(SEND_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::SendScalarFlux);
        Add(RECV_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c::ReceiveScalarFlux);
        Add(INT_SCLR, RECV_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }
      else
      {
        Add(INT_SCLR, CALC_SCLRFLX, &GRMHD_Z4c::IntegrateScalars);
      }
      // there is no SRCTERM_SCLR task
      Add(SEND_SCLR, INT_SCLR, &GRMHD_Z4c::SendScalars);
      Add(RECV_SCLR, NONE, &GRMHD_Z4c::ReceiveScalars);
      Add(SETB_SCLR, (RECV_SCLR | INT_SCLR), &GRMHD_Z4c::SetBoundariesScalars);
      // if (SHEARING_BOX) {
      //   Add(SEND_SCLRSH,SETB_SCLR);
      //   Add(RECV_SCLRSH,SETB_SCLR);
      // }
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      // compute MHD fluxes, integrate field
      Add(CALC_FLDFLX, CALC_HYDFLX, &GRMHD_Z4c::CalculateEMF);
      Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c::SendEMF);
      Add(RECV_FLDFLX, SEND_FLDFLX, &GRMHD_Z4c::ReceiveAndCorrectEMF);
      if (SHEARING_BOX)
      {// Shearingbox BC for EMF
        Add(SEND_EMFSH, RECV_FLDFLX, &GRMHD_Z4c::SendEMFShear);
        Add(RECV_EMFSH, RECV_FLDFLX, &GRMHD_Z4c::ReceiveEMFShear);
        Add(RMAP_EMFSH, RECV_EMFSH, &GRMHD_Z4c::RemapEMFShear);
        Add(INT_FLD, RMAP_EMFSH, &GRMHD_Z4c::IntegrateField);
      }
      else
      {
        Add(INT_FLD, RECV_FLDFLX, &GRMHD_Z4c::IntegrateField);
      }

      Add(SEND_FLD, INT_FLD, &GRMHD_Z4c::SendField);
      Add(RECV_FLD, NONE, &GRMHD_Z4c::ReceiveField);
      Add(SETB_FLD, (RECV_FLD | INT_FLD), &GRMHD_Z4c::SetBoundariesField);
      if (SHEARING_BOX)
      { // Shearingbox BC for Bfield
        Add(SEND_FLDSH, SETB_FLD, &GRMHD_Z4c::SendFieldShear);
        Add(RECV_FLDSH, SETB_FLD, &GRMHD_Z4c::ReceiveFieldShear);
      }


      // prolongate, compute new primitives
      if (pm->multilevel)
      { // SMR or AMR
        if (NSCALARS > 0) {
          Add(PROLONG_HYD, (SEND_HYD | SETB_HYD | SEND_FLD | SETB_FLD | SEND_SCLR | SETB_SCLR | Z4C_TO_ADM), &GRMHD_Z4c::Prolongation_Hyd);
        } else {
          Add(PROLONG_HYD, (SEND_HYD | SETB_HYD | SEND_FLD | SETB_FLD | Z4C_TO_ADM), &GRMHD_Z4c::Prolongation_Hyd);
        }
//        Add(CONS2PRIM,(PROLONG_HYD|UPDATE_SRC|Z4C_TO_ADM));
        Add(CONS2PRIM, (PROLONG_HYD | Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
      }
      else
      {
        if (SHEARING_BOX) {
          if (NSCALARS > 0) {
            Add(CONS2PRIM,
                    (SETB_HYD|SETB_FLD|SETB_SCLR|RECV_HYDSH|RECV_FLDSH|RMAP_EMFSH), &GRMHD_Z4c::Primitives);
          } else {
            Add(CONS2PRIM,(SETB_HYD|SETB_FLD|RECV_HYDSH|RECV_FLDSH|RMAP_EMFSH), &GRMHD_Z4c::Primitives);
          }
        } else {
          if (NSCALARS > 0) {
            Add(CONS2PRIM,(SETB_HYD|SETB_FLD|SETB_SCLR), &GRMHD_Z4c::Primitives);
          } else {
            Add(CONS2PRIM,(SETB_HYD|SETB_FLD|Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
//            Add(CONS2PRIM,(SETB_HYD|SETB_FLD|UPDATE_SRC|Z4C_TO_ADM));
          }
        }
      }
    } else {  // HYDRO
      // prolongate, compute new primitives
      if (pm->multilevel) { // SMR or AMR
        if (NSCALARS > 0) {
          Add(PROLONG_HYD,(SEND_HYD|SETB_HYD|SETB_SCLR|SEND_SCLR|Z4C_TO_ADM), &GRMHD_Z4c::Prolongation_Hyd);
        } else {
          Add(PROLONG_HYD,(SEND_HYD|SETB_HYD|Z4C_TO_ADM), &GRMHD_Z4c::Prolongation_Hyd);
        }
        Add(CONS2PRIM,(PROLONG_HYD|Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
//        Add(CONS2PRIM,(PROLONG_HYD|UPDATE_SRC|Z4C_TO_ADM));
      } else {
        if (SHEARING_BOX) {
          if (NSCALARS > 0) {
            Add(CONS2PRIM,(SETB_HYD|RECV_HYDSH|SETB_SCLR), &GRMHD_Z4c::Primitives);  // RECV_SCLRSH
          } else {
            Add(CONS2PRIM,(SETB_HYD|RECV_HYDSH), &GRMHD_Z4c::Primitives);
          }
        } else {
          if (NSCALARS > 0) {
            Add(CONS2PRIM,(SETB_HYD|SETB_SCLR), &GRMHD_Z4c::Primitives);
          } else {
            Add(CONS2PRIM,(SETB_HYD|Z4C_TO_ADM), &GRMHD_Z4c::Primitives);
//            Add(CONS2PRIM,(SETB_HYD|UPDATE_SRC|Z4C_TO_ADM));
          }
        }
      }
    }
    Add(UPDATE_SRC,(CONS2PRIM|CALC_Z4CRHS), &GRMHD_Z4c::UpdateSource);
    Add(PHY_BVAL_HYD,CONS2PRIM, &GRMHD_Z4c::PhysicalBoundary_Hyd);

    // Z4c tasks
    Add(CALC_Z4CRHS, NONE, &GRMHD_Z4c::CalculateZ4cRHS);
    Add(INT_Z4C, CALC_Z4CRHS, &GRMHD_Z4c::IntegrateZ4c);

    Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c::SendZ4c);
    if(MAGNETIC_FIELDS_ENABLED)
    {
      Add(RECV_Z4C, (INT_Z4C | RECV_HYD | RECV_FLD | RECV_FLDFLX), &GRMHD_Z4c::ReceiveZ4c);
    }
    else
    {
      Add(RECV_Z4C, (INT_Z4C | RECV_HYD), &GRMHD_Z4c::ReceiveZ4c);
    }

    Add(SETB_Z4C, (RECV_Z4C|INT_Z4C), &GRMHD_Z4c::SetBoundariesZ4c);

    if (pm->multilevel)
    { // SMR or AMR
      Add(PROLONG_Z4C, (SEND_Z4C|SETB_Z4C), &GRMHD_Z4c::Prolongation_Z4c);
      Add(PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }
    else
    {
      Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c::PhysicalBoundary_Z4c);
    }

    Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c::EnforceAlgConstr);

    if(MAGNETIC_FIELDS_ENABLED)
    {
      Add(Z4C_TO_ADM, (ALG_CONSTR|INT_HYD|INT_FLD), &GRMHD_Z4c::Z4cToADM);
    }
    else
    {
      Add(Z4C_TO_ADM, (ALG_CONSTR|INT_HYD), &GRMHD_Z4c::Z4cToADM);
    }

    Add(ADM_CONSTR, (Z4C_TO_ADM | UPDATE_SRC), &GRMHD_Z4c::ADM_Constraints);

    Add(Z4C_WEYL, Z4C_TO_ADM, &GRMHD_Z4c::Z4c_Weyl);
    Add(WAVE_EXTR, Z4C_WEYL, &GRMHD_Z4c::WaveExtract);

    Add(USERWORK, (ADM_CONSTR | PHY_BVAL_HYD), &GRMHD_Z4c::UserWork);
    Add(NEW_DT, USERWORK, &GRMHD_Z4c::NewBlockTimeStep);

    if (pm->adaptive)
    {
      Add(FLAG_AMR, USERWORK, &GRMHD_Z4c::CheckRefinement);
      Add(CLEAR_ALLBND, FLAG_AMR, &GRMHD_Z4c::ClearAllBoundary);
    }
    else
    {
      Add(CLEAR_ALLBND, NEW_DT, &GRMHD_Z4c::ClearAllBoundary);
    }


  } // namespace
}

// ----------------------------------------------------------------------------
void GRMHD_Z4c::StartupTaskList(MeshBlock *pmb, int stage)
{
  // application of Sommerfeld boundary conditions
  pmb->pz4c->Z4cBoundaryRHS(pmb->pz4c->storage.u,
                            pmb->pz4c->storage.mat,
                            pmb->pz4c->storage.rhs);

  BoundaryValues *pbval = pmb->pbval;

  Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
  // Scaled coefficient for RHS time-advance within stage
  Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);

  FCN_CC_CX_VC(
    pbval->ApplyPhysicalBoundaries,
    pbval->ApplyPhysicalCellCenteredXBoundaries,
    pbval->ApplyPhysicalVertexCenteredBoundaries
  )(t_end_stage, dt);

  if (stage == 1)
  {
    // For each Meshblock, initialize time abscissae of each memory register pair (u,b)
    // at stage=0 to correspond to the beginning of the interval [t^n, t^{n+1}]
    pmb->stage_abscissae[0][0] = 0.0;
    pmb->stage_abscissae[0][1] = 0.0; // u1 advances to u1 = 0*u1 + 1.0*u in stage=1
    pmb->stage_abscissae[0][2] = 0.0; // u2 = u cached for all stages in 3S* methods

    // Given overall timestep dt, compute the time abscissae for all registers, stages
    for (int l=1; l<=nstages; l++) {
      // Update the dt abscissae of each memory register to values at end of this stage
      const IntegratorWeights w = stage_wghts[l-1];

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

    // Initialize storage registers
    Hydro *ph = pmb->phydro;
    ph->u1.ZeroClear();
    if (integrator == "ssprk5_4")
      ph->u2 = ph->u;

    if (MAGNETIC_FIELDS_ENABLED)
    {
      Field *pf = pmb->pfield;
      pf->b1.x1f.ZeroClear();
      pf->b1.x2f.ZeroClear();
      pf->b1.x3f.ZeroClear();
      if (integrator == "ssprk5_4") {
        std::stringstream msg;
        msg << "### FATAL ERROR in GRMHD_Z4c::StartupTaskList\n"
            << "integrator=" << integrator << " is currently incompatible with MHD"
            << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    if (NSCALARS > 0)
    {
      PassiveScalars *ps = pmb->pscalars;
      ps->s1.ZeroClear();
      if (integrator == "ssprk5_4")
        ps->s2 = ps->s;
    }

    // Auxiliar var u1 needs to be initialized to 0 at the beginning of each cycle
    // Change to emulate PassiveScalars logic
    pmb->pz4c->storage.u1.ZeroClear();
    if (integrator == "ssprk5_4")
      pmb->pz4c->storage.u2 = pmb->pz4c->storage.u;
  }

  pmb->pbval->StartReceiving(BoundaryCommSubset::all);
  return;
}

//----------------------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c::ClearAllBoundary(MeshBlock *pmb, int stage) {
  pmb->pbval->ClearBoundary(BoundaryCommSubset::all);
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to calculates fluxes

TaskStatus GRMHD_Z4c::CalculateHydroFlux(MeshBlock *pmb, int stage) {
  Hydro *phydro = pmb->phydro;
  Field *pfield = pmb->pfield;

//printf("chydfl\n");
  if (stage <= nstages)
  {
    if ((stage == 1) && (integrator == "vl2")) {
      phydro->CalculateFluxes(phydro->w,  pfield->b,  pfield->bcc, 1);
      return TaskStatus::next;
    } else {

      phydro->CalculateFluxes(phydro->w,  pfield->b,  pfield->bcc, pmb->precon->xorder);

      return TaskStatus::next;
    }
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::CalculateEMF(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pfield->ComputeCornerE(pmb->phydro->w,  pmb->pfield->bcc);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to communicate fluxes between MeshBlocks for flux correction with AMR

TaskStatus GRMHD_Z4c::SendHydroFlux(MeshBlock *pmb, int stage) {
  pmb->phydro->hbvar.SendFluxCorrection();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::SendEMF(MeshBlock *pmb, int stage) {
  pmb->pfield->fbvar.SendFluxCorrection();
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive fluxes between MeshBlocks

TaskStatus GRMHD_Z4c::ReceiveAndCorrectHydroFlux(MeshBlock *pmb, int stage) {
  if (pmb->phydro->hbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c::ReceiveAndCorrectEMF(MeshBlock *pmb, int stage) {
  if (pmb->pfield->fbvar.ReceiveFluxCorrection()) {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
}

//----------------------------------------------------------------------------------------
// Functions to integrate conserved variables

TaskStatus GRMHD_Z4c::IntegrateHydro(MeshBlock *pmb, int stage) {
//printf("inthydro\n");
  Hydro *ph = pmb->phydro;
  PassiveScalars *ps = pmb->pscalars;
  Field *pf = pmb->pfield;

  if (pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  if (stage <= nstages)
  {
    // This time-integrator-specific averaging operation logic is identical to FieldInt
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ph->u1, ph->u, ph->u2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
      ph->u.SwapAthenaArray(ph->u1);
    else
      pmb->WeightedAveCC(ph->u, ph->u1, ph->u2, ave_wghts);

    const Real wght = stage_wghts[stage-1].beta*pmb->pmy_mesh->dt;

    ph->AddFluxDivergence(wght, ph->u);

    // add coordinate (geometric) source terms
#if USETM
    pmb->pcoord->AddCoordTermsDivergence(wght, ph->flux, ph->w, ps->r, pf->bcc, ph->u);
#else
    pmb->pcoord->AddCoordTermsDivergence(wght, ph->flux, ph->w, pf->bcc, ph->u);
#endif

    // Hardcode an additional flux divergence weighted average for the penultimate
    // stage of SSPRK(5,4) since it cannot be expressed in a 3S* framework
    if (stage == 4 && integrator == "ssprk5_4")
    {
      // From Gottlieb (2009), u^(n+1) partial calculation
      ave_wghts[0] = -1.0; // -u^(n) coeff.
      ave_wghts[1] = 0.0;
      ave_wghts[2] = 0.0;
      const Real beta = 0.063692468666290; // F(u^(3)) coeff.
      const Real wght_ssp = beta*pmb->pmy_mesh->dt;
      // writing out to u2 register
      pmb->WeightedAveCC(ph->u2, ph->u1, ph->u2, ave_wghts);
      ph->AddFluxDivergence(wght_ssp, ph->u2);
      // add coordinate (geometric) source terms
#if USETM
      pmb->pcoord->AddCoordTermsDivergence(wght_ssp, ph->flux, ph->w, ps->r, pf->bcc, ph->u2);
#else
      pmb->pcoord->AddCoordTermsDivergence(wght_ssp, ph->flux, ph->w, pf->bcc, ph->u2);
#endif
    }
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::IntegrateField(MeshBlock *pmb, int stage) {
  Field *pf = pmb->pfield;

  if (pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  if (stage <= nstages) {
    // This time-integrator-specific averaging operation logic is identical to HydroInt
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveFC(pf->b1, pf->b, pf->b2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;
    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0) {
      pf->b.x1f.SwapAthenaArray(pf->b1.x1f);
      pf->b.x2f.SwapAthenaArray(pf->b1.x2f);
      pf->b.x3f.SwapAthenaArray(pf->b1.x3f);
    } else {
      pmb->WeightedAveFC(pf->b, pf->b1, pf->b2, ave_wghts);
    }

    pf->CT(stage_wghts[stage-1].beta*pmb->pmy_mesh->dt, pf->b);

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//----------------------------------------------------------------------------------------
// Functions to add source terms

TaskStatus GRMHD_Z4c::AddSourceTermsHydro(MeshBlock *pmb, int stage) {
  return TaskStatus::next;
  Hydro *ph = pmb->phydro;
  Field *pf = pmb->pfield;

  // return if there are no source terms to be added
  if (!(ph->hsrc.hydro_sourceterms_defined)
      || pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  if (stage <= nstages) {
    // Time at beginning of stage for u()
    Real t_start_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage-1][0];
    // Scaled coefficient for RHS update
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    // Evaluate the time-dependent source terms at the time at the beginning of the stage
    ph->hsrc.AddHydroSourceTerms(t_start_stage, dt, ph->flux, ph->w, pf->bcc, ph->u);
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

//----------------------------------------------------------------------------------------
// Functions to calculate hydro diffusion fluxes (stored in HydroDiffusion::visflx[],
// cndflx[], added at the end of Hydro::CalculateFluxes()

TaskStatus GRMHD_Z4c::DiffuseHydro(MeshBlock *pmb, int stage) {
// printf("diffhydro\n");
  Hydro *ph = pmb->phydro;

  // return if there are no diffusion to be added
  if (!(ph->hdif.hydro_diffusion_defined)
      || pmb->pmy_mesh->fluid_setup != FluidFormulation::evolve) return TaskStatus::next;

  if (stage <= nstages) {
    ph->hdif.CalcDiffusionFlux(ph->w, ph->u, ph->flux);
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

//----------------------------------------------------------------------------------------
// Functions to calculate diffusion EMF

TaskStatus GRMHD_Z4c::DiffuseField(MeshBlock *pmb, int stage) {
  Field *pf = pmb->pfield;

  // return if there are no diffusion to be added
  if (!(pf->fdif.field_diffusion_defined)) return TaskStatus::next;

  if (stage <= nstages) {
    // TODO(pdmullen): DiffuseField is also called in SuperTimeStepTaskLsit. It must skip
    // Hall effect (once implemented) diffusion process in STS and always calculate those
    // terms in the main integrator.
    pf->fdif.CalcDiffusionEMF(pf->b, pf->bcc, pf->e);
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

//----------------------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks

TaskStatus GRMHD_Z4c::SendHydro(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    // Swap Hydro quantity in BoundaryVariable interface back to conserved var formulation
    // (also needed in SetBoundariesHydro(), since the tasks are independent)
    pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->u, HydroBoundaryQuantity::cons);
    pmb->phydro->hbvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::SendField(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pfield->fbvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus GRMHD_Z4c::ReceiveHydro(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->phydro->hbvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::ReceiveField(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pfield->fbvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::SetBoundariesHydro(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->phydro->hbvar.SwapHydroQuantity(pmb->phydro->u, HydroBoundaryQuantity::cons);
    pmb->phydro->hbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SetBoundariesField(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pfield->fbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SendHydroShear(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->phydro->hbvar.SendShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveHydroShear(MeshBlock *pmb, int stage) {
  bool ret;
  ret = false;
  if (stage <= nstages) {
    ret = pmb->phydro->hbvar.ReceiveShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::SendFieldShear(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pfield->fbvar.SendShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveFieldShear(MeshBlock *pmb, int stage) {
  bool ret;
  ret = false;
  if (stage <= nstages) {
    ret = pmb->pfield->fbvar.ReceiveShearingBoxBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::SendEMFShear(MeshBlock *pmb, int stage) {
  pmb->pfield->fbvar.SendEMFShearingBoxBoundaryCorrection();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveEMFShear(MeshBlock *pmb, int stage) {
  if (pmb->pfield->fbvar.ReceiveEMFShearingBoxBoundaryCorrection()) {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::RemapEMFShear(MeshBlock *pmb, int stage) {
  pmb->pfield->fbvar.RemapEMFShearingBoxBoundary();
  return TaskStatus::success;
}

//--------------------------------------------------------------------------------------
// Functions for everything else

TaskStatus GRMHD_Z4c::Prolongation_Hyd(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;
//printf("prolhydro\n");

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
// this only prolongates hydro vars


// TODO : VC/CC issue, use split Prolongate Boundary functions
    pbval->ProlongateHydroBoundaries(t_end_stage, dt);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::Primitives(MeshBlock *pmb, int stage) {
  Hydro *ph = pmb->phydro;
  Field *pf = pmb->pfield;
  PassiveScalars *ps = pmb->pscalars;
  BoundaryValues *pbval = pmb->pbval;

  int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;
  if (pbval->nblevel[1][1][0] != -1) il -= NGHOST;
  if (pbval->nblevel[1][1][2] != -1) iu += NGHOST;
  if (pbval->nblevel[1][0][1] != -1) jl -= NGHOST;
  if (pbval->nblevel[1][2][1] != -1) ju += NGHOST;
  if (pbval->nblevel[0][1][1] != -1) kl -= NGHOST;
  if (pbval->nblevel[2][1][1] != -1) ku += NGHOST;

  if (stage <= nstages) {
    // At beginning of this task, ph->w contains previous stage's W(U) output
    // and ph->w1 is used as a register to store the current stage's output.
    // For the second order integrators VL2 and RK2, the prim_old initial guess for the
    // Newton-Raphson solver in GR EOS uses the following abscissae:
    // stage=1: W at t^n and
    // stage=2: W at t^{n+1/2} (VL2) or t^{n+1} (RK2)
    pmb->peos->ConservedToPrimitive(ph->u, ph->w, pf->b, ph->w1, 
#if USETM
                                    ps->s, ps->r,
#endif
                                    pf->bcc, pmb->pcoord,
                                    il, iu, jl, ju, kl, ku,0);
#if !USETM
    if (NSCALARS > 0) {
      // r1/r_old for GR is currently unused:
      pmb->peos->PassiveScalarConservedToPrimitive(ps->s, ph->w1, // ph->u, (updated rho)
                                                   ps->r, ps->r,
                                                   pmb->pcoord, il, iu, jl, ju, kl, ku);
    }
#endif
    // this never tested - potential issue for WENO routines?
    // fourth-order EOS:
    if (pmb->precon->xorder == 4) {
      // for hydro, shrink buffer by 1 on all sides
      if (pbval->nblevel[1][1][0] != -1) il += 1;
      if (pbval->nblevel[1][1][2] != -1) iu -= 1;
      if (pbval->nblevel[1][0][1] != -1) jl += 1;
      if (pbval->nblevel[1][2][1] != -1) ju -= 1;
      if (pbval->nblevel[0][1][1] != -1) kl += 1;
      if (pbval->nblevel[2][1][1] != -1) ku -= 1;
      // for MHD, shrink buffer by 3
      // TODO(felker): add MHD loop limit calculation for 4th order W(U)
      pmb->peos->ConservedToPrimitiveCellAverage(ph->u, ph->w, pf->b, ph->w1, 
#if USETM
                                                 ps->s, ps->r,
#endif
                                                 pf->bcc, pmb->pcoord,
                                                 il, iu, jl, ju, kl, ku);
#if !USETM
      if (NSCALARS > 0) {
        pmb->peos->PassiveScalarConservedToPrimitiveCellAverage(
            ps->s, ps->r, ps->r, pmb->pcoord, il, iu, jl, ju, kl, ku);
      }
#endif
    }
    // swap AthenaArray data pointers so that w now contains the updated w_out
    ph->w.SwapAthenaArray(ph->w1);
    // r1/r_old for GR is currently unused:
    // ps->r.SwapAthenaArray(ps->r1);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::PhysicalBoundary_Hyd(MeshBlock *pmb, int stage) {
  Hydro *ph = pmb->phydro;
  PassiveScalars *ps = pmb->pscalars;
  BoundaryValues *pbval = pmb->pbval;
//physical boundaries only for hydro vars

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
    // Swap Hydro and (possibly) passive scalar quantities in BoundaryVariable interface
    // from conserved to primitive formulations:
    ph->hbvar.SwapHydroQuantity(ph->w, HydroBoundaryQuantity::prim);
    if (NSCALARS > 0)
      ps->sbvar.var_cc = &(ps->r);
    //TODO VC/CC issue use correct Physical Boundary function

    pbval->ApplyPhysicalBoundaries(t_end_stage, dt);

  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::UserWork(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->UserWorkInLoop();

  // TODO: BD- this should be shifted to its own task
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::success;
}




TaskStatus GRMHD_Z4c::CheckRefinement(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::CalculateScalarFlux(MeshBlock *pmb, int stage) {
  PassiveScalars *ps = pmb->pscalars;
  if (stage <= nstages) {
    if ((stage == 1) && (integrator == "vl2")) {
      ps->CalculateFluxes(ps->r, 1);
      return TaskStatus::next;
    } else {
      ps->CalculateFluxes(ps->r, pmb->precon->xorder);
      return TaskStatus::next;
    }
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SendScalarFlux(MeshBlock *pmb, int stage) {
  pmb->pscalars->sbvar.SendFluxCorrection();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveScalarFlux(MeshBlock *pmb, int stage) {
  if (pmb->pscalars->sbvar.ReceiveFluxCorrection()) {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c::IntegrateScalars(MeshBlock *pmb, int stage) {
  PassiveScalars *ps = pmb->pscalars;
  if (stage <= nstages) {
    // This time-integrator-specific averaging operation logic is identical to
    // IntegrateHydro, IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ps->s1, ps->s, ps->s2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;
    if (ave_wghts[0] == 0.0 && ave_wghts[1] == 1.0 && ave_wghts[2] == 0.0)
      ps->s.SwapAthenaArray(ps->s1);
    else
      pmb->WeightedAveCC(ps->s, ps->s1, ps->s2, ave_wghts);

    const Real wght = stage_wghts[stage-1].beta*pmb->pmy_mesh->dt;
    ps->AddFluxDivergence(wght, ps->s);

    // Hardcode an additional flux divergence weighted average for the penultimate
    // stage of SSPRK(5,4) since it cannot be expressed in a 3S* framework
    if (stage == 4 && integrator == "ssprk5_4") {
      // From Gottlieb (2009), u^(n+1) partial calculation
      ave_wghts[0] = -1.0; // -u^(n) coeff.
      ave_wghts[1] = 0.0;
      ave_wghts[2] = 0.0;
      const Real beta = 0.063692468666290; // F(u^(3)) coeff.
      const Real wght_ssp = beta*pmb->pmy_mesh->dt;
      // writing out to s2 register
      pmb->WeightedAveCC(ps->s2, ps->s1, ps->s2, ave_wghts);
      ps->AddFluxDivergence(wght_ssp, ps->s2);
    }
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::SendScalars(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    // Swap PassiveScalars quantity in BoundaryVariable interface back to conserved var
    // formulation (also needed in SetBoundariesScalars() since the tasks are independent)
    pmb->pscalars->sbvar.var_cc = &(pmb->pscalars->s);
    pmb->pscalars->sbvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::ReceiveScalars(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pscalars->sbvar.ReceiveBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c::SetBoundariesScalars(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    // Set PassiveScalars quantity in BoundaryVariable interface to cons var formulation
    pmb->pscalars->sbvar.var_cc = &(pmb->pscalars->s);
    pmb->pscalars->sbvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c::DiffuseScalars(MeshBlock *pmb, int stage) {
  PassiveScalars *ps = pmb->pscalars;
  Hydro *ph = pmb->phydro;
  // return if there are no diffusion to be added
  if (!(ps->scalar_diffusion_defined))
    return TaskStatus::next;

  if (stage <= nstages) {
    // TODO(felker): adapted directly from HydroDiffusion::ClearFlux. Deduplicate
    ps->diffusion_flx[X1DIR].ZeroClear();
    ps->diffusion_flx[X2DIR].ZeroClear();
    ps->diffusion_flx[X3DIR].ZeroClear();

    // unlike HydroDiffusion, only 1x passive scalar diffusive process is allowed, so
    // there is no need for counterpart to wrapper fn HydroDiffusion::CalcDiffusionFlux
    ps->DiffusiveFluxIso(ps->r, ph->w, ps->diffusion_flx);
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

// Z4C tasks begin here

TaskStatus GRMHD_Z4c::CalculateZ4cRHS(MeshBlock *pmb, int stage)
{
 // PunctureTracker: interpolate beta at puncture position before evolution
  if (stage == 1) {
    for (auto ptracker : pmb->pmy_mesh->pz4c_tracker) {
      ptracker->InterpolateShift(pmb, pmb->pz4c->storage.u);
    }
  }

  if (stage <= nstages)
  {
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
TaskStatus GRMHD_Z4c::IntegrateZ4c(MeshBlock *pmb, int stage) {
  Z4c *pz4c = pmb->pz4c;
  Hydro *ph = pmb->phydro;
  Field *pf = pmb->pfield;

//printf("intz4c\n");
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

    if (stage == 4 && integrator == "ssprk5_4")
    {
      // From Gottlieb (2009), u^(n+1) partial calculation
      ave_wghts[0] = -1.0; // -u^(n) coeff.
      ave_wghts[1] = 0.0;
      ave_wghts[2] = 0.0;
      const Real beta = 0.063692468666290; // F(u^(3)) coeff.
      const Real wght_ssp = beta*pmb->pmy_mesh->dt;
      // writing out to u2 register
      pz4c->WeightedAve(pz4c->storage.u2,
                        pz4c->storage.u1,
                        pz4c->storage.u2,
                        ave_wghts);

      pz4c->AddZ4cRHS(pz4c->storage.rhs,
                      wght_ssp,
                      pz4c->storage.u2);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}
//----------------------------------------------------------------------------------------
// Functions to communicate conserved variables between MeshBlocks

TaskStatus GRMHD_Z4c::SendZ4c(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
    pmb->pz4c->ubvar.SendBoundaryBuffers();
  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

//----------------------------------------------------------------------------------------
// Functions to receive conserved variables between MeshBlocks

TaskStatus GRMHD_Z4c::ReceiveZ4c(MeshBlock *pmb, int stage) {
  bool ret;
  if (stage <= nstages) {
    ret = pmb->pz4c->ubvar.ReceiveBoundaryBuffers();
//      ret=1;
  } else {
    return TaskStatus::fail;
  }
  if (ret) {
    return TaskStatus::success;
  } else {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c::SetBoundariesZ4c(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
     	  pmb->pz4c->ubvar.SetBoundaries();
    return TaskStatus::success;
  }
  return TaskStatus::fail;
}
//--------------------------------------------------------------------------------------
// Functions for everything else
TaskStatus GRMHD_Z4c::Prolongation_Z4c(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;
//prolongates only z4c vars
  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);
// TODO VC/CC issue: choose appropriate prolongation fn here
    pbval->ProlongateBoundaries(t_end_stage, dt);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}
TaskStatus GRMHD_Z4c::PhysicalBoundary_Z4c(MeshBlock *pmb, int stage) {
  BoundaryValues *pbval = pmb->pbval;

  if (stage <= nstages) {
    // Time at the end of stage for (u, b) register pair
    Real t_end_stage = pmb->pmy_mesh->time + pmb->stage_abscissae[stage][0];
    // Scaled coefficient for RHS time-advance within stage
    Real dt = (stage_wghts[(stage-1)].beta)*(pmb->pmy_mesh->dt);

    // switch based on sampling
    FCN_CC_CX_VC(
        pbval->ApplyPhysicalBoundaries,
        pbval->ApplyPhysicalCellCenteredXBoundaries,
        pbval->ApplyPhysicalVertexCenteredBoundaries
    )(t_end_stage, dt);

  } else {
    return TaskStatus::fail;
  }
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::EnforceAlgConstr(MeshBlock *pmb, int stage)
{
#ifndef DBG_ALGCONSTR_ALL
  if (stage != nstages) return TaskStatus::success; // only do on last stage
#endif // DBG_ALGCONSTR_ALL

  pmb->pz4c->AlgConstr(pmb->pz4c->storage.u);

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::Z4cToADM(MeshBlock *pmb, int stage) {
  if (stage <= nstages) {
//  if (stage != nstages) return TaskStatus::success;
    pmb->pz4c->Z4cToADM(pmb->pz4c->storage.u, pmb->pz4c->storage.adm);
  } else {
    return TaskStatus::fail;
  }

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::Z4c_Weyl(MeshBlock *pmb, int stage) {
  // only do on last stage
  if (stage != nstages) return TaskStatus::success;

  Mesh *pm = pmb->pmy_mesh;
  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.wave_extraction))
  {
    pmb->pz4c->Z4cWeyl(pmb->pz4c->storage.adm,
                       pmb->pz4c->storage.mat,
                       pmb->pz4c->storage.weyl);
  }

  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::WaveExtract(MeshBlock *pmb, int stage) {
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


//WGC end


TaskStatus GRMHD_Z4c::ADM_Constraints(MeshBlock *pmb, int stage)
{

  if (stage != nstages) return TaskStatus::success;
  Mesh *pm = pmb->pmy_mesh;

  if (CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con) ||
      CurrentTimeCalculationThreshold(pm, &TaskListTriggers.con_hst))
  {      // Time at the end of stage for (u, b) register pair

    pmb->pz4c->ADMConstraints(pmb->pz4c->storage.con, pmb->pz4c->storage.adm,
                              pmb->pz4c->storage.mat, pmb->pz4c->storage.u);

  }
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::NewBlockTimeStep(MeshBlock *pmb, int stage)
{
  if (stage != nstages) return TaskStatus::success; // only do on last stage
//NB using the Z4C version of this fn rather than fluid - potential issue?
  pmb->pz4c->NewBlockTimeStep();
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c::UpdateSource(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {

    // Update matter
    pmb->pz4c->GetMatter(pmb->pz4c->storage.mat, pmb->pz4c->storage.adm, pmb->phydro->w, 
#if USETM
    pmb->pscalars->r,
#endif
    pmb->pfield->bcc);

    return TaskStatus::success;
  }
  return TaskStatus::fail;
}

#ifdef Z4C_ASSERT_FINITE
TaskStatus GRMHD_Z4c::AssertFinite(MeshBlock *pmb, int stage) {
  if (stage != nstages) return TaskStatus::success; // only do on last stage

  pmb->pz4c->assert_is_finite_adm();
  pmb->pz4c->assert_is_finite_con();
  pmb->pz4c->assert_is_finite_mat();
  pmb->pz4c->assert_is_finite_z4c();

  return TaskStatus::success;
}
#endif // Z4C_ASSERT_FINITE
//----------------------------------------------------------------------------------------
// \!fn bool GRMHD_Z4c::CurrentTimeCalculationThreshold(
//   MeshBlock *pmb, aux_NextTimeStep *variable)
// \brief Given current time / ncycles, does a specified 'dt' mean we need
//        to calculate something?
//        Secondary effect is to mutate next_time
bool GRMHD_Z4c::CurrentTimeCalculationThreshold(
  Mesh *pm, aux_NextTimeStep *variable)
{

//  printf("dt = %g \n",variable->dt);
  // this variable is not dumped / computed
  if (variable->dt == 0 )
    return false;

  Real cur_time = pm->time + pm->dt;

//printf("update, time = %g, next time = %g\n", pm->time, variable->next_time);
  if    ((cur_time - pm->dt >= variable->next_time) ||
      (cur_time >= pm->tlim)) {
#pragma omp atomic write
    variable->to_update = true;
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
// \!fn void GRMHD_Z4c::UpdateTaskListTriggers()
// \brief Update 'next_time' outside task list to avoid race condition
void GRMHD_Z4c::UpdateTaskListTriggers() {
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
