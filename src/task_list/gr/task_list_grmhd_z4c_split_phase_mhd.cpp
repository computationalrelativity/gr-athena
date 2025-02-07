// C/C++ headers
#include <cmath>
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
#include "../../utils/linear_algebra.hpp"

#include "task_list.hpp"
#include "task_names.hpp"

// #if M1_ENABLED
// #include "../../m1/m1.hpp"
// #endif

// ----------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;
// ----------------------------------------------------------------------------

GRMHD_Z4c_Phase_MHD::GRMHD_Z4c_Phase_MHD(ParameterInput *pin,
                                         Mesh *pm,
                                         Triggers &trgs)
  : LowStorage(pin, pm),
    trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_MHD;

  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR


  // (M)HD sub-system logic ---------------------------------------------------
  Add(CALC_HYDFLX, NONE, &GRMHD_Z4c_Phase_MHD::CalculateHydroFlux);

  if (NSCALARS > 0)
  {
    Add(CALC_SCLRFLX, CALC_HYDFLX, &GRMHD_Z4c_Phase_MHD::CalculateScalarFlux);
  }

  if (multilevel)
  {
    Add(SEND_HYDFLX, CALC_HYDFLX,
        &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, CALC_HYDFLX,
        &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectHydroFlux);
    Add(INT_HYD,     RECV_HYDFLX,
        &GRMHD_Z4c_Phase_MHD::IntegrateHydro);
  }
  else
  {
    Add(INT_HYD, CALC_HYDFLX, &GRMHD_Z4c_Phase_MHD::IntegrateHydro);
  }

  Add(SRCTERM_HYD, INT_HYD,     &GRMHD_Z4c_Phase_MHD::AddSourceTermsHydro);


  if (NSCALARS > 0)
  {
    if (multilevel)
    {
      Add(SEND_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c_Phase_MHD::SendScalarFlux);
      Add(RECV_SCLRFLX, CALC_SCLRFLX, &GRMHD_Z4c_Phase_MHD::ReceiveScalarFlux);
      Add(INT_SCLR,     RECV_SCLRFLX, &GRMHD_Z4c_Phase_MHD::IntegrateScalars);
    }
    else
    {
      Add(INT_SCLR, CALC_SCLRFLX, &GRMHD_Z4c_Phase_MHD::IntegrateScalars);
    }
  }

  if (MAGNETIC_FIELDS_ENABLED)
  {
    // compute MHD fluxes, integrate field
    Add(CALC_FLDFLX, CALC_HYDFLX, &GRMHD_Z4c_Phase_MHD::CalculateEMF);
    Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionEMF);
    Add(RECV_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectEMF);
    Add(INT_FLD,     RECV_FLDFLX, &GRMHD_Z4c_Phase_MHD::IntegrateField);
  }

  // Collect result of integrations as a final block.
  TaskID FIN = SRCTERM_HYD;

  if (NSCALARS > 0)
  {
    FIN = FIN | INT_SCLR;
  }

  if (MAGNETIC_FIELDS_ENABLED)
  {
    FIN = FIN | INT_FLD;
  }

  // We are done for the (M)HD phase
  Add(CLEAR_ALLBND, FIN, &GRMHD_Z4c_Phase_MHD::ClearAllBoundary);
}

// ----------------------------------------------------------------------------
void GRMHD_Z4c_Phase_MHD::StartupTaskList(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb   = pmb->pbval;
  Z4c            *pz4c = pmb->pz4c;

  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GRMHD_Z4c_Phase_MHD::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with GRMHD"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // Initialize storage registers
    Hydro *ph = pmb->phydro;
    ph->u1.ZeroClear();

    if (MAGNETIC_FIELDS_ENABLED)
    {
      Field *pf = pmb->pfield;
      pf->b1.x1f.ZeroClear();
      pf->b1.x2f.ZeroClear();
      pf->b1.x3f.ZeroClear();
    }

    if (NSCALARS > 0)
    {
      PassiveScalars *ps = pmb->pscalars;
      ps->s1.ZeroClear();
    }
  }

  pb->StartReceiving(BoundaryCommSubset::matter_flux_corrected);
  return;
}

//-----------------------------------------------------------------------------
// Functions to end MPI communication
TaskStatus GRMHD_Z4c_Phase_MHD::ClearAllBoundary(MeshBlock *pmb, int stage)
{
  BoundaryValues *pb = pmb->pbval;
  pb->ClearBoundary(BoundaryCommSubset::matter_flux_corrected);

  // pmb->DebugMeshBlock(-15,-15,-15, 2, 20, 3, "@T:MHD\n", "@E:MHD\n");

  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to calculates fluxes
TaskStatus GRMHD_Z4c_Phase_MHD::CalculateHydroFlux(MeshBlock *pmb, int stage)
{

  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    Reconstruction * pr = pmb->precon;

    int xorder = pmb->precon->xorder;

    if ((stage == 1) && (integrator == "vl2"))
    {
      xorder = 1;
    }

    ph->CalculateFluxes(ph->w, pf->b, pf->bcc, xorder);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_MHD::CalculateEMF(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;

    pf->ComputeCornerE(ph->w, pf->bcc);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus GRMHD_Z4c_Phase_MHD::SendFluxCorrectionHydro(
  MeshBlock *pmb, int stage)
{
  Hydro *ph = pmb->phydro;
  ph->hbvar.SendFluxCorrection();
  return TaskStatus::success;
}

TaskStatus GRMHD_Z4c_Phase_MHD::SendFluxCorrectionEMF(
  MeshBlock *pmb, int stage)
{
  Field *pf = pmb->pfield;
  pf->fbvar.SendFluxCorrection();
  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to receive fluxes between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectHydroFlux(
  MeshBlock *pmb, int stage)
{
  Hydro * ph = pmb->phydro;

  if (ph->hbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  }
  else
  {
    return TaskStatus::fail;
  }
}

TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectEMF(MeshBlock *pmb, int stage)
{
  Field * pf = pmb->pfield;

  if (pf->fbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  } else {
    return TaskStatus::fail;
  }
}

//-----------------------------------------------------------------------------
// Functions to integrate conserved variables
TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    Field *pf = pmb->pfield;

    // See IntegrateField
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ph->u1, ph->u, ph->u2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    pmb->WeightedAveCC(ph->u, ph->u1, ph->u2, ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    ph->AddFluxDivergence(dt_scaled, ph->u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateField(MeshBlock *pmb, int stage)
{
  Field *pf = pmb->pfield;

  if (stage <= nstages)
  {
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveFC(pf->b1, pf->b, pf->b2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    pmb->WeightedAveFC(pf->b, pf->b1, pf->b2, ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pf->CT(dt_scaled, pf->b);

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to add source terms
TaskStatus GRMHD_Z4c_Phase_MHD::AddSourceTermsHydro(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro * ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    Field * pf = pmb->pfield;
    Coordinates * pc = pmb->pcoord;

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // add coordinate (geometric) source terms
#if USETM
    pc->AddCoordTermsDivergence(dt_scaled, ph->flux, ph->w,
                                ps->r, pf->bcc, ph->u);
#else
    pc->AddCoordTermsDivergence(dt_scaled, ph->flux, ph->w, pf->bcc, ph->u);
#endif

// #if M1_ENABLED
//     ::M1::M1 * pm1 = pmb->pm1;
//     if (pm1->opt.couple_sources_hydro)
//     {
//       pm1->CoupleSourcesHydro(dt_scaled, ph->u);
//     }
// #endif

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD::CalculateScalarFlux(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars *ps = pmb->pscalars;

    if ((stage == 1) && (integrator == "vl2"))
    {
      ps->CalculateFluxes(ps->r, 1);
      return TaskStatus::next;
    }
    else
    {
      ps->CalculateFluxes(ps->r, pmb->precon->xorder);
      return TaskStatus::next;
    }
  }
  return TaskStatus::fail;
}


TaskStatus GRMHD_Z4c_Phase_MHD::SendScalarFlux(MeshBlock *pmb, int stage)
{
  PassiveScalars * ps = pmb->pscalars;

  ps->sbvar.SendFluxCorrection();
  return TaskStatus::success;
}


TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveScalarFlux(MeshBlock *pmb, int stage)
{
  PassiveScalars * ps = pmb->pscalars;

  if (ps->sbvar.ReceiveFluxCorrection())
  {
    return TaskStatus::next;
  }
  else
  {
    return TaskStatus::fail;
  }
}


TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    PassiveScalars * ps = pmb->pscalars;

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

    pmb->WeightedAveCC(ps->s, ps->s1, ps->s2, ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    ps->AddFluxDivergence(dt_scaled, ps->s);

// #if M1_ENABLED & USETM
//     ::M1::M1 * pm1 = pmb->pm1;

//     if (pm1->opt.couple_sources_Y_e)
//     {
//       if (pm1->N_SPCS != 3)
//       #pragma omp critical
//       {
//         std::cout << "M1: couple_sources_Y_e supported for 3 species \n";
//         std::exit(0);
//       }

//       const Real mb = pmb->peos->GetEOS().GetBaryonMass();
//       pm1->CoupleSourcesYe(dt_scaled, mb, ps->s);
//     }
// #endif

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}


//
// :D
//
