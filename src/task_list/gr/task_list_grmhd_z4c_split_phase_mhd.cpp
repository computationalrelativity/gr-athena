// C/C++ headers
#include <cmath>
#include <iostream>   // endl
#include <limits>
#include <memory>
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

#ifdef USE_COMM_DEPENDENCY
  // Accumulate MPI communication tasks:
  TaskID COMM = NONE;
#endif

  // (M)HD sub-system logic ---------------------------------------------------

  // weighted average of prior stages
  Add(INT_HYDSCLR, NONE, &GRMHD_Z4c_Phase_MHD::IntegrateHydroScalars);

  // hydro + scalar fluxes
  Add(CALC_HYDSCLRFLX, INT_HYDSCLR,
      &GRMHD_Z4c_Phase_MHD::CalculateHydroScalarFlux);

  // collect receive operations
  TaskID RECV_FLX = NONE;
#ifdef USE_COMM_DEPENDENCY
  COMM = COMM | RECV_FLX;
#endif

  if (multilevel)
  {
    RECV_FLX = RECV_FLX | RECV_HYDFLX;

    Add(SEND_HYDFLX, CALC_HYDSCLRFLX,
        &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, CALC_HYDSCLRFLX,
        &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectHydroFlux);
#ifdef USE_COMM_DEPENDENCY
    COMM = COMM | SEND_HYDFLX | RECV_HYDFLX;
#endif
  }

  if (NSCALARS > 0)
  {
    if (multilevel)
    {
      RECV_FLX = RECV_FLX | RECV_SCLRFLX;

      Add(SEND_SCLRFLX, CALC_HYDSCLRFLX,
          &GRMHD_Z4c_Phase_MHD::SendScalarFlux);
      Add(RECV_SCLRFLX, CALC_HYDSCLRFLX,
          &GRMHD_Z4c_Phase_MHD::ReceiveScalarFlux);

#ifdef USE_COMM_DEPENDENCY
      COMM = COMM | SEND_SCLRFLX | RECV_SCLRFLX;
#endif
    }
  }

  // Are fluxes prepared (& sent if req.) for hydro / scalar?
  // If so, we can add flux divergence
  Add(ADD_FLX_DIV, CALC_HYDSCLRFLX | RECV_FLX,
      &GRMHD_Z4c_Phase_MHD::AddFluxDivergenceHydroScalars);

  // Then add sources (which completes the hydro step)
  Add(SRCTERM_HYD, ADD_FLX_DIV,
      &GRMHD_Z4c_Phase_MHD::AddSourceTermsHydro);

  if (MAGNETIC_FIELDS_ENABLED)
  {
    // compute MHD fluxes (can be done whenever), integrate field
    Add(CALC_FLDFLX, NONE, &GRMHD_Z4c_Phase_MHD::CalculateEMF);
    Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionEMF);
    Add(RECV_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectEMF);
#ifdef USE_COMM_DEPENDENCY
    COMM = COMM | SEND_FLDFLX | RECV_FLDFLX;
#endif
    Add(INT_FLD,     RECV_FLDFLX, &GRMHD_Z4c_Phase_MHD::IntegrateField);
  }

  // Collect result of integrations as a final block.
  TaskID FIN = SRCTERM_HYD;

  if (MAGNETIC_FIELDS_ENABLED)
  {
    FIN = FIN | INT_FLD | SEND_FLDFLX;
  }

#ifdef USE_COMM_DEPENDENCY
  // We are done with MPI communication
  Add(CLEAR_ALLBND, COMM, &GRMHD_Z4c_Phase_MHD::ClearAllBoundary);
#else
  // We are done for the (M)HD phase
  Add(CLEAR_ALLBND, FIN, &GRMHD_Z4c_Phase_MHD::ClearAllBoundary);
#endif

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
    Reconstruction *pr = pmb->precon;
    Hydro *ph = pmb->phydro;
    ph->u1.ZeroClear();

    // copy for comparison
    if (pr->xorder_use_dmp)
    {
      ph->u2 = ph->u;
      if (NSCALARS > 0)
      {
        PassiveScalars *ps = pmb->pscalars;
        ps->s2 = ps->s;
      }
    }

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

  return TaskStatus::success;
}

//-----------------------------------------------------------------------------
// Functions to calculates fluxes
TaskStatus GRMHD_Z4c_Phase_MHD::CalculateHydroScalarFlux(
  MeshBlock *pmb, int stage
)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    Field *pf = pmb->pfield;
    Reconstruction * pr = pmb->precon;
    PassiveScalars *ps = pmb->pscalars;

    if (pmb->storage.trivialize_fields.hydro.is_trivial)
    {
      return TaskStatus::next;
    }

    const int num_enlarge_layer = (pr->xorder_use_fb ||
                                   pr->xorder_limit_fluxes) ? 1 : 0;
    AA(& hflux)[3] = ph->flux;
    AA(& sflux)[3] = ps->s_flux;

    // If we can hybridize (fb) and the current block has a density all within
    // xorder_fb_dfloor_fac then we jump immediately to LO
    bool xorder_use_fb = pr->xorder_use_fb;
    Reconstruction::ReconstructionVariant xorder_style = pr->xorder_style;

    if ((xorder_use_fb) &&
        (pr->xorder_fb_dfloor_fac > 1) &&
        (ph->ConservedDensityWithinFloorThreshold(ph->u,
                                                  pr->xorder_fb_dfloor_fac,
                                                  num_enlarge_layer)))
    {
      xorder_style = pr->xorder_style_fb;
      xorder_use_fb = false;
    }

    ph->CalculateFluxes(ph->w, ps->r, pf->b, pf->bcc,
                        hflux, sflux,
                        xorder_style,
                        num_enlarge_layer);

    // if we fall-back and the state is all valid, no need to lmit
    bool skip_limit = false;

    // Logic to test candidate state can go here. This is pre-flux-correction.
    if (xorder_use_fb)
    {
      bool all_valid = true;
      AA_B mask(pmb->ncells3, pmb->ncells2, pmb->ncells1);
      mask.Fill(true);

      const Real dt_scaled = this->dt_scaled(stage, pmb);

      ph->CheckStateWithFluxDivergence(dt_scaled,
                                       ph->u, ps->s,
                                       hflux, sflux,
                                       all_valid, mask,
                                       num_enlarge_layer);

      if (pr->xorder_use_dmp)
      {
        ph->CheckStateWithFluxDivergenceDMP(
          dt_scaled,
          ph->u, ph->u2,
          ps->s, ps->s2,
          hflux, sflux,
          all_valid, mask,
          num_enlarge_layer
        );
      }

      if (!all_valid)
      {
        AA(& lo_hflux)[3] = ph->lo_flux;
        AA(& lo_sflux)[3] = ps->lo_s_flux;

        ph->CalculateFluxes(ph->w, ps->r, pf->b, pf->bcc,
                            lo_hflux, lo_sflux,
                            pr->xorder_style_fb,
                            num_enlarge_layer);

        ph->HybridizeFluxes(hflux, sflux, lo_hflux, lo_sflux, mask);
      }

      CC_GLOOP3(k, j, i)
      {
        ph->fallback_mask(k,j,i) = mask(k,j,i);
      }

      skip_limit = all_valid;
    }

    if (pr->xorder_limit_fluxes && !skip_limit)
    {
      const Real dt_scaled = this->dt_scaled(stage, pmb);
      AA mask_theta(pmb->ncells3, pmb->ncells2, pmb->ncells1);

      ph->LimitMaskFluxDivergence(
        dt_scaled,
        ph->u, ps->s,
        hflux, sflux,
        mask_theta,
        num_enlarge_layer
      );

      ph->LimitFluxes(
        mask_theta,
        hflux, sflux
      );
    }

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
    gra::trivialize::TrivializeFields * ptrif = pmb->pmy_mesh->ptrif;

    if (pmb->storage.trivialize_fields.hydro.is_trivial)
    {
      return TaskStatus::next;
    }

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pmb->CheckFieldsFinite("pre_coord_div", false);

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

    pmb->CheckFieldsFinite("post_coord_div", false);

    return TaskStatus::next;
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

TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateHydroScalars(MeshBlock *pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    PassiveScalars *ps = pmb->pscalars;
    // Field *pf = pmb->pfield;
    Reconstruction *pr = pmb->precon;
    EquationOfState *peos = pmb->peos;

    const int num_enlarge_layer = (pr->xorder_use_fb ||
                                   pr->xorder_limit_fluxes) ? 1 : 0;

    pmb->CheckFieldsFinite("pre_integ", false);

    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage-1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ph->u1, ph->u, ph->u2, ave_wghts, num_enlarge_layer);

    if (NSCALARS > 0)
      pmb->WeightedAveCC(ps->s1, ps->s, ps->s2, ave_wghts, num_enlarge_layer);

    ave_wghts[0] = stage_wghts[stage-1].gamma_1;
    ave_wghts[1] = stage_wghts[stage-1].gamma_2;
    ave_wghts[2] = stage_wghts[stage-1].gamma_3;

    pmb->WeightedAveCC(ph->u, ph->u1, ph->u2, ave_wghts, num_enlarge_layer);

    if (NSCALARS > 0)
      pmb->WeightedAveCC(ps->s, ps->s1, ps->s2, ave_wghts, num_enlarge_layer);

    // Ensure update does not dip below floor ---------------------------------
    if (pr->enforce_limits_integration)
    {
      ph->EnforceFloorsLimits(
        ph->u, ps->s, num_enlarge_layer
      );
    }
    // ------------------------------------------------------------------------


    pmb->CheckFieldsFinite("post_integ", false);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD::AddFluxDivergenceHydroScalars(
  MeshBlock *pmb, int stage
)
{
  if (stage <= nstages)
  {
    Hydro *ph = pmb->phydro;
    PassiveScalars * ps = pmb->pscalars;
    Reconstruction *pr = pmb->precon;
    gra::trivialize::TrivializeFields * ptrif = pmb->pmy_mesh->ptrif;

    if (pmb->storage.trivialize_fields.hydro.is_trivial)
    {
      return TaskStatus::next;
    }

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pmb->CheckFieldsFinite("pre_flx", false);

    ph->AddFluxDivergence(dt_scaled, ph->u);

    if (NSCALARS > 0)
      ps->AddFluxDivergence(dt_scaled, ps->s);

    pmb->CheckFieldsFinite("post_flx", false);

    // Ensure update does not dip below floor ---------------------------------
    const int num_enlarge_layer = 0;
    if (pr->enforce_limits_flux_div)
    {
      ph->EnforceFloorsLimits(
        ph->u, ps->s, num_enlarge_layer
      );
    }
    // ------------------------------------------------------------------------

    // update masks so flux flows into new cells
    if (ptrif->opt.hydro.active)
    {
      ptrif->PrepareMask(pmb);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//
// :D
//
