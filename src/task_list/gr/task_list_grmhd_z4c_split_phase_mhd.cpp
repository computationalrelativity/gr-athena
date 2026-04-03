// C/C++ headers
#include <cmath>
#include <iostream>  // endl
#include <limits>
#include <memory>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Athena++ classes headers
#include "../../athena.hpp"
#include "../../comm/comm_channel.hpp"
#include "../../comm/comm_enums.hpp"
#include "../../comm/comm_registry.hpp"
#include "../../eos/eos.hpp"
#include "../../field/field.hpp"
#include "../../hydro/hydro.hpp"
#include "../../mesh/mesh.hpp"
#include "../../reconstruct/reconstruction.hpp"
#include "../../scalars/scalars.hpp"
#include "../../trackers/extrema_tracker.hpp"
#include "../../utils/linear_algebra.hpp"
#include "../../z4c/puncture_tracker.hpp"
#include "../../z4c/wave_extract.hpp"
#include "../../z4c/z4c.hpp"
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

GRMHD_Z4c_Phase_MHD::GRMHD_Z4c_Phase_MHD(ParameterInput* pin,
                                         Mesh* pm,
                                         Triggers& trgs)
    : LowStorage(pin, pm), trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Split::Phase_MHD;

  // Take the number of stages from the integrator
  nstages = LowStorage::nstages;

  //---------------------------------------------------------------------------

  const bool multilevel = pm->multilevel;  // for SMR or AMR logic
  const bool adaptive   = pm->adaptive;    // AMR

  // (M)HD sub-system logic ---------------------------------------------------

  // weighted average of prior stages
  Add(INT_HYDSCLR, NONE, &GRMHD_Z4c_Phase_MHD::IntegrateHydroScalars);

  // hydro + scalar fluxes
  Add(CALC_HYDSCLRFLX,
      INT_HYDSCLR,
      &GRMHD_Z4c_Phase_MHD::CalculateHydroScalarFlux);

  // collect receive operations
  TaskID RECV_FLX = NONE;

  if (multilevel)
  {
    RECV_FLX = RECV_FLX | RECV_HYDFLX;

    Add(SEND_HYDFLX,
        CALC_HYDSCLRFLX,
        &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, NONE, &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectHydroFlux);
  }

  if (NSCALARS > 0)
  {
    if (multilevel)
    {
      RECV_FLX = RECV_FLX | RECV_SCLRFLX;

      Add(SEND_SCLRFLX, CALC_HYDSCLRFLX, &GRMHD_Z4c_Phase_MHD::SendScalarFlux);
      Add(RECV_SCLRFLX, NONE, &GRMHD_Z4c_Phase_MHD::ReceiveScalarFlux);
    }
  }

  // Are fluxes prepared (& sent if req.) for hydro / scalar?
  // If so, we can add flux divergence
  Add(ADD_FLX_DIV,
      CALC_HYDSCLRFLX | RECV_FLX,
      &GRMHD_Z4c_Phase_MHD::AddFluxDivergenceHydroScalars);

  // Then add sources (which completes the hydro step)
  Add(SRCTERM_HYD, ADD_FLX_DIV, &GRMHD_Z4c_Phase_MHD::AddSourceTermsHydro);

  if (MAGNETIC_FIELDS_ENABLED)
  {
    // compute corner EMFs from face EMFs + weights produced by the Riemann
    // solver in CALC_HYDSCLRFLX, then integrate the magnetic field via CT
    Add(CALC_FLDFLX, CALC_HYDSCLRFLX, &GRMHD_Z4c_Phase_MHD::CalculateEMF);
    Add(SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Phase_MHD::SendFluxCorrectionEMF);
    Add(RECV_FLDFLX, NONE, &GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectEMF);
    Add(INT_FLD, RECV_FLDFLX, &GRMHD_Z4c_Phase_MHD::IntegrateField);
  }

  // Collect result of integrations as a final block.
  TaskID FIN = SRCTERM_HYD;

  if (MAGNETIC_FIELDS_ENABLED)
  {
    FIN = FIN | INT_FLD | SEND_FLDFLX;
  }

  // We are done for the (M)HD phase
  Add(CLEAR_ALLBND, FIN, &GRMHD_Z4c_Phase_MHD::ClearAllBoundary);
}

// ----------------------------------------------------------------------------
void GRMHD_Z4c_Phase_MHD::StartupTaskList(MeshBlock* pmb, int stage)
{
  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GRMHD_Z4c_Phase_MHD::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with GRMHD" << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // Initialize storage registers
    Reconstruction* pr = pmb->precon;
    Hydro* ph          = pmb->phydro;
    ph->u1.ZeroClear();

    // copy for comparison
    if (pr->xorder_use_dmp)
    {
      ph->u2 = ph->u;
      if (NSCALARS > 0)
      {
        PassiveScalars* ps = pmb->pscalars;
        ps->s2             = ps->s;
      }
    }

    if (MAGNETIC_FIELDS_ENABLED)
    {
      Field* pf = pmb->pfield;
      pf->b1.x1f.ZeroClear();
      pf->b1.x2f.ZeroClear();
      pf->b1.x3f.ZeroClear();
    }

    if (NSCALARS > 0)
    {
      PassiveScalars* ps = pmb->pscalars;
      ps->s1.ZeroClear();
    }
  }

  // Post persistent receives for MHD-owned flux correction channels only.
  // Per-channel startup avoids activating M1's channel in the shared FluxCorr
  // group, which would cause a double MPI_Start when M1 runs its own startup.
  if (pmb->pmy_mesh->multilevel)
  {
    pmb->pcomm->StartReceivingFluxCorrSingleChannel(
      pmb->phydro->comm_channel_id);
    if (MAGNETIC_FIELDS_ENABLED)
      pmb->pcomm->StartReceivingFluxCorrSingleChannel(
        pmb->pfield->comm_channel_id);
    if (NSCALARS > 0)
      pmb->pcomm->StartReceivingFluxCorrSingleChannel(
        pmb->pscalars->comm_channel_id);
  }

  return;
}

//-----------------------------------------------------------------------------
// Wait on MHD-owned flux correction sends and reset channel flags.
// Per-channel clear avoids touching M1's channel in the shared FluxCorr group.
// Non-blocking (wait=false): return fail to let the work-stealing scheduler
// advance other blocks while sends complete, avoiding MPI_Wait deadlocks.
TaskStatus GRMHD_Z4c_Phase_MHD::ClearAllBoundary(MeshBlock* pmb, int stage)
{
  if (pmb->pmy_mesh->multilevel)
  {
    if (!pmb->pcomm->ClearFluxCorrSingleChannel(pmb->phydro->comm_channel_id,
                                                false))
      return TaskStatus::fail;
    if (MAGNETIC_FIELDS_ENABLED)
      if (!pmb->pcomm->ClearFluxCorrSingleChannel(pmb->pfield->comm_channel_id,
                                                  false))
        return TaskStatus::fail;
    if (NSCALARS > 0)
      if (!pmb->pcomm->ClearFluxCorrSingleChannel(
            pmb->pscalars->comm_channel_id, false))
        return TaskStatus::fail;
  }
  return TaskStatus::next;
}

//-----------------------------------------------------------------------------
// Functions to calculates fluxes
TaskStatus GRMHD_Z4c_Phase_MHD::CalculateHydroScalarFlux(MeshBlock* pmb,
                                                         int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph          = pmb->phydro;
    Field* pf          = pmb->pfield;
    Reconstruction* pr = pmb->precon;
    PassiveScalars* ps = pmb->pscalars;

    const int num_enlarge_layer =
      (pr->xorder_use_fb || pr->xorder_limit_fluxes) ? 1 : 0;
    AA(&hflux)[3] = ph->flux;
    AA(&sflux)[3] = ps->s_flux;

    // If we can hybridize (fb) and the current block has a density all within
    // xorder_fb_dfloor_fac then we jump immediately to LO
    bool xorder_use_fb                                 = pr->xorder_use_fb;
    Reconstruction::ReconstructionVariant xorder_style = pr->xorder_style;

    if ((xorder_use_fb) && (pr->xorder_fb_dfloor_fac > 1) &&
        (pmb->peos->ConservedDensityWithinFloorThreshold(
          ph->u,
          pmb->pz4c->aux_extended.ms_sqrt_detgamma.array(),
          pr->xorder_fb_dfloor_fac,
          pmb->is - num_enlarge_layer,
          pmb->ie + num_enlarge_layer,
          pmb->js - num_enlarge_layer,
          pmb->je + num_enlarge_layer,
          pmb->ks - num_enlarge_layer,
          pmb->ke + num_enlarge_layer)))
    {
      xorder_style  = pr->xorder_style_fb;
      xorder_use_fb = false;
    }

    // Obtain per-thread geometry cache when hybridization is active.
    // The HO pass stores FC geometry into the cache; the LO fallback
    // pass loads it instead of re-interpolating.
    ThreadCache* cache = nullptr;
    if (xorder_use_fb)
    {
#ifdef OPENMP_PARALLEL
      cache = &pmb->pmy_mesh->thread_cache(omp_get_thread_num());
#else
      cache = &pmb->pmy_mesh->thread_cache(0);
#endif
    }

    ph->CalculateFluxes(ph->w,
                        ps->r,
                        pf->b,
                        pf->bcc,
                        hflux,
                        sflux,
                        xorder_style,
                        num_enlarge_layer,
                        cache);

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
                                       ph->u,
                                       ps->s,
                                       hflux,
                                       sflux,
                                       all_valid,
                                       mask,
                                       num_enlarge_layer);

      // TODO: on cross-block consistency of the DMP mask:
      // Unlike the check about which uses pre-comm ph->u
      // the DMP check compares against ph->u2, a snapshot of ph->u taken at
      // the start of the timestep. This latter means adjustment / floor may
      // have occurred. It is therefore possible (though very rare) that
      // inconsistent masks and mismatched fluxes at same-level faces could
      // occur.
      if (pr->xorder_use_dmp)
      {
        ph->CheckStateWithFluxDivergenceDMP(dt_scaled,
                                            ph->u,
                                            ph->u2,
                                            ps->s,
                                            ps->s2,
                                            hflux,
                                            sflux,
                                            all_valid,
                                            mask,
                                            num_enlarge_layer);
      }

      if (!all_valid)
      {
        AA(&lo_hflux)[3] = cache->lo_hflux;
        AA(&lo_sflux)[3] = cache->lo_sflux;

        ph->CalculateFluxesCachedGeometry(ph->w,
                                          ps->r,
                                          pf->b,
                                          pf->bcc,
                                          lo_hflux,
                                          lo_sflux,
                                          pr->xorder_style_fb,
                                          num_enlarge_layer,
                                          *cache,
                                          mask);

        ph->HybridizeFluxes(hflux, sflux, lo_hflux, lo_sflux, mask);
      }

      CC_GLOOP3(k, j, i)
      {
        ph->fallback_mask(k, j, i) = mask(k, j, i);
      }

      skip_limit = all_valid;
    }

    if (pr->xorder_limit_fluxes && !skip_limit)
    {
      const Real dt_scaled = this->dt_scaled(stage, pmb);
      AA mask_theta(pmb->ncells3, pmb->ncells2, pmb->ncells1);

      ph->LimitMaskFluxDivergence(
        dt_scaled, ph->u, ps->s, hflux, sflux, mask_theta, num_enlarge_layer);

      ph->LimitFluxes(mask_theta, hflux, sflux);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Phase_MHD::CalculateEMF(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph = pmb->phydro;
    Field* pf = pmb->pfield;

    pf->ComputeCornerE(ph->w, pf->bcc);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Communicate fluxes between MeshBlocks for flux correction with AMR
TaskStatus GRMHD_Z4c_Phase_MHD::SendFluxCorrectionHydro(MeshBlock* pmb,
                                                        int stage)
{
  // Per-channel send: hydro CC flux correction.  Scalar and EMF channels are
  // sent separately because their data may not be ready yet (EMF requires
  // CalculateEMF which runs concurrently).
  int hid = pmb->phydro->comm_channel_id;
  if (hid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(hid);

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_MHD::SendFluxCorrectionEMF(MeshBlock* pmb,
                                                      int stage)
{
  // Per-channel send: field (EMF) FC flux correction.
  int fid = pmb->pfield->comm_channel_id;
  if (fid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(fid);

  return TaskStatus::next;
}

//-----------------------------------------------------------------------------
// Functions to receive fluxes between MeshBlocks
TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectHydroFlux(MeshBlock* pmb,
                                                           int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  // Poll hydro flux correction channel.
  int hid = pmb->phydro->comm_channel_id;
  if (hid >= 0)
  {
    if (!pcomm->channel(hid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(hid).UnpackFluxCorr(nc);
  }

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveAndCorrectEMF(MeshBlock* pmb, int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  // Poll field (EMF) flux correction channel.
  // For FC AccumulateAverage, PollReceiveFluxCorrFC runs the two-phase
  // state machine (same-level clear -> cross-level average) internally.
  // UnpackFluxCorr is a no-op for FC.
  int fid = pmb->pfield->comm_channel_id;
  if (fid >= 0)
  {
    if (!pcomm->channel(fid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(fid).UnpackFluxCorr(nc);
  }

  return TaskStatus::next;
}

//-----------------------------------------------------------------------------
// Functions to integrate conserved variables
TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateField(MeshBlock* pmb, int stage)
{
  Field* pf = pmb->pfield;

  if (stage <= nstages)
  {
    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage - 1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveFC(pf->b1, pf->b, pf->b2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage - 1].gamma_1;
    ave_wghts[1] = stage_wghts[stage - 1].gamma_2;
    ave_wghts[2] = stage_wghts[stage - 1].gamma_3;

    pmb->WeightedAveFC(pf->b, pf->b1, pf->b2, ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pf->CT(dt_scaled, pf->b);

    return TaskStatus::next;
  }

  return TaskStatus::fail;
}

//-----------------------------------------------------------------------------
// Functions to add source terms
TaskStatus GRMHD_Z4c_Phase_MHD::AddSourceTermsHydro(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph          = pmb->phydro;
    PassiveScalars* ps = pmb->pscalars;
    Field* pf          = pmb->pfield;
    Coordinates* pc    = pmb->pcoord;

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // add coordinate (geometric) source terms
#if FLUID_ENABLED
    pc->AddCoordTermsDivergence(
      dt_scaled, ph->flux, ph->w, ps->r, pf->bcc, ph->u);
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

TaskStatus GRMHD_Z4c_Phase_MHD::SendScalarFlux(MeshBlock* pmb, int stage)
{
  // Per-channel send: scalar CC flux correction.
  int sid = pmb->pscalars->comm_channel_id;
  if (sid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(sid);

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_MHD::ReceiveScalarFlux(MeshBlock* pmb, int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  // Poll scalar flux correction channel.
  int sid = pmb->pscalars->comm_channel_id;
  if (sid >= 0)
  {
    if (!pcomm->channel(sid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(sid).UnpackFluxCorr(nc);
  }

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Phase_MHD::IntegrateHydroScalars(MeshBlock* pmb,
                                                      int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph          = pmb->phydro;
    PassiveScalars* ps = pmb->pscalars;
    // Field *pf = pmb->pfield;
    Reconstruction* pr    = pmb->precon;
    EquationOfState* peos = pmb->peos;

    const int num_enlarge_layer =
      (pr->xorder_use_fb || pr->xorder_limit_fluxes) ? 1 : 0;

    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage - 1].delta;
    ave_wghts[2] = 0.0;
    pmb->WeightedAveCC(ph->u1, ph->u, ph->u2, ave_wghts, num_enlarge_layer);

    if (NSCALARS > 0)
      pmb->WeightedAveCC(ps->s1, ps->s, ps->s2, ave_wghts, num_enlarge_layer);

    ave_wghts[0] = stage_wghts[stage - 1].gamma_1;
    ave_wghts[1] = stage_wghts[stage - 1].gamma_2;
    ave_wghts[2] = stage_wghts[stage - 1].gamma_3;

    pmb->WeightedAveCC(ph->u, ph->u1, ph->u2, ave_wghts, num_enlarge_layer);

    if (NSCALARS > 0)
      pmb->WeightedAveCC(ps->s, ps->s1, ps->s2, ave_wghts, num_enlarge_layer);

    // Ensure update does not dip below floor ---------------------------------
    if (pr->enforce_limits_integration)
    {
      ph->EnforceFloorsLimits(ph->u, ps->s, num_enlarge_layer);
    }
    // ------------------------------------------------------------------------

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Phase_MHD::AddFluxDivergenceHydroScalars(MeshBlock* pmb,
                                                              int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph          = pmb->phydro;
    PassiveScalars* ps = pmb->pscalars;
    Reconstruction* pr = pmb->precon;

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    ph->AddFluxDivergence(dt_scaled, ph->u);

    if (NSCALARS > 0)
      ps->AddFluxDivergence(dt_scaled, ps->s);

    // Ensure update does not dip below floor ---------------------------------
    const int num_enlarge_layer = 0;
    if (pr->enforce_limits_flux_div)
    {
      ph->EnforceFloorsLimits(ph->u, ps->s, num_enlarge_layer);
    }
    // ------------------------------------------------------------------------

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

//
// :D
//
