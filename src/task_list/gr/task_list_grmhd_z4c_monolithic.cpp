// ===========================================================================
// Monolithic GRMHD+Z4c task list
//
// Fuses the four split phases (Phase_MHD, Phase_Z4c, Phase_MHD_com,
// Phase_Finalize) into a single DAG executed by one DoTaskListOneStage call
// per integrator substep.  This eliminates all inter-phase barriers and
// allows the scheduler to overlap MHD and Z4c computation across MeshBlocks,
// overlap MPI communication with C2P, and start purely geometric tasks
// (NewBlockTimeStep, CheckRefinement) immediately.
//
// Key optimization:  the conserved-variable ghost send (SEND_HYD) depends
// only on MHD integration completing, NOT on C2P.  MainInt channels
// communicate conserved variables (u, b, s), so the send fires before the
// expensive C2P inversion, hiding packing + MPI latency behind C2P compute.
//
// Selected at runtime when hydro/use_split_grmhd_z4c = false (the default).
// ===========================================================================

// C/C++ headers
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Athena++ headers
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

#if CCE_ENABLED
#include "../../z4c/cce/cce.hpp"
#endif

// ---------------------------------------------------------------------------
using namespace TaskLists::GeneralRelativity;
using namespace TaskLists::Integrators;

using namespace gra::triggers;
typedef Triggers::TriggerVariant TriggerVariant;

// ===========================================================================
// Constructor - wires the monolithic DAG
// ===========================================================================
GRMHD_Z4c_Monolithic::GRMHD_Z4c_Monolithic(ParameterInput* pin,
                                           Mesh* pm,
                                           Triggers& trgs)
    : LowStorage(pin, pm), trgs(trgs)
{
  using namespace TaskNames::GeneralRelativity::GRMHD_Z4c_Monolithic;

  nstages = LowStorage::nstages;

  const bool multilevel = pm->multilevel;
  const bool adaptive   = pm->adaptive;

  // =========================================================================
  // MHD compute branch - from NONE
  // =========================================================================

  // Weighted average of prior stages (initialises u for the new substep)
  Add(INT_HYDSCLR, NONE, &GRMHD_Z4c_Monolithic::IntegrateHydroScalars);

  // Hydro + scalar fluxes
  Add(CALC_HYDSCLRFLX,
      INT_HYDSCLR,
      &GRMHD_Z4c_Monolithic::CalculateHydroScalarFlux);

  // Flux correction send/recv for AMR (multilevel only)
  TaskID RECV_FLX = NONE;

  if (multilevel)
  {
    RECV_FLX = RECV_FLX | RECV_HYDFLX;

    Add(SEND_HYDFLX,
        CALC_HYDSCLRFLX,
        &GRMHD_Z4c_Monolithic::SendFluxCorrectionHydro);
    Add(RECV_HYDFLX, NONE, &GRMHD_Z4c_Monolithic::ReceiveAndCorrectHydroFlux);
  }

  if (NSCALARS > 0)
  {
    if (multilevel)
    {
      RECV_FLX = RECV_FLX | RECV_SCLRFLX;

      Add(
        SEND_SCLRFLX, CALC_HYDSCLRFLX, &GRMHD_Z4c_Monolithic::SendScalarFlux);
      Add(RECV_SCLRFLX, NONE, &GRMHD_Z4c_Monolithic::ReceiveScalarFlux);
    }
  }

  // Flux divergence (waits for fluxes AND any flux corrections)
  Add(ADD_FLX_DIV,
      CALC_HYDSCLRFLX | RECV_FLX,
      &GRMHD_Z4c_Monolithic::AddFluxDivergenceHydroScalars);

  // Coordinate/geometric source terms (completes the hydro integration)
  Add(SRCTERM_HYD, ADD_FLX_DIV, &GRMHD_Z4c_Monolithic::AddSourceTermsHydro);

  // Magnetic field: EMF corner calc, flux-correction exchange, CT integration
  if (MAGNETIC_FIELDS_ENABLED)
  {
    Add(CALC_FLDFLX, CALC_HYDSCLRFLX, &GRMHD_Z4c_Monolithic::CalculateEMF);
    Add(
      SEND_FLDFLX, CALC_FLDFLX, &GRMHD_Z4c_Monolithic::SendFluxCorrectionEMF);
    Add(RECV_FLDFLX, NONE, &GRMHD_Z4c_Monolithic::ReceiveAndCorrectEMF);
    Add(INT_FLD, RECV_FLDFLX, &GRMHD_Z4c_Monolithic::IntegrateField);
  }

  // Aggregate dependency that marks "MHD integration is complete"
  TaskID FIN_MHD = SRCTERM_HYD;
  if (MAGNETIC_FIELDS_ENABLED)
  {
    FIN_MHD = FIN_MHD | INT_FLD | SEND_FLDFLX;
  }

  // =========================================================================
  // Z4c compute branch - from NONE, fully independent of MHD
  // =========================================================================

  Add(INIT_Z4C_DERIV, NONE, &GRMHD_Z4c_Monolithic::InitializeZ4cDerivatives);
  Add(CALC_Z4CRHS, INIT_Z4C_DERIV, &GRMHD_Z4c_Monolithic::CalculateZ4cRHS);
  Add(INT_Z4C, CALC_Z4CRHS, &GRMHD_Z4c_Monolithic::IntegrateZ4c);

  // Z4c ghost-zone exchange (send after integration, recv polls from NONE)
  Add(SEND_Z4C, INT_Z4C, &GRMHD_Z4c_Monolithic::SendZ4c);
  Add(RECV_Z4C, NONE, &GRMHD_Z4c_Monolithic::ReceiveZ4c);

  Add(SETB_Z4C, (RECV_Z4C | INT_Z4C), &GRMHD_Z4c_Monolithic::SetBoundariesZ4c);

  if (multilevel)
  {
    Add(PROLONG_Z4C,
        (SEND_Z4C | SETB_Z4C),
        &GRMHD_Z4c_Monolithic::Prolongation_Z4c);
    Add(
      PHY_BVAL_Z4C, PROLONG_Z4C, &GRMHD_Z4c_Monolithic::PhysicalBoundary_Z4c);
  }
  else
  {
    Add(PHY_BVAL_Z4C, SETB_Z4C, &GRMHD_Z4c_Monolithic::PhysicalBoundary_Z4c);
  }

  Add(ALG_CONSTR, PHY_BVAL_Z4C, &GRMHD_Z4c_Monolithic::EnforceAlgConstr);

  // Pre-compute conformal derivative 3D arrays from post-comm z4c.u
  Add(
    PREP_Z4C_DERIV, ALG_CONSTR, &GRMHD_Z4c_Monolithic::PrepareZ4cDerivatives);

  Add(Z4C_TO_ADM, PREP_Z4C_DERIV, &GRMHD_Z4c_Monolithic::Z4cToADM);

#if CCE_ENABLED
  Add(CCE_DUMP, Z4C_TO_ADM, &GRMHD_Z4c_Monolithic::CCEDump);
#endif

  // =========================================================================
  // MHD ghost-zone send + C2P (send conserved BEFORE C2P!)
  // =========================================================================

  // Conserved-variable ghost send fires as soon as MHD integration completes.
  // MainInt channels pack u/b/s (conserved), so no dependency on C2P.
  Add(SEND_HYD, FIN_MHD, &GRMHD_Z4c_Monolithic::SendHydro);

  // Wait on flux-correction sends and reset channel flags (multilevel only).
  if (multilevel)
  {
    Add(CLEAR_FLXCORR, FIN_MHD, &GRMHD_Z4c_Monolithic::ClearFluxCorrection);
  }

  // C2P on the physical interior requires the updated ADM metric from Z4c
  // and the completed MHD conserved state.
  Add(CONS2PRIMP,
      (FIN_MHD | Z4C_TO_ADM),
      &GRMHD_Z4c_Monolithic::PrimitivesPhysical);

  // Ghost-zone receive polls from NONE (non-blocking MPI_Test)
  Add(RECV_HYD, NONE, &GRMHD_Z4c_Monolithic::ReceiveHydro);

  Add(SETB_HYD,
      (RECV_HYD | SEND_HYD),
      &GRMHD_Z4c_Monolithic::SetBoundariesHydro);

  if (multilevel)
  {
    Add(PROLONG_HYD,
        (SETB_HYD | SEND_HYD),
        &GRMHD_Z4c_Monolithic::Prolongation_Hyd);
    Add(
      PHY_BVAL_HYD, PROLONG_HYD, &GRMHD_Z4c_Monolithic::PhysicalBoundary_Hyd);
  }
  else
  {
    Add(PHY_BVAL_HYD, SETB_HYD, &GRMHD_Z4c_Monolithic::PhysicalBoundary_Hyd);
  }

  // =========================================================================
  // Finalize
  // =========================================================================

  // Ghost-zone C2P needs both the filled ghost data AND the w1 state
  // produced by PrimitivesPhysical::RetainState.
  Add(CONS2PRIMG,
      (PHY_BVAL_HYD | CONS2PRIMP),
      &GRMHD_Z4c_Monolithic::PrimitivesGhosts);

  // Re-couple ADM source terms (GetMatter)
  Add(UPDATE_SRC, CONS2PRIMG, &GRMHD_Z4c_Monolithic::UpdateSource);

  // Diagnostics (last substep only - guarded internally)
  Add(ADM_CONSTR, UPDATE_SRC, &GRMHD_Z4c_Monolithic::ADM_Constraints);
  Add(Z4C_WEYL, UPDATE_SRC, &GRMHD_Z4c_Monolithic::Z4c_Weyl);
  Add(USERWORK, UPDATE_SRC, &GRMHD_Z4c_Monolithic::UserWork);

  // Geometric - no data dependencies on evolved state
  Add(NEW_DT, NONE, &GRMHD_Z4c_Monolithic::NewBlockTimeStep);

  if (adaptive)
    Add(FLAG_AMR, NONE, &GRMHD_Z4c_Monolithic::CheckRefinement);

  // =========================================================================
  // Cleanup - wait on outstanding sends and reset channel flags
  // =========================================================================
  Add(CLEAR_Z4C, Z4C_TO_ADM, &GRMHD_Z4c_Monolithic::ClearZ4c);
  Add(CLEAR_MAININT, PHY_BVAL_HYD, &GRMHD_Z4c_Monolithic::ClearMainInt);
}

// ===========================================================================
// StartupTaskList - per-MeshBlock, per-stage initialisation
// ===========================================================================
void GRMHD_Z4c_Monolithic::StartupTaskList(MeshBlock* pmb, int stage)
{
  if (stage == 1)
  {
    if (integrator == "ssprk5_4")
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in GRMHD_Z4c_Monolithic::StartupTaskList\n"
          << "integrator=" << integrator
          << " is currently incompatible with GRMHD" << std::endl;
      ATHENA_ERROR(msg);
    }

    // Initialize time abscissae
    PrepareStageAbscissae(stage, pmb);

    // -- MHD registers -------------------------------------------------------
    Reconstruction* pr = pmb->precon;
    Hydro* ph          = pmb->phydro;
    ph->u1.ZeroClear();

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

    // -- Z4c registers -------------------------------------------------------
    Z4c* pz4c = pmb->pz4c;
    pz4c->storage.u1.ZeroClear();
  }

  // Post persistent receives for all communication groups used in this DAG.
  // Flux-correction channels are started per-channel to avoid double-starting
  // M1's channel in the shared FluxCorr group.
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

  // Z4c ghost exchange
  pmb->pcomm->StartReceiving(comm::CommGroup::Z4c);

  // MainInt ghost exchange (hydro + field + scalars)
  pmb->pcomm->StartReceiving(comm::CommGroup::MainInt);

  return;
}

// ===========================================================================
//                       MHD COMPUTE BRANCH
// ===========================================================================

TaskStatus GRMHD_Z4c_Monolithic::IntegrateHydroScalars(MeshBlock* pmb,
                                                       int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph             = pmb->phydro;
    PassiveScalars* ps    = pmb->pscalars;
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

    // Ensure update does not dip below floor
    if (pr->enforce_limits_integration)
    {
      ph->EnforceFloorsLimits(ph->u, ps->s, num_enlarge_layer);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::CalculateHydroScalarFlux(MeshBlock* pmb,
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

    // If the block density is everywhere within the floor threshold,
    // skip the high-order pass and go straight to the low-order stencil.
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

    // Per-thread geometry cache for hybridisation
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

    bool skip_limit = false;

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

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::CalculateEMF(MeshBlock* pmb, int stage)
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

// ---------------------------------------------------------------------------
// Flux-correction send/recv for AMR (per-channel to avoid M1 conflicts)
TaskStatus GRMHD_Z4c_Monolithic::SendFluxCorrectionHydro(MeshBlock* pmb,
                                                         int stage)
{
  int hid = pmb->phydro->comm_channel_id;
  if (hid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(hid);
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::SendFluxCorrectionEMF(MeshBlock* pmb,
                                                       int stage)
{
  int fid = pmb->pfield->comm_channel_id;
  if (fid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(fid);
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::ReceiveAndCorrectHydroFlux(MeshBlock* pmb,
                                                            int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  int hid = pmb->phydro->comm_channel_id;
  if (hid >= 0)
  {
    if (!pcomm->channel(hid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(hid).UnpackFluxCorr(nc);
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::ReceiveAndCorrectEMF(MeshBlock* pmb,
                                                      int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  int fid = pmb->pfield->comm_channel_id;
  if (fid >= 0)
  {
    if (!pcomm->channel(fid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(fid).UnpackFluxCorr(nc);
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::SendScalarFlux(MeshBlock* pmb, int stage)
{
  int sid = pmb->pscalars->comm_channel_id;
  if (sid >= 0)
    pmb->pcomm->SendFluxCorrSingleChannel(sid);
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::ReceiveScalarFlux(MeshBlock* pmb, int stage)
{
  comm::CommRegistry* pcomm            = pmb->pcomm;
  const comm::NeighborConnectivity& nc = pcomm->connectivity();

  int sid = pmb->pscalars->comm_channel_id;
  if (sid >= 0)
  {
    if (!pcomm->channel(sid).PollReceiveFluxCorr(nc))
      return TaskStatus::fail;
    pcomm->channel(sid).UnpackFluxCorr(nc);
  }
  return TaskStatus::next;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::AddFluxDivergenceHydroScalars(MeshBlock* pmb,
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

    // Ensure update does not dip below floor
    const int num_enlarge_layer = 0;
    if (pr->enforce_limits_flux_div)
    {
      ph->EnforceFloorsLimits(ph->u, ps->s, num_enlarge_layer);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::AddSourceTermsHydro(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph          = pmb->phydro;
    PassiveScalars* ps = pmb->pscalars;
    Field* pf          = pmb->pfield;
    Coordinates* pc    = pmb->pcoord;

    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pc->AddCoordTermsDivergence(dt_scaled, ph->w, ps->r, pf->bcc, ph->u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::IntegrateField(MeshBlock* pmb, int stage)
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

// ===========================================================================
//                          Z4C COMPUTE BRANCH
// ===========================================================================

TaskStatus GRMHD_Z4c_Monolithic::InitializeZ4cDerivatives(MeshBlock* pmb,
                                                          int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    pz4c->InitializeZ4cDerivatives(pz4c->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::CalculateZ4cRHS(MeshBlock* pmb, int stage)
{
  Z4c* pz4c = pmb->pz4c;

  // Puncture tracker: interpolate shift at puncture position before evolution
  if (stage == 1)
  {
    for (auto ptracker : pmb->pmy_mesh->pz4c_tracker)
    {
      ptracker->InterpolateShift(pmb, pz4c->storage.u);
    }
  }

  if (stage <= nstages)
  {
    pz4c->Z4cRHS(pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

    // Sommerfeld boundary conditions
    pz4c->Z4cBoundaryRHS(
      pz4c->storage.u, pz4c->storage.mat, pz4c->storage.rhs);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::IntegrateZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;

    Real ave_wghts[3];
    ave_wghts[0] = 1.0;
    ave_wghts[1] = stage_wghts[stage - 1].delta;
    ave_wghts[2] = 0.0;
    pz4c->WeightedAve(
      pz4c->storage.u1, pz4c->storage.u, pz4c->storage.u2, ave_wghts);

    ave_wghts[0] = stage_wghts[stage - 1].gamma_1;
    ave_wghts[1] = stage_wghts[stage - 1].gamma_2;
    ave_wghts[2] = stage_wghts[stage - 1].gamma_3;

    pz4c->WeightedAve(
      pz4c->storage.u, pz4c->storage.u1, pz4c->storage.u2, ave_wghts);

    const Real dt_scaled = this->dt_scaled(stage, pmb);
    pz4c->AddZ4cRHS(pz4c->storage.rhs, dt_scaled, pz4c->storage.u);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
// Z4c ghost-zone exchange
TaskStatus GRMHD_Z4c_Monolithic::SendZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pcomm->SendBoundaryBuffers(comm::CommGroup::Z4c);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Monolithic::ReceiveZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    bool done = pmb->pcomm->ReceiveBoundaryBuffers(comm::CommGroup::Z4c);
    if (done)
      return TaskStatus::next;
    return TaskStatus::fail;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Monolithic::SetBoundariesZ4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pcomm->SetBoundaries(comm::CommGroup::Z4c);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::Prolongation_Z4c(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c                 = pmb->pz4c;
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    // Z4c uses module-specific coarse indices (may differ from pmb->cis/cie)
    const int cil = pz4c->mbi.cil, ciu = pz4c->mbi.ciu;
    const int cjl = pz4c->mbi.cjl, cju = pz4c->mbi.cju;
    const int ckl = pz4c->mbi.ckl, cku = pz4c->mbi.cku;
    const int cng = pz4c->mbi.cng;

    pcomm->ProlongateAndApplyPhysicalBCs(comm::CommGroup::Z4c,
                                         t_end,
                                         dt_scaled,
                                         cil,
                                         ciu,
                                         cjl,
                                         cju,
                                         ckl,
                                         cku,
                                         cng);
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::PhysicalBoundary_Z4c(MeshBlock* pmb,
                                                      int stage)
{
  if (stage <= nstages)
  {
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end     = this->t_end(stage, pmb);
    const Real dt_scaled = this->dt_scaled(stage, pmb);

    pcomm->ApplyPhysicalBCs(comm::CommGroup::Z4c, t_end, dt_scaled);
  }
  else
  {
    return TaskStatus::fail;
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::EnforceAlgConstr(MeshBlock* pmb, int stage)
{
  Z4c* pz4c = pmb->pz4c;
  pz4c->AlgConstr(pz4c->storage.u);
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::PrepareZ4cDerivatives(MeshBlock* pmb,
                                                       int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    pz4c->PrepareZ4cDerivatives(pz4c->storage.u);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Monolithic::Z4cToADM(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c = pmb->pz4c;
    pz4c->Z4cToADM(pz4c->storage.u, pz4c->storage.adm);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

#if CCE_ENABLED
TaskStatus GRMHD_Z4c_Monolithic::CCEDump(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  using namespace gra::triggers;
  typedef Triggers::TriggerVariant tvar;
  typedef Triggers::OutputVariant ovar;

  Mesh* pm = pmb->pmy_mesh;

  for (auto cce : pm->pcce)
  {
    const Real dt_cce = trgs.GetTrigger_dt(tvar::Z4c_CCE, ovar::user);
    if (dt_cce > 0)
    {
      const Real time = pm->time;
      if (std::fabs(std::fmod(time, dt_cce)) > 1e-12)
        continue;
      cce->Interpolate(pmb);
    }
  }

  return TaskStatus::next;
}
#endif

// ===========================================================================
//                   MHD GHOST-ZONE SEND + C2P
// ===========================================================================

// Conserved-variable ghost send - fires before C2P.
TaskStatus GRMHD_Z4c_Monolithic::SendHydro(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    // Group-level send packs all MainInt channels (hydro u, field b, scalars
    // s). Restriction into coarse buffers is handled internally for
    // multilevel.
    pmb->pcomm->SendBoundaryBuffers(comm::CommGroup::MainInt);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
// Non-blocking clear of flux-correction sends and channel flag reset
// (multilevel only).  Separate from ClearMainInt because flux-correction and
// ghost-zone sends use different CommGroups (FluxCorr vs MainInt).  Uses
// MPI_Test (wait=false) to avoid blocking under the work-stealing lock.
TaskStatus GRMHD_Z4c_Monolithic::ClearFluxCorrection(MeshBlock* pmb, int stage)
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

// ---------------------------------------------------------------------------
// Ghost-zone receive: poll for incoming MainInt data.
TaskStatus GRMHD_Z4c_Monolithic::ReceiveHydro(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    bool done = pmb->pcomm->ReceiveBoundaryBuffers(comm::CommGroup::MainInt);
    if (done)
      return TaskStatus::next;
    return TaskStatus::fail;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
// C2P on the physical interior - requires the ADM metric from Z4c.
TaskStatus GRMHD_Z4c_Monolithic::PrimitivesPhysical(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph             = pmb->phydro;
    Field* pf             = pmb->pfield;
    PassiveScalars* ps    = pmb->pscalars;
    EquationOfState* peos = pmb->peos;

    int il = pmb->is, iu = pmb->ie;
    int jl = pmb->js, ju = pmb->je;
    int kl = pmb->ks, ku = pmb->ke;

    // Recompute cell-centred B from face-centred field on the physical
    // interior so that C2P sees bcc consistent with the current stage.
    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->CalculateCellCenteredField(
        pf->b, pf->bcc, pmb->pcoord, il, iu, jl, ju, kl, ku);
    }

    static const int coarseflag = 0;
    peos->ConservedToPrimitive(ph->u,
                               ph->w1,
                               ph->w,
                               ps->s,
                               ps->r,
                               pf->bcc,
                               pmb->pcoord,
                               il,
                               iu,
                               jl,
                               ju,
                               kl,
                               ku,
                               coarseflag);

    // Update w1 to retain the current primitive state
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::SetBoundariesHydro(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    pmb->pcomm->SetBoundaries(comm::CommGroup::MainInt);
    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::Prolongation_Hyd(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end =
      (stage >= 1) ? this->t_end(stage, pmb) : pmb->pmy_mesh->time;
    const Real dt_scaled = (stage >= 1) ? this->dt_scaled(stage, pmb) : 0.0;

    const int cis = pmb->cis, cie = pmb->cie;
    const int cjs = pmb->cjs, cje = pmb->cje;
    const int cks = pmb->cks, cke = pmb->cke;

    pcomm->ProlongateAndApplyPhysicalBCs(comm::CommGroup::MainInt,
                                         t_end,
                                         dt_scaled,
                                         cis,
                                         cie,
                                         cjs,
                                         cje,
                                         cks,
                                         cke,
                                         NGHOST);

    // Recompute cell-centred B on prolongated ghost slabs
    if (MAGNETIC_FIELDS_ENABLED)
      pmb->CalculateCellCenteredFieldOnProlongedBoundaries();

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

TaskStatus GRMHD_Z4c_Monolithic::PhysicalBoundary_Hyd(MeshBlock* pmb,
                                                      int stage)
{
  if (stage <= nstages)
  {
    Field* pf                 = pmb->pfield;
    comm::CommRegistry* pcomm = pmb->pcomm;

    const Real t_end =
      (stage >= 1) ? this->t_end(stage, pmb) : pmb->pmy_mesh->time;
    const Real dt_scaled = (stage >= 1) ? this->dt_scaled(stage, pmb) : 0.0;

    pcomm->ApplyPhysicalBCs(comm::CommGroup::MainInt, t_end, dt_scaled);

    // Recompute bcc globally (ghost zones now filled by physical BCs above)
    if (MAGNETIC_FIELDS_ENABLED)
    {
      pf->CalculateCellCenteredField(pf->b,
                                     pf->bcc,
                                     pmb->pcoord,
                                     0,
                                     pmb->ncells1 - 1,
                                     0,
                                     pmb->ncells2 - 1,
                                     0,
                                     pmb->ncells3 - 1);
    }

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ===========================================================================
//                           FINALIZE
// ===========================================================================

// C2P on ghost zones - needs both recv'd ghost data AND w1 from phys C2P.
TaskStatus GRMHD_Z4c_Monolithic::PrimitivesGhosts(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Hydro* ph             = pmb->phydro;
    Field* pf             = pmb->pfield;
    PassiveScalars* ps    = pmb->pscalars;
    EquationOfState* peos = pmb->peos;

    int il = 0, iu = pmb->ncells1 - 1;
    int jl = 0, ju = pmb->ncells2 - 1;
    int kl = 0, ku = pmb->ncells3 - 1;

    static const int coarseflag     = 0;
    static const bool skip_physical = true;
    peos->ConservedToPrimitive(ph->u,
                               ph->w1,
                               ph->w,
                               ps->s,
                               ps->r,
                               pf->bcc,
                               pmb->pcoord,
                               il,
                               iu,
                               jl,
                               ju,
                               kl,
                               ku,
                               coarseflag,
                               skip_physical);

    if (pmb->peos->smooth_temperature)
    {
      peos->SmoothTemperatureAndRecompute(ph->w,
                                          ph->w1,
                                          ph->derived_ms,
                                          ps->r,
                                          il,
                                          iu,
                                          jl,
                                          ju,
                                          kl,
                                          ku,
                                          pmb->precon->xorder_use_aux_cs2,
                                          pmb->precon->xorder_use_aux_s);
    }

    // Update w1 to retain the current primitive state
    ph->RetainState(ph->w1, ph->w, il, iu, jl, ju, kl, ku);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
// Re-couple ADM source terms (GetMatter)
TaskStatus GRMHD_Z4c_Monolithic::UpdateSource(MeshBlock* pmb, int stage)
{
  if (stage <= nstages)
  {
    Z4c* pz4c          = pmb->pz4c;
    Hydro* ph          = pmb->phydro;
    Field* pf          = pmb->pfield;
    PassiveScalars* ps = pmb->pscalars;

    pz4c->GetMatter(
      pz4c->storage.mat, pz4c->storage.adm, ph->w, ps->r, pf->bcc);

    return TaskStatus::next;
  }
  return TaskStatus::fail;
}

// ---------------------------------------------------------------------------
TaskStatus GRMHD_Z4c_Monolithic::ADM_Constraints(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Z4c* pz4c = pmb->pz4c;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_ADM_constraints))
  {
    pz4c->ADMConstraints(pz4c->storage.con,
                         pz4c->storage.adm,
                         pz4c->storage.mat,
                         pz4c->storage.u);
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::Z4c_Weyl(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  if (trgs.IsSatisfied(TriggerVariant::Z4c_Weyl))
  {
    pmb->pz4c->Z4cWeyl(
      pmb->pz4c->storage.adm, pmb->pz4c->storage.mat, pmb->pz4c->storage.weyl);
  }
  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::UserWork(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  pmb->UserWorkInLoop();
  pmb->ptracker_extrema_loc->TreatCentreIfLocalMember();

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::NewBlockTimeStep(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  Z4c* pz4c = pmb->pz4c;
  pz4c->NewBlockTimeStep();

  return TaskStatus::next;
}

TaskStatus GRMHD_Z4c_Monolithic::CheckRefinement(MeshBlock* pmb, int stage)
{
  if (stage != nstages)
    return TaskStatus::next;

  pmb->pmr->CheckRefinementCondition();
  return TaskStatus::next;
}

// ===========================================================================
//                           CLEANUP
// ===========================================================================

// Non-blocking clear of outstanding Z4c ghost sends and channel flag reset.
// Uses MPI_Test (wait=false) so that the per-block work-stealing lock is
// released immediately when sends are still in flight.  This prevents a
// deadlock where every thread blocks inside MPI_Wait while holding a block
// lock, leaving no thread free to drive MPI_Test on the receive side of
// other blocks -- which is required for the remote MPI_Wait to complete.
TaskStatus GRMHD_Z4c_Monolithic::ClearZ4c(MeshBlock* pmb, int stage)
{
  if (!pmb->pcomm->ClearBoundary(
        comm::CommGroup::Z4c, comm::CommTarget::All, false))
    return TaskStatus::fail;
  return TaskStatus::next;
}

// Non-blocking clear of outstanding MainInt ghost sends and channel flag
// reset. See ClearZ4c comment.
TaskStatus GRMHD_Z4c_Monolithic::ClearMainInt(MeshBlock* pmb, int stage)
{
  if (!pmb->pcomm->ClearBoundary(
        comm::CommGroup::MainInt, comm::CommTarget::All, false))
    return TaskStatus::fail;
  return TaskStatus::next;
}

//
// :D
//
