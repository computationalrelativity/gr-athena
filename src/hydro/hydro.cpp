//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

// C headers

// C++ headers
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

// Athena++ headers
#include "../comm/amr_registry.hpp"
#include "../comm/amr_spec.hpp"
#include "../comm/comm_registry.hpp"
#include "../comm/comm_spec.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "hydro.hpp"

// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlock* pmb, ParameterInput* pin)
    : pmy_block(pmb),
      u(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      w(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      u1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      w1(NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      derived_ms(NDRV_HYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      derived_int(NIDRV_HYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      // C++11: nested brace-init-list in Hydro member initializer list =
      // aggregate init. of flux[3] array --> direct list init. of each array
      // element --> direct init. via constructor overload resolution of
      // non-aggregate class type AthenaArray<Real>
      flux{ { NHYDRO, pmb->ncells3, pmb->ncells2, pmb->ncells1 + 1 },
            { NHYDRO,
              pmb->ncells3,
              pmb->ncells2 + 1,
              pmb->ncells1,
              (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated
                                 : AthenaArray<Real>::DataStatus::empty) },
            { NHYDRO,
              pmb->ncells3 + 1,
              pmb->ncells2,
              pmb->ncells1,
              (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated
                                 : AthenaArray<Real>::DataStatus::empty) } },
      coarse_cons_(
        NHYDRO,
        pmb->ncc3,
        pmb->ncc2,
        pmb->ncc1,
        (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated
                                   : AthenaArray<Real>::DataStatus::empty))
{
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh* pm = pmy_block->pmy_mesh;

  flux_reconstruction =
    pin->GetOrAddBoolean("hydro", "flux_reconstruction", false);

  // Riemann solver method (runtime selection)
  {
    std::string rsolver_str = pin->GetOrAddString("hydro", "rsolver", "llf");
    if (rsolver_str == "llf")
    {
      rsolver_method_ = RSolverMethod::llf;
    }
    else if (rsolver_str == "hlle")
    {
      rsolver_method_ = RSolverMethod::hlle;
    }
    else
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in Hydro constructor" << std::endl
          << "[hydro] rsolver=" << rsolver_str
          << " not a valid choice (llf, hlle)" << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  opt_excision.alpha_threshold =
    pin->GetOrAddReal("excision", "alpha_threshold", -1.0);
  opt_excision.horizon_based =
    pin->GetOrAddBoolean("excision", "horizon_based", false);
  opt_excision.horizon_factor =
    pin->GetOrAddReal("excision", "horizon_factor", 1.0);

  opt_excision.hybrid_hydro =
    pin->GetOrAddBoolean("excision", "hybrid_hydro", false);

  opt_excision.hybrid_fac_min_alpha =
    pin->GetOrAddReal("excision", "hybrid_fac_min_alpha", 1.5);

  opt_excision.use_taper =
    pin->GetOrAddBoolean("excision", "use_taper", false);

  opt_excision.excise_hydro_damping =
    pin->GetOrAddBoolean("excision", "excise_hydro_damping", false);

  opt_excision.hydro_damping_factor =
    pin->GetOrAddReal("excision", "hydro_damping_factor", 0.69);

  opt_excision.excise_flux =
    pin->GetOrAddBoolean("excision", "excise_flux", true);

  opt_excision.excise_c2p =
    pin->GetOrAddBoolean("excision", "excise_c2p", true);

  if (opt_excision.use_taper || opt_excision.excise_hydro_damping)
  {
    excision_mask.NewAthenaArray(nc3, nc2, nc1);
    excision_mask.Fill(1);
  }

  opt_excision.taper_pow = pin->GetOrAddReal("excision", "taper_pow", 1.0);

  opt_excision.taper_min = pin->GetOrAddReal("excision", "taper_min", 0.0);

  opt_excision.taper_dt_response =
    pin->GetOrAddReal("excision", "taper_dt_response", 0.0);

  opt_excision.excise_hydro_freeze_evo =
    pin->GetOrAddBoolean("excision", "excise_hydro_freeze_evo", false);

  opt_excision.excise_hydro_taper =
    pin->GetOrAddBoolean("excision", "excise_hydro_taper", false);

  if (pmb->precon->xorder_use_fb)
  {
    fallback_mask.NewAthenaArray(nc3, nc2, nc1);
  }

  // If user-requested time integrator is type 3S*, allocate additional memory
  // registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" ||
      (pmb->precon->xorder_use_fb && pmb->precon->xorder_use_dmp))
  {
    // future extension may add "int nregister" to Hydro class
    u2.NewAthenaArray(NHYDRO, nc3, nc2, nc1);
  }

  // Register with AMR redistribution system (new comm layer).
  if (pm->multilevel)
  {
    comm::AMRSpec amr;
    amr.label       = "hydro_cons";
    amr.var         = &u;
    amr.coarse_var  = &coarse_cons_;
    amr.nvar        = NHYDRO;
    amr.sampling    = comm::Sampling::CC;
    amr.group       = comm::AMRGroup::Main;
    amr.prolong_op  = comm::ProlongOp::MinmodLinear;
    amr.restrict_op = comm::RestrictOp::VolumeWeighted;
    pmb->pamr->Register(amr);
  }

  // Register hydro conserved variables with the new comm system.
  // Communicates u (conserved) with coarse_cons_ as the coarse buffer.
  // Parity: {Scalar,1} for D, {Vector,3} for S_d, {Scalar,1} for tau.
  {
    comm::CommSpec spec;
    spec.label       = "hydro_cons";
    spec.var         = &u;
    spec.coarse_var  = &coarse_cons_;
    spec.nvar        = NHYDRO;
    spec.sampling    = comm::Sampling::CC;
    spec.targets     = comm::CommTarget::All;
    spec.group       = comm::CommGroup::MainInt;
    spec.prolong_op  = comm::ProlongOp::MinmodLinear;
    spec.restrict_op = comm::RestrictOp::VolumeWeighted;
    comm::SetPhysicalBCFromBlockBCs(spec, pmb->nc());
    spec.component_groups = {
      { comm::GeomType::Scalar, 1 },  // D
      { comm::GeomType::Vector, 3 },  // S_d_{1,2,3}
      { comm::GeomType::Scalar, 1 }   // tau
    };
    // Flux correction: area-weighted restricted fluxes overwrite coarse fluxes
    // at fine/coarse interfaces.  Only active for AMR/SMR.
    if (pm->multilevel)
    {
      spec.flx_cc[0]  = &flux[0];
      spec.flx_cc[1]  = &flux[1];
      spec.flx_cc[2]  = &flux[2];
      spec.flcor_mode = comm::FluxCorrMode::OverwriteFromFiner;
      spec.flux_group = comm::CommGroup::FluxCorr;
    }
    comm_channel_id = pmb->pcomm->Register(spec);
#if M1_ENABLED
    // M1 re-scatter needs hydro ghost exchange without B-field overhead.
    pmb->pcomm->AddToGroup(comm_channel_id, comm::CommGroup::M1Rescatter);
#endif
  }

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  dxw_.NewAthenaArray(nc1);

  // storage for reconstruction of primitives
  wl_.NewAthenaArray(NWAVE, nc1);
  wr_.NewAthenaArray(NWAVE, nc1);
  wlb_.NewAthenaArray(NWAVE, nc1);

#if FLUID_ENABLED
  // storage for reconstruction of passive scalars
  rl_.NewAthenaArray(NSCALARS, nc1);
  rr_.NewAthenaArray(NSCALARS, nc1);
  rlb_.NewAthenaArray(NSCALARS, nc1);
#endif

  if (pmy_block->precon->xorder_use_auxiliaries)
  {
    al_.NewAthenaArray(NDRV_HYDRO, nc1);
    ar_.NewAthenaArray(NDRV_HYDRO, nc1);
    alb_.NewAthenaArray(NDRV_HYDRO, nc1);
  }

  dflx_.NewAthenaArray(NHYDRO, nc1);

  UserTimeStep_ = pmb->pmy_mesh->UserTimeStep_;

  // scratches for rsolver ----------------------------------------------------
#if Z4C_ENABLED
  const int nn1 = pmy_block->nverts1;

  sqrt_detgamma_.NewAthenaTensor(nn1);

  alpha_.NewAthenaTensor(nn1);
  beta_u_.NewAthenaTensor(nn1);
  gamma_dd_.NewAthenaTensor(nn1);
  gamma_uu_.NewAthenaTensor(nn1);

  chi_.NewAthenaTensor(nn1);

  w_v_u_l_.NewAthenaTensor(nn1);
  w_v_u_r_.NewAthenaTensor(nn1);

  w_norm2_v_l_.NewAthenaTensor(nn1);
  w_norm2_v_r_.NewAthenaTensor(nn1);

  lambda_p_l.NewAthenaTensor(nn1);
  lambda_m_l.NewAthenaTensor(nn1);
  lambda_p_r.NewAthenaTensor(nn1);
  lambda_m_r.NewAthenaTensor(nn1);
  lambda.NewAthenaTensor(nn1);

  w_util_d_l_.NewAthenaTensor(nn1);
  w_util_d_r_.NewAthenaTensor(nn1);

  W_l_.NewAthenaTensor(nn1);
  W_r_.NewAthenaTensor(nn1);

  w_hrho_l_.NewAthenaTensor(nn1);
  w_hrho_r_.NewAthenaTensor(nn1);

  cons_l_.NewAthenaTensor(nn1);
  cons_r_.NewAthenaTensor(nn1);

  flux_l_.NewAthenaTensor(nn1);
  flux_r_.NewAthenaTensor(nn1);

#if MAGNETIC_FIELDS_ENABLED
  oo_W_l_.NewAthenaTensor(nn1);
  oo_W_r_.NewAthenaTensor(nn1);

  w_v_d_l_.NewAthenaTensor(nn1);
  w_v_d_r_.NewAthenaTensor(nn1);

  alpha_w_vtil_u_l_.NewAthenaTensor(nn1);
  alpha_w_vtil_u_r_.NewAthenaTensor(nn1);

  beta_d_.NewAthenaTensor(nn1);

  q_scB_u_l_.NewAthenaTensor(nn1);
  q_scB_u_r_.NewAthenaTensor(nn1);

  b0_l_.NewAthenaTensor(nn1);
  b0_r_.NewAthenaTensor(nn1);

  b2_l_.NewAthenaTensor(nn1);
  b2_r_.NewAthenaTensor(nn1);

  bi_u_l_.NewAthenaTensor(nn1);
  bi_u_r_.NewAthenaTensor(nn1);

  bi_d_l_.NewAthenaTensor(nn1);
  bi_d_r_.NewAthenaTensor(nn1);
#endif  // MAGNETIC_FIELDS_ENABLED

#endif
}

//----------------------------------------------------------------------------------------
//! \fn Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx,
//! Real dt)
//  \brief Calculate the weighting factor for the constrained transport method

Real Hydro::GetWeightForCT(Real dflx, Real rhol, Real rhor, Real dx, Real dt)
{
  Real v_over_c = (1024.0) * dt * dflx / (dx * (rhol + rhor));
  Real tmp_min  = std::min(static_cast<Real>(0.5), v_over_c);
  return 0.5 + std::max(static_cast<Real>(-0.5), tmp_min);
}