//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file scalars.cpp
//  \brief implementation of functions in class PassiveScalars

// C headers

// C++ headers
#include <algorithm>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../comm/amr_registry.hpp"
#include "../comm/amr_spec.hpp"
#include "../comm/comm_registry.hpp"
#include "../comm/comm_spec.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "scalars.hpp"

// constructor, initializes data structures and parameters

PassiveScalars::PassiveScalars(MeshBlock* pmb, ParameterInput* pin)
    : s(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      s1(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      r(NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
      s_flux{ { NSCALARS, pmb->ncells3, pmb->ncells2, pmb->ncells1 + 1 },
              { NSCALARS,
                pmb->ncells3,
                pmb->ncells2 + 1,
                pmb->ncells1,
                (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated
                                   : AthenaArray<Real>::DataStatus::empty) },
              { NSCALARS,
                pmb->ncells3 + 1,
                pmb->ncells2,
                pmb->ncells1,
                (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated
                                   : AthenaArray<Real>::DataStatus::empty) } },
      coarse_s_(
        NSCALARS,
        pmb->ncc3,
        pmb->ncc2,
        pmb->ncc1,
        (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated
                                   : AthenaArray<Real>::DataStatus::empty)),
      pmy_block(pmb)
{
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh* pm = pmy_block->pmy_mesh;

  // If user-requested time integrator is type 3S*, allocate additional memory
  // registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" ||
      (pmb->precon->xorder_use_fb && pmb->precon->xorder_use_dmp))
    // future extension may add "int nregister" to Hydro class
    s2.NewAthenaArray(NSCALARS, nc3, nc2, nc1);

  // Register with AMR redistribution system (new comm layer).
  if (pm->multilevel)
  {
    comm::AMRSpec amr;
    amr.label       = "scalars_s";
    amr.var         = &s;
    amr.coarse_var  = &coarse_s_;
    amr.nvar        = NSCALARS;
    amr.sampling    = comm::Sampling::CC;
    amr.group       = comm::AMRGroup::Main;
    amr.prolong_op  = comm::ProlongOp::MinmodLinear;
    amr.restrict_op = comm::RestrictOp::VolumeWeighted;
    pmb->pamr->Register(amr);
  }

  // Register passive scalars with the new comm system.
  // All components are scalar (no parity sign flips).
  {
    comm::CommSpec spec;
    spec.label       = "scalars_s";
    spec.var         = &s;
    spec.coarse_var  = &coarse_s_;
    spec.nvar        = NSCALARS;
    spec.sampling    = comm::Sampling::CC;
    spec.targets     = comm::CommTarget::All;
    spec.group       = comm::CommGroup::MainInt;
    spec.prolong_op  = comm::ProlongOp::MinmodLinear;
    spec.restrict_op = comm::RestrictOp::VolumeWeighted;
    comm::SetPhysicalBCFromBlockBCs(spec, pmb->nc());
    // component_groups left empty - all scalars, no sign flips
    // Flux correction: area-weighted restricted fluxes overwrite coarse
    // fluxes.
    if (pm->multilevel)
    {
      spec.flx_cc[0]  = &s_flux[0];
      spec.flx_cc[1]  = &s_flux[1];
      spec.flx_cc[2]  = &s_flux[2];
      spec.flcor_mode = comm::FluxCorrMode::OverwriteFromFiner;
      spec.flux_group = comm::CommGroup::FluxCorr;
    }
    comm_channel_id = pmb->pcomm->Register(spec);
  }

  // Allocate memory for scratch arrays
  dflx_.NewAthenaArray(NSCALARS, nc1);
}
